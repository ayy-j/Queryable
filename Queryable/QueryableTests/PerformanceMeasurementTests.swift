import XCTest
import CoreML
import CryptoKit
import Darwin
import Metal
#if canImport(UIKit)
import UIKit
#endif
@testable import Queryable

/// Opt-in component benchmark. Run with tools/measure-performance.py.
/// Never reads Photos or the saved app index.
final class PerformanceMeasurementTests: XCTestCase {
    func testModelPerformance() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["QUERYABLE_BENCHMARK"] == "1" else {
            throw XCTSkip("Opt in with tools/measure-performance.py")
        }
        let json = try XCTUnwrap(environment["QUERYABLE_BENCHMARK_CONFIG"])
        let config = try JSONDecoder().decode(BenchmarkConfiguration.self, from: Data(json.utf8))
        try config.validate()
        #if targetEnvironment(simulator)
        guard !config.includeSearch else {
            throw BenchmarkError.invalid("Use --model-only on Simulator; GPU measurements require a phone or Mac")
        }
        #endif
        let spec = config.modelID == "mobileclip2-s4" ? EmbeddingModelSpec.mobileCLIP2S4 : .mobileCLIPS2
        var report = BenchmarkReport(configuration: config, spec: spec)
        defer {
            do {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                let data = try encoder.encode(report)
                if let path = environment["QUERYABLE_BENCHMARK_REPORT_PATH"] {
                    try data.write(to: URL(fileURLWithPath: path), options: .atomic)
                }
                let attachment = XCTAttachment(data: data, uniformTypeIdentifier: "public.json")
                attachment.name = "queryable-performance.json"
                attachment.lifetime = .keepAlways
                add(attachment)
            } catch { XCTFail("Could not attach performance report: \(error)") }
        }
        let sampler = FootprintSampler()
        sampler.start()
        defer { sampler.stop() }
        do {
            let resources: URL
            if let path = environment["QUERYABLE_BENCHMARK_RESOURCES"] {
                resources = URL(fileURLWithPath: path)
            } else {
                resources = try XCTUnwrap(Bundle.main.url(forResource: "CoreMLModels", withExtension: nil))
            }
            let missing = spec.missingArtifacts(resourcesAt: resources)
            guard missing.isEmpty else { throw BenchmarkError.invalid("Missing model artifacts: \(missing.joined(separator: ", "))") }
            var search: GPUSimilaritySearch?
            if config.includeSearch {
                guard let gpu = GPUSimilaritySearch(embeddingDimension: spec.embeddingDimension) else {
                    throw BenchmarkError.invalid("Metal search is unavailable on this destination")
                }
                search = gpu
            }
            var imageEncoder: ImgEncoder?
            report.metrics["image_model_load"] = try recordMeasurement(samples: 1, sampler: sampler) {
                imageEncoder = try ImgEncoder(resourcesAt: resources, spec: spec, configuration: config.modelConfiguration())
            }
            let image = try XCTUnwrap(imageEncoder)
            report.actualImageComputeUnits = computeName(image.model.configuration.computeUnits)
            var textEncoder: TextEncoder?
            report.metrics["text_model_load"] = try recordMeasurement(samples: 1, sampler: sampler) {
                textEncoder = try TextEncoder(resourcesAt: resources, spec: spec, configuration: config.modelConfiguration())
            }
            let text = try XCTUnwrap(textEncoder)
            report.actualTextComputeUnits = computeName(text.model.configuration.computeUnits)
            let fixtures = try makeImages()
            let batch = (0..<config.batchSize).map { fixtures[$0 % fixtures.count] }
            let prompts = ["a photo of a cat", "a mountain beside a lake", "people walking in a city", "a red car on a road"]
            func encodeImages(_ images: [UIImage]) throws {
                let results = try image.encodeBatch(images: images).map(ImgEncoder.detachFromIOSurface)
                guard results.count == images.count else { throw BenchmarkError.invalid("Incomplete image batch") }
                for result in results { try validateVector(MLShapedArray<Float32>(converting: result), dimension: spec.embeddingDimension) }
            }
            report.metrics["image_first_batch"] = try recordMeasurement(samples: 1, workItems: config.batchSize, sampler: sampler) { try encodeImages(batch) }
            report.metrics["text_first_query"] = try recordMeasurement(samples: 1, sampler: sampler) {
                try validateVector(text.computeTextEmbedding(prompt: prompts[0]), dimension: spec.embeddingDimension)
            }
            for _ in 0..<config.warmups { try autoreleasepool { try encodeImages(batch) } }
            report.metrics["image_batch"] = try recordMeasurement(samples: config.samples, workItems: config.batchSize, sampler: sampler) { try encodeImages(batch) }
            report.metrics["image_batch_1"] = try recordMeasurement(samples: config.samples, sampler: sampler) { try encodeImages([fixtures[0]]) }
            var promptIndex = 0
            func query() throws -> MLShapedArray<Float32> {
                defer { promptIndex += 1 }
                let result = try text.computeTextEmbedding(prompt: prompts[promptIndex % prompts.count])
                try validateVector(result, dimension: spec.embeddingDimension)
                return result
            }
            for _ in 0..<config.warmups { _ = try autoreleasepool { try query() } }
            report.metrics["text_query"] = try recordMeasurement(samples: config.samples, sampler: sampler) { _ = try query() }
            let fixedQuery = try query()
            if let gpu = search {
              for count in config.indexSizes {
                let vectors = try makeVectors(count: count, dimension: spec.embeddingDimension)
                report.metrics["index_build_\(count)"] = try recordMeasurement(samples: 1, workItems: count, sampler: sampler) { try gpu.buildIndex(from: vectors) }
                func rank(_ embedding: MLShapedArray<Float32>) throws {
                    let scores = try gpu.search(queryEmbedding: embedding)
                    guard scores.count == count, scores.values.allSatisfy({ $0.isFinite }) else {
                        throw BenchmarkError.invalid("Search returned incomplete or nonfinite scores")
                    }
                    // Match the app's full sort, not an unrealistically cheaper top-k.
                    let top = Array(scores.sorted { $0.value > $1.value }.prefix(min(config.topK, count)))
                    guard top.count == min(config.topK, count) else { throw BenchmarkError.invalid("Incomplete top-k") }
                }
                report.metrics["search_first_\(count)"] = try recordMeasurement(samples: 1, sampler: sampler) { try rank(fixedQuery) }
                for _ in 0..<config.warmups { try autoreleasepool { try rank(fixedQuery) } }
                report.metrics["search_\(count)"] = try recordMeasurement(samples: config.samples, sampler: sampler) { try rank(fixedQuery) }
                report.metrics["text_search_\(count)"] = try recordMeasurement(samples: config.samples, sampler: sampler) { try rank(query()) }
              }
            }
            sampler.stop()
            report.memory = sampler.snapshot()
            report.thermalEnd = thermalName()
            // Hash AFTER timing: hashing first would warm disk caches.
            report.artifacts = try spec.requiredArtifactNames.map { try fingerprint(resources.appendingPathComponent($0)) }
            report.status = "complete"
        } catch {
            report.failure = String(describing: error)
            report.memory = sampler.snapshot()
            report.thermalEnd = thermalName()
            throw error
        }
    }
}

private struct BenchmarkConfiguration: Codable {
    let modelID: String
    let computeUnits: String
    let samples: Int
    let warmups: Int
    let batchSize: Int
    let indexSizes: [Int]
    let topK: Int
    let includeSearch: Bool

    func validate() throws {
        guard ["mobileclip2-s4", "mobileclip-s2"].contains(modelID),
              ["all", "cpuOnly", "cpuAndGPU", "cpuAndNeuralEngine"].contains(computeUnits),
              (3...1000).contains(samples), (1...100).contains(warmups),
              (1...128).contains(batchSize), (1...1000).contains(topK),
              !indexSizes.isEmpty, Set(indexSizes).count == indexSizes.count,
              indexSizes.allSatisfy({ (1...100_000).contains($0) }) else {
            throw BenchmarkError.invalid("Invalid benchmark configuration")
        }
    }

    func modelConfiguration() -> MLModelConfiguration {
        let configuration = MLModelConfiguration()
        switch computeUnits {
        case "cpuOnly": configuration.computeUnits = .cpuOnly
        case "cpuAndGPU": configuration.computeUnits = .cpuAndGPU
        case "cpuAndNeuralEngine": configuration.computeUnits = .cpuAndNeuralEngine
        default: configuration.computeUnits = .all
        }
        return configuration
    }
}

private struct BenchmarkReport: Encodable {
    let schemaVersion = 1
    let workloadVersion = "generated-rgba-1024x768-and-lcg-vectors-v1"
    let startedAt = ISO8601DateFormatter().string(from: Date())
    let configuration: BenchmarkConfiguration
    let modelID: String
    let modelRevision: String
    let modelContract: String
    let embeddingDimension: Int
    let os = ProcessInfo.processInfo.operatingSystemVersionString
    let hardware = hardwareIdentifier()
    let simulatorModel = ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"]
    let platform: String = {
        #if targetEnvironment(simulator)
        return "simulator"
        #elseif os(macOS)
        return "macOS-native-components"
        #else
        return ProcessInfo.processInfo.isiOSAppOnMac ? "iOS-app-on-Mac" : "iOS-device"
        #endif
    }()
    let physicalMemoryBytes = ProcessInfo.processInfo.physicalMemory
    let gpuName = MTLCreateSystemDefaultDevice()?.name
    let thermalStart = thermalName()
    var thermalEnd: String?
    var actualImageComputeUnits: String?
    var actualTextComputeUnits: String?
    var status = "failed"
    var failure: String?
    var metrics: [String: Measurement] = [:]
    var memory: FootprintSnapshot?
    var artifacts: [ArtifactFingerprint] = []

    init(configuration: BenchmarkConfiguration, spec: EmbeddingModelSpec) {
        self.configuration = configuration
        modelID = spec.modelID
        modelRevision = spec.revision
        modelContract = spec.compatibilityIdentity
        embeddingDimension = spec.embeddingDimension
    }
}

private struct Measurement: Encodable {
    let samplesMilliseconds: [Double]
    let medianMilliseconds: Double
    let p95Milliseconds: Double
    let minMilliseconds: Double
    let maxMilliseconds: Double
    let meanMilliseconds: Double
    let standardDeviationMilliseconds: Double
    let workItemsPerSample: Int
    let workItemsPerSecond: Double
    let footprintBeforeBytes: UInt64?
    let footprintAfterBytes: UInt64?
    let thermalBefore: String
    let thermalAfter: String

    init(samples: [Double], workItems: Int, before: UInt64?, after: UInt64?, thermalBefore: String) {
        samplesMilliseconds = samples
        let sorted = samples.sorted()
        let middle = sorted.count / 2
        medianMilliseconds = sorted.count.isMultiple(of: 2) ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
        p95Milliseconds = sorted[max(0, Int(ceil(Double(sorted.count) * 0.95)) - 1)]
        minMilliseconds = sorted.first!
        maxMilliseconds = sorted.last!
        let mean = samples.reduce(0, +) / Double(samples.count)
        meanMilliseconds = mean
        standardDeviationMilliseconds = sqrt(samples.reduce(0) { $0 + pow($1 - mean, 2) } / Double(samples.count))
        workItemsPerSample = workItems
        workItemsPerSecond = Double(workItems * samples.count) * 1000 / samples.reduce(0, +)
        footprintBeforeBytes = before
        footprintAfterBytes = after
        self.thermalBefore = thermalBefore
        thermalAfter = thermalName()
    }
}

private func recordMeasurement(samples: Int, workItems: Int = 1, sampler: FootprintSampler, operation: () throws -> Void) throws -> Measurement {
    let before = sampler.sample()
    let thermal = thermalName()
    var times: [Double] = []
    for _ in 0..<samples {
        let start = DispatchTime.now().uptimeNanoseconds
        try autoreleasepool(invoking: operation)
        times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000)
    }
    return Measurement(samples: times, workItems: workItems, before: before, after: sampler.sample(), thermalBefore: thermal)
}

private struct FootprintSnapshot: Encodable {
    let sampleIntervalMilliseconds = 20
    let successfulSamples: Int
    let baselineBytes: UInt64?
    let sampledPeakBytes: UInt64?
    let lastBytes: UInt64?
    let thermalStates: [String]
}

private final class FootprintSampler {
    private let queue = DispatchQueue(label: "queryable.benchmark.memory", qos: .utility)
    private let lock = NSLock()
    private var timer: DispatchSourceTimer?
    private var baseline: UInt64?
    private var peak: UInt64?
    private var last: UInt64?
    private var count = 0
    private var thermalStates = Set<String>()

    func start() {
        _ = sample()
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now(), repeating: .milliseconds(20))
        timer.setEventHandler { [weak self] in _ = self?.sample() }
        self.timer = timer
        timer.resume()
    }
    func stop() {
        timer?.cancel()
        timer = nil
        queue.sync {}
    }
    @discardableResult func sample() -> UInt64? {
        var info = task_vm_info_data_t()
        var size = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(size)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &size)
            }
        }
        let footprint: UInt64? = result == KERN_SUCCESS ? info.phys_footprint : nil
        lock.lock()
        defer { lock.unlock() }
        thermalStates.insert(thermalName())
        if let footprint {
            count += 1
            if baseline == nil { baseline = footprint }
            last = footprint
            peak = max(peak ?? 0, footprint)
        }
        return footprint
    }
    func snapshot() -> FootprintSnapshot {
        lock.lock()
        defer { lock.unlock() }
        return FootprintSnapshot(successfulSamples: count, baselineBytes: baseline, sampledPeakBytes: peak, lastBytes: last, thermalStates: thermalStates.sorted())
    }
}

private func validateVector(_ vector: MLShapedArray<Float32>, dimension: Int) throws {
    guard vector.scalarCount == dimension, vector.scalars.allSatisfy({ $0.isFinite }),
          vector.scalars.contains(where: { $0 != 0 }) else { throw BenchmarkError.invalid("Invalid model output") }
}

private func makeImages() throws -> [UIImage] {
    try (0..<4).map { seed in
        let width = 1024, height = 768
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * width + x) * 4
                pixels[offset] = UInt8((x + seed * 31) % 256)
                pixels[offset + 1] = UInt8((y + seed * 67) % 256)
                pixels[offset + 2] = UInt8((x / 8 + y / 8 + seed * 13) % 256)
            }
        }
        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
              let cgImage = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                                    bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                                    provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
            throw BenchmarkError.invalid("Could not generate benchmark images")
        }
        return UIImage(cgImage: cgImage)
    }
}

private func makeVectors(count: Int, dimension: Int) throws -> [String: MLMultiArray] {
    var state: UInt64 = 0x515545525941424c
    var result = [String: MLMultiArray](minimumCapacity: count)
    for row in 0..<count {
        let vector = try MLMultiArray(shape: [1, NSNumber(value: dimension)], dataType: .float32)
        let pointer = vector.dataPointer.assumingMemoryBound(to: Float32.self)
        for column in 0..<dimension {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            pointer[column] = Float(state >> 40) / 16_777_216 - 0.5
        }
        result["synthetic-\(row)"] = vector
    }
    return result
}

private struct ArtifactFingerprint: Encodable {
    let name: String
    let sha256: String
    let bytes: UInt64
}

private func fingerprint(_ url: URL) throws -> ArtifactFingerprint {
    let manager = FileManager.default
    var files = [url]
    if try url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true {
        guard let enumerator = manager.enumerator(at: url, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]) else {
            throw BenchmarkError.invalid("Cannot enumerate model artifact")
        }
        files = try enumerator.compactMap { $0 as? URL }.filter {
            try $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true && $0.lastPathComponent != ".DS_Store"
        }.sorted { $0.path < $1.path }
    }
    var hasher = SHA256()
    var bytes: UInt64 = 0
    for file in files {
        let size = UInt64(try file.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0)
        let relative = file == url ? url.lastPathComponent : String(file.path.dropFirst(url.path.count + 1))
        hasher.update(data: Data("\(relative.utf8.count):\(relative):\(size):".utf8))
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty { hasher.update(data: data) }
        bytes += size
    }
    return ArtifactFingerprint(name: url.lastPathComponent, sha256: hasher.finalize().map { String(format: "%02x", $0) }.joined(), bytes: bytes)
}

private func hardwareIdentifier() -> String {
    #if os(macOS)
    var length = 0
    if sysctlbyname("hw.model", nil, &length, nil, 0) == 0, length > 0 {
        var bytes = [UInt8](repeating: 0, count: length)
        let result = bytes.withUnsafeMutableBytes { buffer in
            sysctlbyname("hw.model", buffer.baseAddress, &length, nil, 0)
        }
        if result == 0 { return String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self) }
    }
    #endif
    var system = utsname()
    uname(&system)
    return withUnsafeBytes(of: &system.machine) { bytes in String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self) }
}

private func computeName(_ units: MLComputeUnits) -> String {
    switch units {
    case .all: return "all"
    case .cpuOnly: return "cpuOnly"
    case .cpuAndGPU: return "cpuAndGPU"
    case .cpuAndNeuralEngine: return "cpuAndNeuralEngine"
    @unknown default: return "unknown"
    }
}

private func thermalName() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

private enum BenchmarkError: Error { case invalid(String) }
