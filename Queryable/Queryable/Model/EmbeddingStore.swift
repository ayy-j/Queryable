//
//  EmbeddingStore.swift
//  Queryable
//
//  Crash-safe immutable record segments with an atomic manifest/checkpoint commit.
//  See docs/qemb-v2-format.md for the wire format, migration, and recovery protocol.
//

import Foundation
import CoreML
import CryptoKit
import Darwin

enum EmbeddingStoreError: Error, LocalizedError {
    case incompatible
    case corrupt
    case staleGeneration
    case invalidRecords
    /// The store has not been created for the active model spec yet.
    case notReady
    /// A transaction could not be persisted.
    case writeFailed
    /// The header metadata does not fit the on-disk length field.
    case metadataTooLarge(Int)

    var errorDescription: String? {
        switch self {
        case .incompatible: return "The saved index belongs to a different model contract. It has been preserved."
        case .corrupt: return "The saved index could not be verified. It has been preserved; restart indexing to build a replacement."
        case .staleGeneration: return "This operation belongs to an earlier indexing generation."
        case .invalidRecords: return "The index contains an invalid identifier or embedding vector."
        case .notReady: return "The index is not ready."
        case .writeFailed: return "The index could not be saved. Reload its committed checkpoint before retrying."
        case .metadataTooLarge: return "The model metadata is too large to save."
        }
    }
}

/// Every transaction and load is serialized across store instances. Snapshot arrays are
/// fresh values owned by the caller. Production callers do not install fault hooks.
struct EmbeddingStoreSnapshot {
    let embeddings: [String: MLMultiArray]
    let generation: UUID
    let revision: UInt64
    let checkpoint: Data?
}

class EmbeddingStore: @unchecked Sendable {
    enum Boundary: CaseIterable {
        case segmentWritten, segmentSynced, segmentDirectorySynced
        case manifestWritten, manifestSynced, manifestRenamed, directorySynced, cleanupFinished
    }
    /// Test-only deterministic interruption hook; throw to simulate a stopped writer.
    var faultInjector: ((Boundary) throws -> Void)?
    private static let transactionLock = NSRecursiveLock()
    private let selectedGeneration: UUID?
    private struct Segment: Codable {
        let name: String
        let digest: String
        let count: UInt64
        let deleting: [String]
    }
    private struct Manifest: Codable {
        let version: Int
        let compatibilityIdentity: String
        let metadata: HeaderMetadata
        let generation: UUID
        let revision: UInt64
        let segments: [Segment]
        let checkpoint: Data?
    }
    private struct Envelope: Codable { let payload: Data; let digest: String }
    private var metadata: HeaderMetadata {
        HeaderMetadata(modelID: spec.modelID, checkpointHash: checkpointHash,
                       dimension: spec.embeddingDimension, scalarType: spec.storageScalarType.rawValue,
                       preprocessingFingerprint: spec.preprocessingFingerprint, normalized: spec.normalizeEmbeddings)
    }
    private func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    private var transactionDirectory: URL {
        let encoder = JSONEncoder(); encoder.outputFormatting = .sortedKeys
        let identity = digest((try! encoder.encode(metadata)) + Data(spec.compatibilityIdentity.utf8))
        return baseDir.appendingPathComponent("qemb-transactions/\(identity)/\(selectedGeneration?.uuidString ?? "default")", isDirectory: true)
    }
    private var manifestURL: URL { transactionDirectory.appendingPathComponent("manifest.json") }

    private func readManifest() throws -> Manifest? {
        guard FileManager.default.fileExists(atPath: manifestURL.path) else { return nil }
        do {
            let envelope = try JSONDecoder().decode(Envelope.self, from: Data(contentsOf: manifestURL))
            guard digest(envelope.payload) == envelope.digest else { throw EmbeddingStoreError.corrupt }
            let manifest = try JSONDecoder().decode(Manifest.self, from: envelope.payload)
            guard manifest.version == 1, manifest.metadata == metadata, manifest.compatibilityIdentity == spec.compatibilityIdentity else { throw EmbeddingStoreError.incompatible }
            if let selectedGeneration, manifest.generation != selectedGeneration { throw EmbeddingStoreError.staleGeneration }
            return manifest
        } catch let error as EmbeddingStoreError { throw error }
        catch { throw EmbeddingStoreError.corrupt }
    }

    func load() throws -> EmbeddingStoreSnapshot? {
        Self.transactionLock.lock(); defer { Self.transactionLock.unlock() }
        guard let manifest = try readManifest() else {
            guard selectedGeneration == nil else { return nil }
            let main = baseDir.appendingPathComponent(mainFileName)
            let sidecarsExist = [journalFileName, tombstoneFileName].contains {
                FileManager.default.fileExists(atPath: baseDir.appendingPathComponent($0).path)
            }
            guard FileManager.default.fileExists(atPath: main.path) || sidecarsExist else { return nil }
            guard FileManager.default.fileExists(atPath: main.path) else { throw EmbeddingStoreError.corrupt }
            let handle = try FileHandle(forReadingFrom: main); defer { try? handle.close() }
            _ = try readHeader(from: handle)
            guard let embeddings = loadFromBinaryFormat() else { throw EmbeddingStoreError.corrupt }
            // Stable adoption identity across repeated recovery before the first transaction.
            let hash = SHA256.hash(data: Data((spec.modelID + checkpointHash).utf8))
            let bytes = Array(hash.prefix(16))
            let generation = UUID(uuid: (bytes[0],bytes[1],bytes[2],bytes[3],bytes[4],bytes[5],bytes[6],bytes[7],bytes[8],bytes[9],bytes[10],bytes[11],bytes[12],bytes[13],bytes[14],bytes[15]))
            return EmbeddingStoreSnapshot(embeddings: embeddings, generation: generation, revision: 0, checkpoint: nil)
        }
        var embeddings: [String: MLMultiArray] = [:]
        for segment in manifest.segments {
            guard segment.name == URL(fileURLWithPath: segment.name).lastPathComponent else { throw EmbeddingStoreError.corrupt }
            let url = transactionDirectory.appendingPathComponent(segment.name)
            guard let bytes = try? Data(contentsOf: url, options: .mappedIfSafe), digest(bytes) == segment.digest else { throw EmbeddingStoreError.corrupt }
            let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
            let header = try readHeader(from: handle)
            guard header.recordCount == segment.count else { throw EmbeddingStoreError.corrupt }
            var remaining = segment.count
            for id in segment.deleting { embeddings.removeValue(forKey: id) }
            guard readRecordsFromBinary(handle, startingAt: header.recordsOffset, remainingRecords: &remaining, into: &embeddings), remaining == 0 else { throw EmbeddingStoreError.corrupt }
        }
        return EmbeddingStoreSnapshot(embeddings: embeddings, generation: manifest.generation,
                                      revision: manifest.revision, checkpoint: manifest.checkpoint)
    }

    /// Atomically commit vector mutations and their durable indexing checkpoint.
    @discardableResult
    func commit(upserts: [String: MLMultiArray] = [:], deleting: Set<String> = [],
                checkpoint: Data? = nil, generation: UUID, replacing: Bool = false) throws -> UInt64 {
        Self.transactionLock.lock(); defer { Self.transactionLock.unlock() }
        guard isValid(upserts), deleting.allSatisfy({ !$0.isEmpty && !$0.contains(where: \.isNewline) && $0.utf8.count <= Int(UInt16.max) }) else { throw EmbeddingStoreError.invalidRecords }
        if let selectedGeneration, selectedGeneration != generation { throw EmbeddingStoreError.staleGeneration }
        let previous = try load()
        if let previous, previous.generation != generation { throw EmbeddingStoreError.staleGeneration }
        let oldManifest = try readManifest()
        var records = upserts
        var segments = replacing ? [] : (oldManifest?.segments ?? [])
        if !replacing, oldManifest == nil, let previous {
            records = previous.embeddings
            for id in deleting { records.removeValue(forKey: id) }
            records.merge(upserts) { _, new in new }
        }
        guard FileManager.default.fileExists(atPath: baseDir.path) else { throw EmbeddingStoreError.writeFailed }
        try FileManager.default.createDirectory(at: transactionDirectory, withIntermediateDirectories: true)
        for directory in [baseDir, baseDir.appendingPathComponent("qemb-transactions"), transactionDirectory.deletingLastPathComponent()] {
            try syncDirectory(directory)
        }
        let name = UUID().uuidString + ".qemb"
        let url = transactionDirectory.appendingPathComponent(name)
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.write(contentsOf: makeHeader(recordCount: UInt64(records.count)))
        for id in records.keys.sorted() { try handle.write(contentsOf: recordData(id: id, mlArray: records[id]!)) }
        try faultInjector?(.segmentWritten)
        try handle.synchronize(); try faultInjector?(.segmentSynced)
        try syncDirectory(); try faultInjector?(.segmentDirectorySynced)
        let checksum = digest(try Data(contentsOf: url, options: .mappedIfSafe))
        segments.append(Segment(name: name, digest: checksum, count: UInt64(records.count), deleting: deleting.sorted()))
        guard (previous?.revision ?? 0) < UInt64.max else { throw EmbeddingStoreError.writeFailed }
        let revision = (previous?.revision ?? 0) + 1
        let manifest = Manifest(version: 1, compatibilityIdentity: spec.compatibilityIdentity, metadata: metadata, generation: generation, revision: revision,
                                segments: segments, checkpoint: checkpoint)
        let payload = try JSONEncoder().encode(manifest)
        let bytes = try JSONEncoder().encode(Envelope(payload: payload, digest: digest(payload)))
        let temporary = transactionDirectory.appendingPathComponent(UUID().uuidString + ".tmp")
        FileManager.default.createFile(atPath: temporary.path, contents: nil)
        let manifestHandle = try FileHandle(forWritingTo: temporary)
        defer { try? manifestHandle.close() }
        try manifestHandle.write(contentsOf: bytes); try faultInjector?(.manifestWritten)
        try manifestHandle.synchronize(); try faultInjector?(.manifestSynced)
        guard rename(temporary.path, manifestURL.path) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        try faultInjector?(.manifestRenamed)
        try syncDirectory(); try faultInjector?(.directorySynced)
        // Readers and writers share the lock; no live reader can retain an old
        // manifest while its segments are removed. Cleanup is never a commit step.
        let referenced = Set(segments.map(\.name))
        if let files = try? FileManager.default.contentsOfDirectory(at: transactionDirectory, includingPropertiesForKeys: nil) {
            for file in files where (file.pathExtension == "qemb" || file.pathExtension == "tmp") && !referenced.contains(file.lastPathComponent) {
                try? FileManager.default.removeItem(at: file)
            }
        }
        try faultInjector?(.cleanupFinished)
        return revision
    }

    private func syncDirectory(_ directory: URL? = nil) throws {
        let fd = open((directory ?? transactionDirectory).path, O_RDONLY)
        guard fd >= 0 else { throw EmbeddingStoreError.writeFailed }
        defer { close(fd) }
        guard fsync(fd) == 0 else { throw EmbeddingStoreError.writeFailed }
    }

    private struct HeaderMetadata: Codable, Equatable {
        let modelID: String
        let checkpointHash: String
        let dimension: Int
        let scalarType: String
        let preprocessingFingerprint: String
        let normalized: Bool
    }

    private let spec: EmbeddingModelSpec
    private let checkpointHash: String
    private let headerMagic: [UInt8] = [0x51, 0x45, 0x4D, 0x42] // "QEMB"
    private let formatVersion: UInt32 = 2
    private let recordEmbeddingSize: Int

    private let mainFileName: String
    private let journalFileName: String
    private let tombstoneFileName: String
    private let baseDir: URL

    init(spec: EmbeddingModelSpec, checkpointHash: String, directory: URL? = nil, generation: UUID? = nil) {
        self.selectedGeneration = generation
        self.spec = spec
        self.checkpointHash = checkpointHash
        self.recordEmbeddingSize = spec.embeddingDimension * MemoryLayout<Float32>.size
        let baseName = "imageEmbedding.\(spec.modelID).\(checkpointHash).v2"
        self.mainFileName = "\(baseName).qemb"
        self.journalFileName = "\(baseName)_journal.qemb"
        self.tombstoneFileName = "\(baseName)_tombstones.txt"
        self.baseDir = directory ?? FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    // MARK: - Load

    /// Load embeddings only when their model and checkpoint metadata match this store.
    /// Returns nil if no data exists.
    func loadAll() -> [String: MLMultiArray]? {
        guard let snapshot = try? load() else { return nil }
        return snapshot.embeddings.isEmpty ? nil : snapshot.embeddings
    }

    /// Validate the pre-transaction v2 format for non-destructive migration.
    private func loadFromBinaryFormat() -> [String: MLMultiArray]? {
        let startTime = Date()
        var embeddings = [String: MLMultiArray]()

        let mainPath = baseDir.appendingPathComponent(mainFileName)
        guard let mainHandle = try? FileHandle(forReadingFrom: mainPath) else { return nil }
        defer { try? mainHandle.close() }
        guard let header = try? readHeader(from: mainHandle) else { return nil }
        var remainingRecords = header.recordCount
        guard readRecordsFromBinary(mainHandle, startingAt: header.recordsOffset,
                                    remainingRecords: &remainingRecords, into: &embeddings) else {
            return nil
        }

        // Load journal (incremental additions)
        let journalPath = baseDir.appendingPathComponent(journalFileName)
        if FileManager.default.fileExists(atPath: journalPath.path) {
            guard let journalHandle = try? FileHandle(forReadingFrom: journalPath) else { return nil }
            defer { try? journalHandle.close() }
            guard readRecordsFromBinary(journalHandle, startingAt: 0,
                                        remainingRecords: &remainingRecords, into: &embeddings) else { return nil }
        }

        // A missing complete record is corruption too, even when EOF is aligned.
        guard remainingRecords == 0 else { return nil }

        // Apply tombstones (deletions)
        guard let tombstones = try? loadTombstones() else { return nil }
        for id in tombstones {
            embeddings.removeValue(forKey: id)
        }

        print("[EmbeddingStore] Loaded \(embeddings.count) embeddings in \(String(format: "%.3f", Date().timeIntervalSince(startTime)))s")
        return embeddings
    }

    // MARK: - Save

    @discardableResult
    func saveAll(_ embeddings: [String: MLMultiArray]) -> Bool {
        do {
            let old = try load()
            try commit(upserts: embeddings, checkpoint: old?.checkpoint,
                       generation: old?.generation ?? selectedGeneration ?? UUID(), replacing: true)
            return true
        } catch { return false }
    }

    @discardableResult
    func appendNew(_ embeddings: [String: MLMultiArray]) -> Bool {
        do {
            let old = try load()
            try commit(upserts: embeddings, checkpoint: old?.checkpoint,
                       generation: old?.generation ?? selectedGeneration ?? UUID())
            return true
        } catch { return false }
    }

    @discardableResult
    func markDeleted(_ ids: [String]) -> Bool {
        do {
            let old = try load()
            try commit(deleting: Set(ids), checkpoint: old?.checkpoint,
                       generation: old?.generation ?? selectedGeneration ?? UUID())
            return true
        } catch { return false }
    }

    @discardableResult
    func compact(_ embeddings: [String: MLMultiArray]) -> Bool { saveAll(embeddings) }
    func needsCompaction() -> Bool { ((try? readManifest())?.segments.count ?? 0) > 32 }

    // MARK: - Binary Format Helpers

    private func makeHeader(recordCount: UInt64) throws -> Data {
        let metadata = HeaderMetadata(
            modelID: spec.modelID,
            checkpointHash: checkpointHash,
            dimension: spec.embeddingDimension,
            scalarType: spec.storageScalarType.rawValue,
            preprocessingFingerprint: spec.preprocessingFingerprint,
            normalized: spec.normalizeEmbeddings
        )
        let metadataData = try JSONEncoder().encode(metadata)
        guard metadataData.count <= 65_536 else { throw EmbeddingStoreError.metadataTooLarge(metadataData.count) }

        var data = Data(headerMagic)
        var version = formatVersion.littleEndian
        var count = recordCount.littleEndian
        var metadataLength = UInt32(metadataData.count).littleEndian
        data.append(Data(bytes: &version, count: MemoryLayout<UInt32>.size))
        data.append(Data(bytes: &count, count: MemoryLayout<UInt64>.size))
        data.append(Data(bytes: &metadataLength, count: MemoryLayout<UInt32>.size))
        data.append(metadataData)
        return data
    }

    private func readAndValidateHeader(_ data: Data) -> (recordsOffset: Int, recordCount: UInt64)? {
        guard data.count >= 20,
              Array(data[0..<4]) == headerMagic,
              readUInt32(data, at: 4) == formatVersion,
              let recordCount = readUInt64(data, at: 8),
              let metadataLength = readUInt32(data, at: 16) else { return nil }
        guard metadataLength <= 65_536 else { return nil }
        let recordsOffset = 20 + Int(metadataLength)
        guard recordsOffset <= data.count,
              let metadata = try? JSONDecoder().decode(
                HeaderMetadata.self,
                from: data[20..<recordsOffset]
              ),
              metadata == HeaderMetadata(
                modelID: spec.modelID,
                checkpointHash: checkpointHash,
                dimension: spec.embeddingDimension,
                scalarType: spec.storageScalarType.rawValue,
                preprocessingFingerprint: spec.preprocessingFingerprint,
                normalized: spec.normalizeEmbeddings
              ) else { return nil }
        return (recordsOffset, recordCount)
    }

    private func isValid(_ embeddings: [String: MLMultiArray]) -> Bool {
        embeddings.allSatisfy {
            !$0.key.isEmpty &&
            !$0.key.contains(where: { $0.isNewline }) &&
            $0.key.utf8.count <= Int(UInt16.max) &&
            $0.value.dataType == .float32 &&
            $0.value.count == spec.embeddingDimension &&
            MLShapedArray<Float32>(converting: $0.value).scalars.allSatisfy(\.isFinite)
        }
    }

    private func recordData(id: String, mlArray: MLMultiArray) -> Data {
        var data = Data()
        let idBytes = Array(id.utf8)
        var idLen = UInt16(idBytes.count).littleEndian
        data.append(Data(bytes: &idLen, count: 2))
        data.append(contentsOf: idBytes)

        // Write embedding as raw Float32 bytes
        let shaped = MLShapedArray<Float32>(converting: mlArray)
        let scalars = shaped.scalars
        let norm = sqrt(scalars.reduce(0.0) { $0 + Double($1) * Double($1) })
        let values = spec.normalizeEmbeddings && norm > 1e-8
            ? scalars.map { Float32(Double($0) / norm) }
            : scalars
        let bits = values.map { $0.bitPattern.littleEndian }
        bits.withUnsafeBytes { data.append(contentsOf: $0) }
        return data
    }

    private func readRecordsFromBinary(
        _ handle: FileHandle,
        startingAt startOffset: Int,
        remainingRecords: inout UInt64,
        into embeddings: inout [String: MLMultiArray]
    ) -> Bool {
        do {
            try handle.seek(toOffset: UInt64(startOffset))
            while true {
                let idLengthData = try readExactly(2, from: handle)
                if idLengthData.isEmpty { break }
                guard remainingRecords > 0, idLengthData.count == 2,
                      let idLen = readUInt16(idLengthData, at: 0), idLen > 0 else { return false }
                let idData = try readExactly(Int(idLen), from: handle)
                guard idData.count == Int(idLen),
                      let id = String(data: idData, encoding: .utf8),
                      !id.contains(where: { $0.isNewline }) else { return false }
                let embeddingData = try readExactly(recordEmbeddingSize, from: handle)
                guard embeddingData.count == recordEmbeddingSize else { return false }
                var floats = [Float32](repeating: 0, count: spec.embeddingDimension)
                embeddingData.withUnsafeBytes { bytes in
                    for index in floats.indices {
                        let bits = bytes.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self)
                        floats[index] = Float32(bitPattern: UInt32(littleEndian: bits))
                    }
                }
                guard floats.allSatisfy(\.isFinite) else { return false }
                remainingRecords -= 1

                let shaped = MLShapedArray<Float32>(scalars: floats, shape: [1, spec.embeddingDimension])
                embeddings[id] = MLMultiArray(shaped)
            }
            return true
        } catch {
            return false
        }
    }

    private func readHeader(from handle: FileHandle) throws -> (recordsOffset: Int, recordCount: UInt64) {
        try handle.seek(toOffset: 0)
        let fixedHeader = try readExactly(20, from: handle)
        guard fixedHeader.count == 20,
              let metadataLength = readUInt32(fixedHeader, at: 16),
              metadataLength <= 65_536 else {
            throw EmbeddingStoreError.corrupt
        }
        let metadata = try readExactly(Int(metadataLength), from: handle)
        guard metadata.count == Int(metadataLength) else {
            throw EmbeddingStoreError.corrupt
        }

        var headerData = fixedHeader
        headerData.append(metadata)
        guard let decoded = try? JSONDecoder().decode(HeaderMetadata.self, from: metadata) else { throw EmbeddingStoreError.corrupt }
        guard decoded == self.metadata else { throw EmbeddingStoreError.incompatible }
        guard let header = readAndValidateHeader(headerData) else { throw EmbeddingStoreError.corrupt }
        return header
    }

    private func readExactly(_ count: Int, from handle: FileHandle) throws -> Data {
        var result = Data()
        while result.count < count {
            guard let chunk = try handle.read(upToCount: count - result.count), !chunk.isEmpty else {
                break
            }
            result.append(chunk)
        }
        return result
    }

    private func readUInt16(_ data: Data, at offset: Int) -> UInt16? {
        guard offset + MemoryLayout<UInt16>.size <= data.count else { return nil }
        return data.withUnsafeBytes { UInt16(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt16.self)) }
    }

    private func readUInt32(_ data: Data, at offset: Int) -> UInt32? {
        guard offset + MemoryLayout<UInt32>.size <= data.count else { return nil }
        return data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)) }
    }

    private func readUInt64(_ data: Data, at offset: Int) -> UInt64? {
        guard offset + MemoryLayout<UInt64>.size <= data.count else { return nil }
        return data.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt64.self)) }
    }

    private func loadTombstones() throws -> Set<String> {
        let path = baseDir.appendingPathComponent(tombstoneFileName)
        guard FileManager.default.fileExists(atPath: path.path) else { return [] }
        let content = try String(contentsOf: path, encoding: .utf8)
        guard content.isEmpty || content.hasSuffix("\n") else { throw EmbeddingStoreError.corrupt }
        let ids = content.split(separator: "\n", omittingEmptySubsequences: true).map(String.init)
        guard ids.allSatisfy({ !$0.contains(where: \.isNewline) && $0.utf8.count <= Int(UInt16.max) }) else { throw EmbeddingStoreError.corrupt }
        return Set(ids)
    }
}
