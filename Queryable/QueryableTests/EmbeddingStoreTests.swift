import XCTest
import CoreML
import CryptoKit
@testable import Queryable

final class EmbeddingStoreTests: XCTestCase {
    private var directory: URL!
    private let checkpoint = "storage-tests"
    private let spec = EmbeddingModelSpec.mobileCLIP2S4

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: directory)
    }

    private func store() -> EmbeddingStore {
        EmbeddingStore(spec: spec, checkpointHash: checkpoint, directory: directory)
    }

    private func file(_ suffix: String = ".qemb") -> URL {
        directory.appendingPathComponent("imageEmbedding.\(spec.modelID).\(checkpoint).v2\(suffix)")
    }

    private func vector(_ first: Float32 = 1) -> MLMultiArray {
        var values = [Float32](repeating: 0, count: spec.embeddingDimension)
        values[0] = first
        return MLMultiArray(MLShapedArray(scalars: values, shape: [1, spec.embeddingDimension]))
    }

    private func mutate(_ url: URL, _ change: (inout Data) throws -> Void) throws {
        var data = try Data(contentsOf: url)
        try change(&data)
        try data.write(to: url)
    }

    private func segments(in root: URL? = nil) throws -> [URL] {
        let enumerator = FileManager.default.enumerator(at: root ?? directory!, includingPropertiesForKeys: nil)!
        return enumerator.compactMap { $0 as? URL }.filter { $0.pathExtension == "qemb" }
    }

    func testRestartReplacementDeletionAndCompaction() throws {
        XCTAssertTrue(store().saveAll(["one": vector(), "two": vector()]))
        XCTAssertTrue(store().appendNew(["one": vector(-1), "three": vector()]))
        XCTAssertTrue(store().markDeleted(["two"]))
        let loaded = try XCTUnwrap(store().load())
        XCTAssertEqual(Set(loaded.embeddings.keys), ["one", "three"])
        XCTAssertEqual(loaded.embeddings["one"]?[0].floatValue, -1)
        XCTAssertTrue(store().compact(loaded.embeddings))
        XCTAssertEqual(Set(try XCTUnwrap(store().load()).embeddings.keys), ["one", "three"])
    }

    func testLittleEndianHeaderIdentifierLengthAndFloatBytes() throws {
        XCTAssertTrue(store().saveAll(["id": vector()]))
        let bytes = try Data(contentsOf: XCTUnwrap(segments().first))
        XCTAssertEqual(Array(bytes[0..<16]), [81, 69, 77, 66, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        let offset = 20 + Int(bytes[16]) + (Int(bytes[17]) << 8)
        XCTAssertEqual(Array(bytes[offset..<(offset + 8)]), [2, 0, 105, 100, 0, 0, 128, 63])
    }

    func testMissingEmptyIncompatibleAndCorruptAreDistinct() throws {
        XCTAssertNil(try store().load())
        XCTAssertTrue(store().saveAll([:]))
        XCTAssertEqual(try store().load()?.embeddings.count, 0)
        let segment = try XCTUnwrap(segments().first)
        try Data().write(to: segment)
        XCTAssertThrowsError(try store().load()) {
            guard case EmbeddingStoreError.corrupt = $0 else { return XCTFail("\($0)") }
        }
    }

    func testEveryPersistenceBoundaryRecoversWholeTransactionRepeatedly() throws {
        for boundary in EmbeddingStore.Boundary.allCases {
            let generation = UUID()
            let writer = EmbeddingStore(spec: spec, checkpointHash: checkpoint, directory: directory, generation: generation)
            try writer.commit(upserts: ["retained": vector(), "deleted": vector()], checkpoint: Data("before".utf8), generation: generation)
            writer.faultInjector = { reached in
                if reached == boundary { throw EmbeddingStoreError.writeFailed }
            }
            XCTAssertThrowsError(try writer.commit(upserts: ["new": vector()], deleting: ["deleted"], checkpoint: Data("after".utf8), generation: generation))
            for _ in 0..<3 {
                let reopened = EmbeddingStore(spec: spec, checkpointHash: checkpoint, directory: directory, generation: generation)
                let snapshot = try XCTUnwrap(reopened.load())
                let committed = boundary == .manifestRenamed || boundary == .directorySynced || boundary == .cleanupFinished
                XCTAssertEqual(Set(snapshot.embeddings.keys), committed ? ["retained", "new"] : ["retained", "deleted"])
                XCTAssertEqual(snapshot.checkpoint, Data((committed ? "after" : "before").utf8))
                XCTAssertEqual(snapshot.revision, committed ? 2 : 1)
            }
        }
    }

    func testCompactionAndTombstoneReplacementAtEveryBoundary() throws {
        for replacing in [false, true] {
            for boundary in EmbeddingStore.Boundary.allCases {
                let generation = UUID()
                let writer = EmbeddingStore(spec: spec, checkpointHash: checkpoint, directory: directory, generation: generation)
                try writer.commit(upserts: ["deleted": vector(), "keep": vector()], checkpoint: nil, generation: generation)
                try writer.commit(deleting: ["deleted"], checkpoint: nil, generation: generation)
                writer.faultInjector = { if $0 == boundary { throw EmbeddingStoreError.writeFailed } }
                XCTAssertThrowsError(try writer.commit(upserts: ["deleted": vector(-1), "keep": vector()], checkpoint: nil, generation: generation, replacing: replacing))
                let recovered = try XCTUnwrap(writer.load())
                let committed = boundary == .manifestRenamed || boundary == .directorySynced || boundary == .cleanupFinished
                XCTAssertEqual(recovered.embeddings["deleted"]?[0].floatValue, committed ? -1 : nil)
                XCTAssertNotNil(recovered.embeddings["keep"])
            }
        }
    }

    func testGenerationIsolationAndStaleWriterRejection() throws {
        let generation = UUID()
        let writer = EmbeddingStore(spec: spec, checkpointHash: checkpoint, directory: directory, generation: generation)
        try writer.commit(upserts: ["working": vector()], checkpoint: nil, generation: generation)
        XCTAssertThrowsError(try writer.commit(upserts: ["stale": vector()], checkpoint: nil, generation: UUID()))
        let replacement = UUID()
        let rebuilding = EmbeddingStore(spec: spec, checkpointHash: checkpoint, directory: directory, generation: replacement)
        try rebuilding.commit(upserts: ["replacement": vector()], checkpoint: nil, generation: replacement)
        XCTAssertEqual(Set(try XCTUnwrap(writer.load()).embeddings.keys), ["working"])
    }

    func testCorruptAndTruncatedSegmentsRejectWithoutValidPrefix() throws {
        XCTAssertTrue(store().saveAll(["valid": vector()]))
        let segment = try XCTUnwrap(segments().first)
        let original = try Data(contentsOf: segment)
        for length in [0, 4, 19, 20, original.count - 1] {
            try original.prefix(length).write(to: segment)
            XCTAssertThrowsError(try store().load())
        }
        for offset in [0, 4, 8, 16, original.count - 1] {
            var damaged = original; damaged[offset] ^= 255
            try damaged.write(to: segment)
            XCTAssertThrowsError(try store().load())
        }
    }

    func testLegacyMigrationPreservesSourcesAndRejectsCountMismatch() throws {
        XCTAssertTrue(store().saveAll(["legacy": vector()]))
        let source = try XCTUnwrap(segments().first)
        let legacyBytes = try Data(contentsOf: source)
        try FileManager.default.removeItem(at: directory.appendingPathComponent("qemb-transactions"))
        try legacyBytes.write(to: file())
        XCTAssertEqual(Set(try XCTUnwrap(store().load()).embeddings.keys), ["legacy"])
        XCTAssertTrue(store().appendNew(["new": vector()]))
        XCTAssertEqual(try Data(contentsOf: file()), legacyBytes)
        XCTAssertEqual(Set(try XCTUnwrap(store().load()).embeddings.keys), ["legacy", "new"])
        try FileManager.default.removeItem(at: directory.appendingPathComponent("qemb-transactions"))
        try mutate(file()) { $0[8] = 2 }
        XCTAssertThrowsError(try store().load())
        XCTAssertFalse(store().appendNew(["new": vector()]))
    }

    func testTornUncommittedFilesAreIgnoredAndManifestDamageIsExplicit() throws {
        XCTAssertTrue(store().saveAll(["committed": vector()]))
        let segment = try XCTUnwrap(segments().first)
        let folder = segment.deletingLastPathComponent()
        try Data([81, 69]).write(to: folder.appendingPathComponent("torn.qemb"))
        try Data("{partial".utf8).write(to: folder.appendingPathComponent("torn.tmp"))
        for _ in 0..<3 {
            XCTAssertEqual(Set(try XCTUnwrap(store().load()).embeddings.keys), ["committed"])
        }
        try mutate(folder.appendingPathComponent("manifest.json")) { $0[$0.count / 2] ^= 1 }
        XCTAssertThrowsError(try store().load()) {
            guard case EmbeddingStoreError.corrupt = $0 else { return XCTFail("\($0)") }
        }
        XCTAssertFalse(store().appendNew(["replacement": vector()]))
    }

    func testLegacyJournalExactCountAndEveryTruncationRejects() throws {
        XCTAssertTrue(store().saveAll(["é": vector()]))
        let bytes = try Data(contentsOf: XCTUnwrap(segments().first))
        try FileManager.default.removeItem(at: directory.appendingPathComponent("qemb-transactions"))
        let offset = 20 + Int(bytes[16]) + (Int(bytes[17]) << 8)
        var header = Data(bytes.prefix(offset))
        header[8] = 1
        try header.write(to: file())
        let record = Data(bytes.dropFirst(offset))
        for length in [0, 1, 2, 3, 4, 5, record.count - 1] {
            try record.prefix(length).write(to: file("_journal.qemb"))
            XCTAssertThrowsError(try store().load())
        }
        try (record + record).write(to: file("_journal.qemb"))
        XCTAssertThrowsError(try store().load())
        try record.write(to: file("_journal.qemb"))
        XCTAssertEqual(Set(try XCTUnwrap(store().load()).embeddings.keys), ["é"])
    }

    func testLegacyMalformedHeadersRecordsAndTombstonesFailClosed() throws {
        XCTAssertTrue(store().saveAll(["x": vector()]))
        let original = try Data(contentsOf: XCTUnwrap(segments().first))
        try FileManager.default.removeItem(at: directory.appendingPathComponent("qemb-transactions"))
        let offset = 20 + Int(original[16]) + (Int(original[17]) << 8)
        for (index, value): (Int, UInt8) in [(0, 0), (4, 3), (8, 0), (8, 2), (16, 255),
                                          (offset, 0), (offset + 2, 255), (offset + 2, 10)] {
            var damaged = original; damaged[index] = value
            try damaged.write(to: file())
            XCTAssertThrowsError(try store().load())
        }
        var nan = original
        nan.replaceSubrange((offset + 3)..<(offset + 7), with: [0, 0, 192, 127])
        try nan.write(to: file())
        XCTAssertThrowsError(try store().load())
        try original.write(to: file())
        for invalid in [Data([255]), Data("x".utf8), Data("x\r\n".utf8)] {
            try invalid.write(to: file("_tombstones.txt"))
            XCTAssertThrowsError(try store().load())
        }
        try Data("x\n".utf8).write(to: file("_tombstones.txt"))
        XCTAssertEqual(try store().load()?.embeddings.count, 0)
    }

    func testLegacyIncompatibleMetadataIsExplicit() throws {
        XCTAssertTrue(store().saveAll(["x": vector()]))
        var bytes = try Data(contentsOf: XCTUnwrap(segments().first))
        try FileManager.default.removeItem(at: directory.appendingPathComponent("qemb-transactions"))
        let end = 20 + Int(bytes[16]) + (Int(bytes[17]) << 8)
        var metadata = try XCTUnwrap(JSONSerialization.jsonObject(with: bytes[20..<end]) as? [String: Any])
        metadata["checkpointHash"] = "different"
        let encoded = try JSONSerialization.data(withJSONObject: metadata)
        var length = UInt32(encoded.count).littleEndian
        bytes.replaceSubrange(20..<end, with: encoded)
        bytes.replaceSubrange(16..<20, with: withUnsafeBytes(of: &length) { Data($0) })
        try bytes.write(to: file())
        XCTAssertThrowsError(try store().load()) {
            guard case EmbeddingStoreError.incompatible = $0 else { return XCTFail("\($0)") }
        }
    }

    func testCompactionReclaimsOnlyUnreferencedSegments() throws {
        XCTAssertTrue(store().saveAll(["one": vector()]))
        XCTAssertTrue(store().appendNew(["two": vector()]))
        XCTAssertEqual(try segments().count, 2)
        XCTAssertTrue(store().compact(try XCTUnwrap(store().load()).embeddings))
        XCTAssertEqual(try segments().count, 1)
        XCTAssertEqual(Set(try XCTUnwrap(store().load()).embeddings.keys), ["one", "two"])
    }

    func testInvalidWritesLeaveCommittedStateUntouched() throws {
        XCTAssertTrue(store().saveAll(["valid": vector()]))
        for bad in [["": vector()], ["line\nbreak": vector()], ["nan": vector(.nan)],
                    ["infinity": vector(.infinity)], [String(repeating: "x", count: 65_536): vector()]] {
            XCTAssertFalse(store().saveAll(bad))
            XCTAssertFalse(store().appendNew(bad))
            XCTAssertEqual(Set(try XCTUnwrap(store().load()).embeddings.keys), ["valid"])
            XCTAssertEqual(try store().load()?.revision, 1)
        }
    }

    func testLargeFiniteValuesNormalizeWithoutOverflowAndBlankValuesRemainRepairable() throws {
        XCTAssertTrue(store().saveAll(["large": vector(.greatestFiniteMagnitude), "blank": vector(0)]))
        let loaded = try XCTUnwrap(store().loadAll())
        XCTAssertEqual(try XCTUnwrap(loaded["large"])[0].floatValue, 1)
        XCTAssertEqual(try XCTUnwrap(loaded["blank"])[0].floatValue, 0)
    }

    func testFloat16RoundTripsAtSupportedDimensions() throws {
        for dimension in [512, 768, 1_152, 1_536] {
            let model = try makeModel(dimension: dimension)
            let root = try makeRoot("round-trip-\(dimension)")
            let input = (0..<dimension).map { Float32(($0 * 37) % 101 - 50) }
            let store = EmbeddingStore(spec: model, checkpointHash: checkpoint, directory: root)
            XCTAssertTrue(store.saveAll(["vector": array(input)]))
            let snapshot = try XCTUnwrap(store.load())
            let persisted = values(try XCTUnwrap(snapshot.embeddings["vector"]))
            let norm = sqrt(input.reduce(0.0) { $0 + Double($1) * Double($1) })
            XCTAssertEqual(persisted.count, dimension)
            for index in input.indices {
                assertFloat16Rounding(persisted[index], of: Double(input[index]) / norm)
            }
            let bytes = try Data(contentsOf: XCTUnwrap(segments(in: root).first))
            XCTAssertEqual(bytes.count - recordsOffset(bytes), 2 + "vector".utf8.count + dimension * 2)
        }
    }

    func testFloat16WritesExplicitLittleEndianBitPatterns() throws {
        let model = try makeModel(dimension: 8, normalization: .none)
        let root = try makeRoot("little-endian-write")
        let store = EmbeddingStore(spec: model, checkpointHash: checkpoint, directory: root)
        XCTAssertTrue(store.saveAll(["id": array([1, -2, 0.5, 65_504, 0, 0, 0, 0])]))
        let bytes = try Data(contentsOf: XCTUnwrap(segments(in: root).first))
        let offset = recordsOffset(bytes)
        XCTAssertEqual(Array(bytes[offset..<(offset + 4)]), [2, 0, 105, 100])
        // 1.0 = 0x3C00, -2.0 = 0xC000, 0.5 = 0x3800, 65504 = 0x7BFF, each stored low byte first.
        XCTAssertEqual(Array(bytes[(offset + 4)..<(offset + 20)]),
                       [0x00, 0x3C, 0x00, 0xC0, 0x00, 0x38, 0xFF, 0x7B, 0, 0, 0, 0, 0, 0, 0, 0])
    }

    func testFloat16ReadsLittleEndianBitPatternsAndRejectsNonFiniteOnes() throws {
        let model = try makeModel(dimension: 8, normalization: .none)
        let root = try makeRoot("little-endian-read")
        let store = EmbeddingStore(spec: model, checkpointHash: checkpoint, directory: root)
        XCTAssertTrue(store.saveAll(["id": array(filled(8, 0))]))
        let saved = try Data(contentsOf: XCTUnwrap(segments(in: root).first))
        let header = saved.prefix(recordsOffset(saved))
        let finite: [UInt8] = [0x00, 0x3C, 0x00, 0xC0, 0x00, 0x38, 0xFF, 0x7B,
                               0x00, 0x80, 0x01, 0x00, 0x00, 0x04, 0x00, 0x00]
        let snapshot = try XCTUnwrap(loadLegacy(header: header, payload: finite, model: model, root: root))
        XCTAssertEqual(values(try XCTUnwrap(snapshot.embeddings["id"])),
                       [1, -2, 0.5, 65_504, -0.0, 0x1p-24, 0x1p-14, 0])

        // Big-endian bytes for 1.0 (3C 00) must decode as the little-endian value 0x003C, i.e. 60 * 2^-24.
        let bigEndian: [UInt8] = [0x3C, 0x00] + [UInt8](repeating: 0, count: 14)
        let swapped = try XCTUnwrap(loadLegacy(header: header, payload: bigEndian, model: model, root: root))
        XCTAssertEqual(values(try XCTUnwrap(swapped.embeddings["id"]))[0], 60 * 0x1p-24)

        let nonFinite: [[UInt8]] = [[0x00, 0x7E], [0x00, 0x7C], [0x00, 0xFC]]
        for bits in nonFinite {
            var payload = finite
            payload.replaceSubrange(0..<2, with: bits)
            XCTAssertThrowsError(try loadLegacy(header: header, payload: payload, model: model, root: root)) {
                guard case EmbeddingStoreError.corrupt = $0 else { return XCTFail("\($0)") }
            }
        }

        let transactional = try makeRoot("little-endian-transaction")
        XCTAssertTrue(EmbeddingStore(spec: model, checkpointHash: checkpoint, directory: transactional)
            .saveAll(["id": array(filled(8, 0))]))
        try rewriteSegment(in: transactional) { bytes in
            var damaged = bytes
            damaged.replaceSubrange((self.recordsOffset(bytes) + 4)..<(self.recordsOffset(bytes) + 6), with: [0x00, 0x7E])
            return damaged
        }
        XCTAssertThrowsError(try EmbeddingStore(spec: model, checkpointHash: checkpoint, directory: transactional).load()) {
            guard case EmbeddingStoreError.corrupt = $0 else { return XCTFail("\($0)") }
        }
    }

    func testFloat16RejectsValuesThatCannotBePersisted() throws {
        let model = try makeModel(dimension: 8, normalization: .none)
        let root = try makeRoot("invalid-float16")
        let store = EmbeddingStore(spec: model, checkpointHash: checkpoint, directory: root)
        XCTAssertTrue(store.saveAll(["valid": array(filled(8, 1))]))
        // 65520 is the first magnitude that rounds to infinity in Float16 (round-to-nearest-even).
        let candidates: [Float32] = [65_520, 70_000, .greatestFiniteMagnitude, .nan, .infinity]
        for candidate in candidates {
            XCTAssertFalse(store.saveAll(["bad": array(filled(8, candidate))]))
            XCTAssertFalse(store.appendNew(["bad": array(filled(8, candidate))]))
            let snapshot = try XCTUnwrap(store.load())
            XCTAssertEqual(Set(snapshot.embeddings.keys), ["valid"])
            XCTAssertEqual(snapshot.revision, 1)
        }
    }

    func testFloat16NormalizesBeforeRoundingSoLargeRawMagnitudesPersist() throws {
        let model = try makeModel(dimension: 8)
        let root = try makeRoot("large-normalized")
        let store = EmbeddingStore(spec: model, checkpointHash: checkpoint, directory: root)
        XCTAssertTrue(store.saveAll(["large": array([70_000, 70_000, 0, 0, 0, 0, 0, 0])]))
        let snapshot = try XCTUnwrap(store.load())
        let persisted = values(try XCTUnwrap(snapshot.embeddings["large"]))
        let exact = 1 / Double(2).squareRoot()
        assertFloat16Rounding(persisted[0], of: exact)
        assertFloat16Rounding(persisted[1], of: exact)
        XCTAssertEqual(Array(persisted[2...]), [0, 0, 0, 0, 0, 0])
    }

    func testFloat16FilesRejectMetadataThatDoesNotMatchTheActiveModel() throws {
        let model = try makeModel(dimension: 512)
        let legacyRoot = try makeRoot("metadata-legacy")
        XCTAssertTrue(EmbeddingStore(spec: model, checkpointHash: checkpoint, directory: legacyRoot)
            .saveAll(["id": array(filled(512, 0.25))]))
        try writeLegacyMain(try Data(contentsOf: XCTUnwrap(segments(in: legacyRoot).first)), model: model, root: legacyRoot)
        XCTAssertNotNil(try EmbeddingStore(spec: model, checkpointHash: checkpoint, directory: legacyRoot).load())
        let mismatched = [
            try makeModel(dimension: 512, scalarType: "float32"),
            try makeModel(dimension: 1_152),
            try makeModel(dimension: 512, normalization: .none)
        ]
        for other in mismatched {
            XCTAssertThrowsError(try EmbeddingStore(spec: other, checkpointHash: checkpoint, directory: legacyRoot).load()) {
                guard case EmbeddingStoreError.incompatible = $0 else { return XCTFail("\($0)") }
            }
        }

        let manifestRoot = try makeRoot("metadata-manifest")
        XCTAssertTrue(EmbeddingStore(spec: model, checkpointHash: checkpoint, directory: manifestRoot)
            .saveAll(["id": array(filled(512, 0.25))]))
        try editManifest(in: manifestRoot) { manifest in
            var metadata = try XCTUnwrap(manifest["metadata"] as? [String: Any])
            metadata["scalarType"] = "float32"
            manifest["metadata"] = metadata
        }
        XCTAssertThrowsError(try EmbeddingStore(spec: model, checkpointHash: checkpoint, directory: manifestRoot).load()) {
            guard case EmbeddingStoreError.incompatible = $0 else { return XCTFail("\($0)") }
        }

        let segmentRoot = try makeRoot("metadata-segment")
        XCTAssertTrue(EmbeddingStore(spec: model, checkpointHash: checkpoint, directory: segmentRoot)
            .saveAll(["id": array(filled(512, 0.25))]))
        try rewriteSegment(in: segmentRoot) { bytes in
            let offset = self.recordsOffset(bytes)
            let header = String(decoding: bytes[20..<offset], as: UTF8.self)
                .replacingOccurrences(of: model.modelID, with: "mobileclip2-s5")
            let metadata = Data(header.utf8)
            var length = UInt32(metadata.count).littleEndian
            var rewritten = Data(bytes.prefix(16))
            rewritten.append(Data(bytes: &length, count: 4))
            rewritten.append(metadata)
            rewritten.append(bytes[offset...])
            return rewritten
        }
        XCTAssertThrowsError(try EmbeddingStore(spec: model, checkpointHash: checkpoint, directory: segmentRoot).load()) {
            guard case EmbeddingStoreError.incompatible = $0 else { return XCTFail("\($0)") }
        }
    }

    func testFloat16AndFloat32StoresKeepSeparateTransactionsInTheSameRoot() throws {
        let root = try makeRoot("isolation")
        let half = EmbeddingStore(spec: try makeModel(dimension: 512), checkpointHash: checkpoint, directory: root)
        let full = EmbeddingStore(spec: try makeModel(dimension: 512, scalarType: "float32"), checkpointHash: checkpoint, directory: root)
        XCTAssertTrue(half.saveAll(["half": array(filled(512, 0.25))]))
        XCTAssertNil(try full.load())
        XCTAssertTrue(full.saveAll(["full": array(filled(512, 0.5))]))
        XCTAssertEqual(Set(try XCTUnwrap(half.load()).embeddings.keys), ["half"])
        XCTAssertEqual(Set(try XCTUnwrap(full.load()).embeddings.keys), ["full"])
    }

    func testFloat16LifecycleAndCompactionStayWithinStorageTolerance() throws {
        for dimension in [512, 1_152, 1_536] {
            let model = try makeModel(dimension: dimension)
            let root = try makeRoot("lifecycle-\(dimension)")
            let store = EmbeddingStore(spec: model, checkpointHash: checkpoint, directory: root)
            let first = (0..<dimension).map { Float32(($0 % 13) - 6) + 0.5 }
            let second = (0..<dimension).map { Float32(($0 % 7) - 3) - 0.25 }
            let third = (0..<dimension).map { Float32(($0 % 5) - 2) + 0.125 }
            XCTAssertTrue(store.saveAll(["one": array(first), "two": array(second)]))
            XCTAssertTrue(store.appendNew(["three": array(third)]))
            XCTAssertTrue(store.markDeleted(["two"]))
            let loaded = try XCTUnwrap(store.load())
            XCTAssertEqual(Set(loaded.embeddings.keys), ["one", "three"])
            XCTAssertTrue(store.compact(loaded.embeddings))
            XCTAssertEqual(try segments(in: root).count, 1)
            let compacted = try XCTUnwrap(store.load())
            XCTAssertEqual(Set(compacted.embeddings.keys), ["one", "three"])
            // Compaction re-normalizes persisted Float16 values, so one compaction may move a value by up to one Float16 ulp (≤ 2^-10 relative).
            // This asserts a single compaction; repeated compactions can accumulate drift (see docs/qemb-v2-format.md).
            for id in ["one", "three"] {
                let before = values(try XCTUnwrap(loaded.embeddings[id]))
                let after = values(try XCTUnwrap(compacted.embeddings[id]))
                for index in before.indices {
                    let drift = abs(Double(after[index]) - Double(before[index]))
                    XCTAssertLessThanOrEqual(drift, max(abs(Double(before[index])) * 0x1p-9, 0x1p-24))
                }
            }
        }
    }

    // MARK: - Float16 helpers

    private func makeModel(dimension: Int, scalarType: String = "float16",
                           normalization: EmbeddingNormalization = .l2) throws -> EmbeddingModelSpec {
        let base = EmbeddingModelSpec.mobileCLIP2S4
        return try EmbeddingModelSpec(
            modelID: base.modelID,
            revision: base.revision,
            imageModelName: base.imageModelName,
            textModelName: base.textModelName,
            imageInputName: base.imageInputName,
            imageInputType: base.imageInputType,
            imageOutputName: base.imageOutputName,
            imageOutputType: base.imageOutputType,
            textInputName: base.textInputName,
            textInputType: base.textInputType,
            textOutputName: base.textOutputName,
            textOutputType: base.textOutputType,
            imageSize: base.imageSize,
            imagePreprocessing: base.imagePreprocessing,
            embeddingDimension: dimension,
            tokenizerKind: base.tokenizerKind,
            tokenizerAssets: base.tokenizerAssets,
            contextLength: base.contextLength,
            vocabularySize: base.vocabularySize,
            normalization: normalization,
            storageScalarType: scalarType
        )
    }

    private func makeRoot(_ name: String) throws -> URL {
        let root = directory.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func array(_ values: [Float32]) -> MLMultiArray {
        MLMultiArray(MLShapedArray(scalars: values, shape: [1, values.count]))
    }

    private func filled(_ count: Int, _ value: Float32) -> [Float32] {
        [Float32](repeating: value, count: count)
    }

    private func values(_ array: MLMultiArray) -> [Float32] {
        MLShapedArray<Float32>(converting: array).scalars
    }

    private func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Byte offset of the first record, read from the little-endian metadata length at bytes 16...19.
    private func recordsOffset(_ bytes: Data) -> Int {
        let metadataLength = Int(bytes[16]) | Int(bytes[17]) << 8 | Int(bytes[18]) << 16 | Int(bytes[19]) << 24
        return 20 + metadataLength
    }

    /// `actual` must be exactly representable in Float16 and within half a Float16 step of `exact`.
    /// The 2^-10 relative bound leaves margin over 2^-11 for the Double-to-Float32-to-Float16 path.
    private func assertFloat16Rounding(_ actual: Float32, of exact: Double, line: UInt = #line) {
        XCTAssertEqual(Float32(Float16(actual)), actual, line: line)
        XCTAssertLessThanOrEqual(abs(Double(actual) - exact), max(abs(exact) * 0x1p-10, 0x1p-24), line: line)
    }

    private func writeLegacyMain(_ bytes: Data, model: EmbeddingModelSpec, root: URL) throws {
        try? FileManager.default.removeItem(at: root.appendingPathComponent("qemb-transactions"))
        try bytes.write(to: root.appendingPathComponent("imageEmbedding.\(model.modelID).\(checkpoint).v2.qemb"))
    }

    /// Writes `header` and one record ("id", `payload`) as the legacy main file, then loads it.
    private func loadLegacy(header: Data, payload: [UInt8], model: EmbeddingModelSpec, root: URL) throws -> EmbeddingStoreSnapshot? {
        let record: [UInt8] = [2, 0, 105, 100] + payload
        try writeLegacyMain(header + Data(record), model: model, root: root)
        return try EmbeddingStore(spec: model, checkpointHash: checkpoint, directory: root).load()
    }

    private func manifestURL(in root: URL) throws -> URL {
        let enumerator = try XCTUnwrap(FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil))
        return try XCTUnwrap(enumerator.compactMap { $0 as? URL }.first(where: { $0.lastPathComponent == "manifest.json" }))
    }

    /// Decodes the signed manifest envelope, applies `change`, and re-signs it so only the metadata is wrong.
    private func editManifest(in root: URL, _ change: (inout [String: Any]) throws -> Void) throws {
        let url = try manifestURL(in: root)
        var envelope = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        guard let encoded = envelope["payload"] as? String, let payload = Data(base64Encoded: encoded) else {
            return XCTFail("malformed manifest envelope")
        }
        var manifest = try XCTUnwrap(JSONSerialization.jsonObject(with: payload) as? [String: Any])
        try change(&manifest)
        let edited = try JSONSerialization.data(withJSONObject: manifest)
        envelope["payload"] = edited.base64EncodedString()
        envelope["digest"] = sha256Hex(edited)
        try JSONSerialization.data(withJSONObject: envelope).write(to: url)
    }

    /// Rewrites the only segment and re-signs its digest in the manifest, so the segment is well-formed to the digest check.
    private func rewriteSegment(in root: URL, _ transform: (Data) -> Data) throws {
        let segment = try XCTUnwrap(segments(in: root).first)
        let rewritten = transform(try Data(contentsOf: segment))
        try rewritten.write(to: segment)
        try editManifest(in: root) { manifest in
            var entries = try XCTUnwrap(manifest["segments"] as? [[String: Any]])
            entries[0]["digest"] = self.sha256Hex(rewritten)
            manifest["segments"] = entries
        }
    }
}
