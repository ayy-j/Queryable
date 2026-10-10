import XCTest
import CoreML
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

    private func segments() throws -> [URL] {
        let enumerator = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil)!
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
}
