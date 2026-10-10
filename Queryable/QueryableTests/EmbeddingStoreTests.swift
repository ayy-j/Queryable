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

    func testRestartReplacementDeletionAndCompactionCountPhysicalRecords() throws {
        XCTAssertTrue(store().saveAll(["one": vector(), "two": vector()]))
        XCTAssertTrue(store().appendNew(["one": vector(-1), "three": vector()]))
        XCTAssertTrue(store().markDeleted(["two"]))
        let loaded = try XCTUnwrap(store().loadAll())
        XCTAssertEqual(Set(loaded.keys), ["one", "three"])
        XCTAssertEqual(try XCTUnwrap(loaded["one"])[0].floatValue, -1)
        // Four physical records, although only two live identifiers remain.
        XCTAssertEqual(Array(try Data(contentsOf: file())[8..<16]), [4, 0, 0, 0, 0, 0, 0, 0])
        XCTAssertTrue(store().compact(loaded))
        XCTAssertEqual(Set(try XCTUnwrap(store().loadAll()).keys), ["one", "three"])
        XCTAssertEqual(Array(try Data(contentsOf: file())[8..<16]), [2, 0, 0, 0, 0, 0, 0, 0])
    }

    func testLittleEndianHeaderIdentifierLengthAndFloatBytes() throws {
        XCTAssertTrue(store().saveAll(["id": vector()]))
        let bytes = try Data(contentsOf: file())
        XCTAssertEqual(Array(bytes[0..<16]), [81, 69, 77, 66, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        let offset = 20 + Int(bytes[16]) + (Int(bytes[17]) << 8)
        XCTAssertEqual(Array(bytes[offset..<(offset + 8)]), [2, 0, 105, 100, 0, 0, 128, 63])
    }

    func testIncorrectCountsRejectEntireIndex() throws {
        for count: UInt8 in [0, 1, 3, 255] {
            XCTAssertTrue(store().saveAll(["one": vector(), "two": vector()]))
            try mutate(file()) { $0[8] = count }
            XCTAssertNil(store().loadAll(), "count \(count)")
        }
    }

    func testMissingWholeJournalRecordAndUnexpectedExtraRecordAreRejected() throws {
        XCTAssertTrue(store().saveAll(["main": vector()]))
        XCTAssertTrue(store().appendNew(["journal": vector()]))
        let journal = try Data(contentsOf: file("_journal.qemb"))
        try Data().write(to: file("_journal.qemb"))
        XCTAssertNil(store().loadAll())
        try (journal + journal).write(to: file("_journal.qemb"))
        XCTAssertNil(store().loadAll())
    }

    func testEveryTruncatedRecordBoundaryRejectsWithoutReturningValidPrefix() throws {
        XCTAssertTrue(store().saveAll(["valid-main": vector()]))
        XCTAssertTrue(store().appendNew(["é": vector()]))
        let journal = try Data(contentsOf: file("_journal.qemb"))
        for length in [0, 1, 2, 3, 4, 5, journal.count - 1] {
            try journal.prefix(length).write(to: file("_journal.qemb"))
            XCTAssertNil(store().loadAll(), "length \(length)")
        }
    }

    func testMalformedHeadersRejectEntireIndex() throws {
        XCTAssertTrue(store().saveAll(["valid": vector()]))
        let original = try Data(contentsOf: file())
        for (offset, value): (Int, UInt8) in [(0, 0), (4, 3), (16, 255), (18, 1)] {
            var damaged = original
            damaged[offset] = value
            try damaged.write(to: file())
            XCTAssertNil(store().loadAll(), "offset \(offset)")
        }
        for length in [0, 4, 19, 20] {
            try original.prefix(length).write(to: file())
            XCTAssertNil(store().loadAll())
        }
    }

    func testInvalidUTF8EmptyIDAndNonFiniteJournalValuesRejectEntireIndex() throws {
        XCTAssertTrue(store().saveAll(["valid": vector()]))
        XCTAssertTrue(store().appendNew(["x": vector()]))
        let journal = try Data(contentsOf: file("_journal.qemb"))
        var invalidUTF8 = journal
        invalidUTF8[2] = 255
        var emptyID = journal
        emptyID[0] = 0
        var newlineID = journal
        newlineID[2] = 10
        var nan = journal
        nan.replaceSubrange(3..<7, with: [0, 0, 192, 127])
        var infinity = journal
        infinity.replaceSubrange(3..<7, with: [0, 0, 128, 127])
        for damaged in [invalidUTF8, emptyID, newlineID, nan, infinity] {
            try damaged.write(to: file("_journal.qemb"))
            XCTAssertNil(store().loadAll())
        }
    }

    func testInvalidWritesLeavePreviouslySavedFilesUntouched() throws {
        XCTAssertTrue(store().saveAll(["valid": vector()]))
        let original = try Data(contentsOf: file())
        for bad in [["": vector()], ["line\nbreak": vector()], ["nan": vector(.nan)],
                    ["infinity": vector(.infinity)], [String(repeating: "x", count: 65_536): vector()]] {
            XCTAssertFalse(store().saveAll(bad))
            XCTAssertFalse(store().appendNew(bad))
            XCTAssertEqual(try Data(contentsOf: file()), original)
            XCTAssertFalse(FileManager.default.fileExists(atPath: file("_journal.qemb").path))
        }
    }

    func testLargeFiniteValuesNormalizeWithoutOverflowAndBlankLegacyValuesRemainRepairable() throws {
        XCTAssertTrue(store().saveAll(["large": vector(.greatestFiniteMagnitude), "blank": vector(0)]))
        let loaded = try XCTUnwrap(store().loadAll())
        XCTAssertEqual(try XCTUnwrap(loaded["large"])[0].floatValue, 1)
        XCTAssertEqual(try XCTUnwrap(loaded["blank"])[0].floatValue, 0)
    }
}
