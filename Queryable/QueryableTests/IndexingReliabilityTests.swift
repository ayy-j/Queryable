import XCTest
import Photos
import CoreML
import UIKit
@testable import Queryable

final class IndexingImageRequestTests: XCTestCase {
    func testWaitsForFinalImageAndIgnoresPreviewAndRepeatedCallbacks() async throws {
        let started = expectation(description: "request started")
        var callback: IndexingImageRequest.Completion?
        let preview = UIImage()
        let final = UIImage()
        let task = Task {
            try await IndexingImageRequest.image { completion in
                callback = completion
                started.fulfill()
                return 42
            } cancel: { _ in XCTFail("Unexpected cancellation") }
        }
        await fulfillment(of: [started], timeout: 2)
        callback?(preview, [PHImageResultIsDegradedKey: true])
        // The final callback deliberately arrives after requestImage has returned.
        await Task.yield()
        callback?(final, [PHImageResultIsDegradedKey: false])
        callback?(nil, [PHImageErrorKey: CocoaError(.fileReadUnknown)])
        let image = try await task.value
        XCTAssertTrue(image === final)
    }

    func testMissingLocalImageErrorAndPhotosCancellationThrow() async {
        for info: [AnyHashable: Any] in [
            [:],
            [PHImageResultIsInCloudKey: true, PHImageResultIsDegradedKey: true],
            [PHImageErrorKey: CocoaError(.fileReadNoSuchFile)],
            [PHImageCancelledKey: true]
        ] {
            do {
                _ = try await IndexingImageRequest.image { completion in
                    completion(nil, info)
                    completion(UIImage(), [:])
                    return 1
                } cancel: { _ in }
                XCTFail("A failed request must not produce an image")
            } catch { }
        }
    }

    func testCancellationWhileRequestIDIsBeingAssignedCancelsPhotosRequest() async {
        var cancelledID: PHImageRequestID?
        let task = Task {
            try await IndexingImageRequest.image { _ in
                withUnsafeCurrentTask { $0?.cancel() }
                return 57
            } cancel: { cancelledID = $0 }
        }
        do { _ = try await task.value; XCTFail("Expected cancellation") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertEqual(cancelledID, 57)
    }

    func testAlreadyCancelledTaskDoesNotStartPhotosRequest() async {
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await IndexingImageRequest.image { _ in
                XCTFail("Must not start a request after cancellation")
                return 1
            } cancel: { _ in }
        }
        do { _ = try await task.value; XCTFail("Expected cancellation") }
        catch { XCTAssertTrue(error is CancellationError) }
    }

    func testCancellationAfterStartIgnoresLateCallback() async {
        let started = expectation(description: "request started")
        var callback: IndexingImageRequest.Completion?
        var cancelledID: PHImageRequestID?
        let task = Task {
            try await IndexingImageRequest.image { completion in
                callback = completion
                started.fulfill()
                return 19
            } cancel: { cancelledID = $0 }
        }
        await fulfillment(of: [started], timeout: 2)
        task.cancel()
        callback?(UIImage(), [:])
        do { _ = try await task.value; XCTFail("Expected cancellation") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertEqual(cancelledID, 19)
    }
}

@MainActor
final class IndexingReliabilityTests: XCTestCase {
    private let spec = EmbeddingModelSpec.mobileCLIP2S4
    private func assets(_ ids: [String]) -> [PhotoAsset] {
        ids.map { PhotoAsset(identifier: $0, phAsset: nil) }
    }
    private func vector(_ value: Float32 = 1) -> MLShapedArray<Float32> {
        MLShapedArray(scalars: Array(repeating: value, count: spec.embeddingDimension),
                      shape: [1, spec.embeddingDimension])
    }

    func testUnavailablePhotoStaysRetryableWhileSuccessfulPhotoIsSaved() async {
        var unavailable = true
        var saved = Set<String>()
        let searcher = PhotoSearcher(indexingOperations: PhotoIndexingOperations(
            fetchImage: { asset, _ in
                if asset.id == "cloud" && unavailable { throw CocoaError(.fileReadNoSuchFile) }
                try await Task.sleep(nanoseconds: 20_000_000)
                return UIImage()
            },
            encodeBatch: { images in images.map { _ in self.vector() } },
            encodeImage: { _ in XCTFail("Unexpected fallback"); return self.vector() },
            save: { saved.formUnion($0.keys) }
        ))
        await searcher.buildIndex(assets: assets(["local", "cloud"]))
        XCTAssertEqual(searcher.buildIndexCode, .BUILD_INCOMPLETE)
        XCTAssertEqual(searcher.curIndexingNums, 2)
        XCTAssertEqual(searcher.savedIndexingPhotosNum, 1)
        XCTAssertEqual(searcher.failedIndexingPhotosNum, 1)
        XCTAssertEqual(searcher.remainingIndexingPhotosNum, 1)
        XCTAssertEqual(saved, ["local"])
        XCTAssertNil(searcher.savedEmbedding["cloud"])
        unavailable = false
        await searcher.buildIndex(assets: assets(["cloud"]))
        XCTAssertEqual(searcher.buildIndexCode, .BUILD_FINISHED)
        XCTAssertEqual(searcher.remainingIndexingPhotosNum, 0)
        XCTAssertEqual(saved, ["local", "cloud"])
    }

    func testBatchFailureFallsBackAndIndividualFailureIsNotSaved() async {
        var singleCalls = 0
        let searcher = PhotoSearcher(indexingOperations: PhotoIndexingOperations(
            fetchImage: { _, _ in UIImage() },
            encodeBatch: { _ in throw PhotoIndexingError.invalidEmbedding },
            encodeImage: { _ in
                singleCalls += 1
                if singleCalls == 1 { throw PhotoIndexingError.invalidEmbedding }
                return self.vector()
            },
            save: { XCTAssertEqual($0.count, 1) }
        ))
        await searcher.buildIndex(assets: assets(["one", "two"]))
        XCTAssertEqual(singleCalls, 2)
        XCTAssertEqual(searcher.savedEmbedding.count, 1)
        XCTAssertEqual(searcher.failedIndexingPhotosNum, 1)
        XCTAssertEqual(searcher.buildIndexCode, .BUILD_INCOMPLETE)
    }

    func testWrongBatchCountFallsBackWithoutLosingPhotos() async {
        var singleCalls = 0
        let searcher = PhotoSearcher(indexingOperations: PhotoIndexingOperations(
            fetchImage: { _, _ in UIImage() },
            encodeBatch: { _ in [] },
            encodeImage: { _ in singleCalls += 1; return self.vector() },
            save: { XCTAssertEqual($0.count, 2) }
        ))
        await searcher.buildIndex(assets: assets(["one", "two"]))
        XCTAssertEqual(singleCalls, 2)
        XCTAssertEqual(searcher.buildIndexCode, .BUILD_FINISHED)
    }

    func testBlankNonFiniteAndWrongDimensionResultsStayRetryable() async {
        for invalid in [vector(0), vector(.nan),
                        MLShapedArray<Float32>(scalars: [1], shape: [1, 1])] {
            let searcher = PhotoSearcher(indexingOperations: PhotoIndexingOperations(
                fetchImage: { _, _ in UIImage() },
                encodeBatch: { _ in [invalid] },
                encodeImage: { _ in XCTFail("Unexpected fallback"); return self.vector() },
                save: { _ in XCTFail("Invalid results must not be saved") }
            ))
            await searcher.buildIndex(assets: assets(["bad"]))
            XCTAssertEqual(searcher.buildIndexCode, .BUILD_INCOMPLETE)
            XCTAssertEqual(searcher.failedIndexingPhotosNum, 1)
            XCTAssertTrue(searcher.savedEmbedding.isEmpty)
            XCTAssertTrue(searcher.buildingEmbedding.isEmpty)
        }
    }

    func testFailedSaveDoesNotFinishAndRetrySavesWithoutReencoding() async {
        var savesFail = true
        var batchCalls = 0
        let searcher = PhotoSearcher(indexingOperations: PhotoIndexingOperations(
            fetchImage: { _, _ in UIImage() },
            encodeBatch: { images in batchCalls += 1; return images.map { _ in self.vector() } },
            encodeImage: { _ in self.vector() },
            save: { _ in if savesFail { throw EmbeddingStoreError.writeFailed } }
        ))
        let photos = assets(["one", "two"])
        await searcher.buildIndex(assets: photos)
        XCTAssertEqual(searcher.buildIndexCode, .BUILD_ERROR)
        XCTAssertNotNil(searcher.indexingErrorMessage)
        XCTAssertEqual(searcher.savedIndexingPhotosNum, 0)
        XCTAssertEqual(searcher.remainingIndexingPhotosNum, 2)
        XCTAssertTrue(searcher.savedEmbedding.isEmpty)
        XCTAssertEqual(searcher.buildingEmbedding.count, 2)
        savesFail = false
        await searcher.buildIndex(assets: photos)
        XCTAssertEqual(searcher.buildIndexCode, .BUILD_FINISHED)
        XCTAssertNil(searcher.indexingErrorMessage)
        XCTAssertEqual(searcher.savedIndexingPhotosNum, 2)
        XCTAssertEqual(searcher.curIndexingNums, 2)
        XCTAssertEqual(searcher.savedEmbedding.count, 2)
        XCTAssertTrue(searcher.buildingEmbedding.isEmpty)
        XCTAssertEqual(batchCalls, 1)
    }

    func testInterruptedFetchDoesNotReportFinished() async {
        let searcher = PhotoSearcher(indexingOperations: PhotoIndexingOperations(
            fetchImage: { _, _ in throw CancellationError() },
            encodeBatch: { _ in XCTFail("Must not encode"); return [] },
            encodeImage: { _ in self.vector() },
            save: { _ in XCTFail("Must not save") }
        ))
        await searcher.buildIndex(assets: assets(["one"]))
        XCTAssertEqual(searcher.buildIndexCode, .BUILD_ERROR)
        XCTAssertEqual(searcher.remainingIndexingPhotosNum, 1)
        XCTAssertEqual(searcher.savedIndexingPhotosNum, 0)
    }

    func testFinalSaveFailurePreservesEarlierCheckpointAndRetriesOnlyUnsavedWork() async {
        var saveCalls = 0
        var encodedCount = 0
        var failFinalSave = true
        let searcher = PhotoSearcher(indexingOperations: PhotoIndexingOperations(
            fetchImage: { _, _ in UIImage() },
            encodeBatch: { images in
                encodedCount += images.count
                return images.map { _ in self.vector() }
            },
            encodeImage: { _ in self.vector() },
            save: { _ in
                saveCalls += 1
                if saveCalls == 2 && failFinalSave { throw EmbeddingStoreError.writeFailed }
            }
        ))
        let photos = assets((0..<5_001).map { "photo-\($0)" })
        await searcher.buildIndex(assets: photos)
        XCTAssertEqual(searcher.buildIndexCode, .BUILD_ERROR)
        XCTAssertEqual(searcher.savedIndexingPhotosNum, 5_000)
        XCTAssertEqual(searcher.savedEmbedding.count, 5_000)
        XCTAssertEqual(searcher.buildingEmbedding.count, 1)
        XCTAssertEqual(searcher.remainingIndexingPhotosNum, 1)
        failFinalSave = false
        await searcher.buildIndex(assets: photos)
        XCTAssertEqual(searcher.buildIndexCode, .BUILD_FINISHED)
        XCTAssertEqual(searcher.savedEmbedding.count, 5_001)
        XCTAssertEqual(searcher.savedIndexingPhotosNum, 1)
        XCTAssertEqual(encodedCount, 5_001)
    }

    func testJournalRoundTripAndWriteFailureUseIsolatedDirectory() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = EmbeddingStore(spec: spec, checkpointHash: "test", directory: directory)
        XCTAssertTrue(store.appendNew(["one": MLMultiArray(vector())]))
        XCTAssertTrue(store.appendNew(["two": MLMultiArray(vector(2))]))
        XCTAssertEqual(Set(try XCTUnwrap(store.loadAll()).keys), ["one", "two"])
        let missing = EmbeddingStore(spec: spec, checkpointHash: "test",
                                     directory: directory.appendingPathComponent("missing"))
        XCTAssertFalse(missing.appendNew(["three": MLMultiArray(vector())]))
        XCTAssertEqual(Set(try XCTUnwrap(store.loadAll()).keys), ["one", "two"])
    }

    func testTextComputePolicyKeepsModernDevicesAndUnknownHostsUnrestricted() {
        for id in ["iPhone12,1", "iPhone18,1", "iPad12,1", "iPad13,1", "iPad16,1", "arm64", "iPhone", "Mac17,3"] {
            XCTAssertFalse(TextEncoder.requiresCPUOnlyTextEncoding(hardwareIdentifier: id), id)
        }
        for id in ["iPhone10,6", "iPhone11,8", "iPad8,12", "iPad11,7"] {
            XCTAssertTrue(TextEncoder.requiresCPUOnlyTextEncoding(hardwareIdentifier: id), id)
        }
    }
}
