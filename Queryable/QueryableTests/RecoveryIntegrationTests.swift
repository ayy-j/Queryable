import XCTest
import CoreML
import UIKit
@testable import Queryable

@MainActor
final class RecoveryIntegrationTests: XCTestCase {
    private func vector(_ dimension: Int = 768) -> MLShapedArray<Float32> {
        MLShapedArray(scalars: [1] + Array(repeating: 0, count: dimension - 1), shape: [1, dimension])
    }
    private func assets(_ ids: [String]) -> [PhotoAsset] { ids.map { PhotoAsset(identifier: $0, phAsset: nil) } }
    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }
    private func operations(unavailable: Set<String> = []) -> PhotoIndexingOperations {
        PhotoIndexingOperations(fetchImage: { asset, _ in
            if unavailable.contains(asset.id) { throw CocoaError(.fileReadNoSuchFile) }
            return UIImage()
        }, encodeBatch: { images in images.map { _ in self.vector() } },
           encodeImage: { _ in self.vector() }, save: { _ in })
    }

    func testDurablePartialResumeKeepsOldActiveUntilAllRequiredPhotosCommit() async throws {
        let dir = try directory()
        let first = PhotoSearcher(indexingOperations: operations(), storageDirectory: dir)
        try first.beginRequest(spec: .mobileCLIP2S4, checkpointHash: "test")
        await first.buildIndex(assets: assets(["old"]))
        XCTAssertEqual(first.buildIndexCode, .BUILD_FINISHED)
        let active = try ModelIndexCoordinator(directoryURL: dir).state.active
        let rebuilding = PhotoSearcher(indexingOperations: operations(unavailable: ["cloud"]), storageDirectory: dir)
        await rebuilding.prepareModelForSearch()
        try rebuilding.beginRequest(spec: .mobileCLIP2S4, checkpointHash: "test")
        await rebuilding.buildIndex(assets: assets(["new", "cloud"]))
        XCTAssertEqual(rebuilding.buildIndexCode, .BUILD_INCOMPLETE)
        XCTAssertEqual(Set(rebuilding.savedEmbedding.keys), ["old"])
        XCTAssertEqual(try ModelIndexCoordinator(directoryURL: dir).state.active, active)
        let resumed = PhotoSearcher(indexingOperations: operations(), storageDirectory: dir)
        await resumed.prepareModelForSearch()
        XCTAssertEqual(resumed.savedIndexingPhotosNum, 1)
        XCTAssertEqual(resumed.buildIndexCode, .BUILD_PAUSED)
        await resumed.buildIndex(assets: assets(["new", "cloud"]))
        XCTAssertEqual(resumed.buildIndexCode, .BUILD_FINISHED)
        XCTAssertEqual(Set(resumed.savedEmbedding.keys), ["new", "cloud"])
        let final = try ModelIndexCoordinator(directoryURL: dir).state
        XCTAssertEqual(final.previous, active)
        XCTAssertNil(final.requested)
        XCTAssertNotEqual(final.active, active)
    }

    func testLateFetchAfterCancelCannotCommitOrActivate() async throws {
        let dir = try directory()
        let started = expectation(description: "fetch started")
        var continuation: CheckedContinuation<UIImage, Never>?
        let searcher = PhotoSearcher(indexingOperations: PhotoIndexingOperations(
            fetchImage: { _, _ in
                await withCheckedContinuation { continuation = $0; started.fulfill() }
            }, encodeBatch: { images in images.map { _ in self.vector() } },
            encodeImage: { _ in self.vector() }, save: { _ in XCTFail("Stale save") }), storageDirectory: dir)
        try searcher.beginRequest(spec: .mobileCLIP2S4, checkpointHash: "test")
        let task = Task { await searcher.buildIndex(assets: assets(["late"])) }
        await fulfillment(of: [started], timeout: 2)
        searcher.cancelIndexing()
        continuation?.resume(returning: UIImage())
        await task.value
        XCTAssertEqual(searcher.buildIndexCode, .BUILD_CANCELLED)
        XCTAssertTrue(searcher.buildingEmbedding.isEmpty)
        XCTAssertNil(try ModelIndexCoordinator(directoryURL: dir).state.active)
        XCTAssertFalse(searcher.isIndexing)
        XCTAssertFalse(UIApplication.shared.isIdleTimerDisabled)
    }

    func testLibraryReconciliationDoesNotDeleteOnLimitedAccess() async throws {
        let dir = try directory()
        let searcher = PhotoSearcher(indexingOperations: operations(), storageDirectory: dir)
        try searcher.beginRequest(spec: .mobileCLIP2S4, checkpointHash: "test")
        await searcher.buildIndex(assets: assets(["visible", "hidden"]))
        try searcher.reconcileLibrary(ids: ["visible"], fullAccess: false)
        XCTAssertEqual(Set(searcher.savedEmbedding.keys), ["visible", "hidden"])
        try searcher.reconcileLibrary(ids: ["visible"], fullAccess: true)
        XCTAssertEqual(Set(searcher.savedEmbedding.keys), ["visible"])
        let restarted = PhotoSearcher(indexingOperations: operations(), storageDirectory: dir)
        await restarted.prepareModelForSearch()
        XCTAssertEqual(Set(restarted.savedEmbedding.keys), ["visible"])
    }
}
