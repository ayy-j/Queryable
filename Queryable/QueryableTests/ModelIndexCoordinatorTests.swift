import XCTest
@testable import Queryable

final class ModelIndexCoordinatorTests: XCTestCase {
    private enum Fault: Error { case injected }
    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }
    private func evidence(_ reference: ModelIndexReference, revision: UInt64 = 1, complete: Bool = true) -> ModelIndexActivationEvidence {
        ModelIndexActivationEvidence(reference: reference, committedRevision: revision, complete: complete)
    }

    func testRequestRestartResumeActivationAndRollbackPreserveActive() throws {
        let url = try directory()
        var coordinator = try ModelIndexCoordinator(directoryURL: url)
        let original = ModelIndexReference(spec: .mobileCLIP2S4, checkpointHash: "original")
        try coordinator.adoptActive(original)
        let first = try coordinator.request(spec: .mobileCLIP2S4, checkpointHash: "replacement")
        XCTAssertEqual(coordinator.state.active, original)
        coordinator = try ModelIndexCoordinator(directoryURL: url)
        XCTAssertEqual(coordinator.state.phase, .paused)
        XCTAssertEqual(coordinator.state.requested, first)
        try coordinator.resume(first)
        try coordinator.cancel(first)
        XCTAssertEqual(coordinator.state.active, original)
        try coordinator.resume(first)
        let second = try coordinator.request(spec: .mobileCLIP2S4, checkpointHash: "replacement")
        XCTAssertNotEqual(first.generation, second.generation)
        XCTAssertThrowsError(try coordinator.activate(first, evidence: evidence(first)))
        XCTAssertThrowsError(try coordinator.pause(first))
        XCTAssertThrowsError(try coordinator.activate(second, evidence: evidence(second, complete: false)))
        XCTAssertThrowsError(try coordinator.activate(second, evidence: evidence(second, revision: 0)))
        XCTAssertThrowsError(try coordinator.activate(second, evidence: evidence(first)))
        try coordinator.activate(second, evidence: evidence(second))
        coordinator = try ModelIndexCoordinator(directoryURL: url)
        XCTAssertEqual(coordinator.state.active, second)
        XCTAssertEqual(coordinator.state.previous, original)
        XCTAssertNil(coordinator.state.requested)
        XCTAssertEqual(try coordinator.rollback(), original)
        coordinator = try ModelIndexCoordinator(directoryURL: url)
        XCTAssertEqual(coordinator.state.active, original)
        XCTAssertEqual(coordinator.state.previous, second)
    }

    func testFailedAndPausedRequestsRequireExplicitResume() throws {
        let url = try directory()
        let coordinator = try ModelIndexCoordinator(directoryURL: url)
        let request = try coordinator.request(spec: .mobileCLIP2S4, checkpointHash: "checkpoint")
        try coordinator.fail(request, message: "Cloud asset unavailable")
        XCTAssertEqual(try ModelIndexCoordinator(directoryURL: url).state.error, "Cloud asset unavailable")
        XCTAssertThrowsError(try coordinator.activate(request, evidence: evidence(request)))
        try coordinator.resume(request)
        XCTAssertNil(coordinator.state.error)
        try coordinator.pause(request)
        XCTAssertThrowsError(try coordinator.activate(request, evidence: evidence(request)))
    }

    func testEveryActivationBoundaryRecoversOldOrNewCompleteStateRepeatedly() throws {
        for boundary in ModelIndexCoordinator.CommitBoundary.allCases {
            let url = try directory()
            let setup = try ModelIndexCoordinator(directoryURL: url)
            let original = ModelIndexReference(spec: .mobileCLIP2S4, checkpointHash: "old")
            try setup.adoptActive(original)
            let target = try setup.request(spec: .mobileCLIP2S4, checkpointHash: "new")
            var armed = false
            let coordinator = try ModelIndexCoordinator(directoryURL: url) { point in
                if armed && point == boundary { throw Fault.injected }
            }
            try coordinator.resume(target)
            armed = true
            XCTAssertThrowsError(try coordinator.activate(target, evidence: evidence(target)))
            let committed = boundary == .afterRename || boundary == .afterDirectorySync
            for _ in 0..<3 {
                let recovered = try ModelIndexCoordinator(directoryURL: url)
                XCTAssertEqual(recovered.state.active, committed ? target : original)
                XCTAssertEqual(recovered.state.previous, committed ? original : nil)
                XCTAssertEqual(recovered.state.requested, committed ? nil : target)
            }
            XCTAssertEqual(coordinator.state.active, committed ? target : original)
        }
    }

    func testRequestCommitBoundariesNeverEraseExistingActive() throws {
        for boundary in ModelIndexCoordinator.CommitBoundary.allCases {
            let url = try directory()
            let setup = try ModelIndexCoordinator(directoryURL: url)
            let active = ModelIndexReference(spec: .mobileCLIP2S4, checkpointHash: "old")
            try setup.adoptActive(active)
            let coordinator = try ModelIndexCoordinator(directoryURL: url) { point in
                if point == boundary { throw Fault.injected }
            }
            XCTAssertThrowsError(try coordinator.request(spec: .mobileCLIP2S4, checkpointHash: "new"))
            let recovered = try ModelIndexCoordinator(directoryURL: url)
            XCTAssertEqual(recovered.state.active, active)
            XCTAssertEqual(recovered.state.requested != nil, boundary == .afterRename || boundary == .afterDirectorySync)
        }
    }

    func testCorruptionDoesNotDefaultOrOverwriteAndAdoptionIsOneTime() throws {
        let url = try directory()
        let coordinator = try ModelIndexCoordinator(directoryURL: url)
        XCTAssertNil(coordinator.state.active)
        let active = ModelIndexReference(spec: .mobileCLIP2S4, checkpointHash: "old")
        try coordinator.adoptActive(active)
        XCTAssertThrowsError(try coordinator.adoptActive(active))
        let stateURL = url.appendingPathComponent("model-index-state.json")
        let invalid = Data("corrupt".utf8)
        try invalid.write(to: stateURL)
        XCTAssertThrowsError(try ModelIndexCoordinator(directoryURL: url)) { error in
            guard case ModelIndexCoordinatorError.corruptState = error else { return XCTFail("Unexpected \(error)") }
        }
        XCTAssertEqual(try Data(contentsOf: stateURL), invalid)
    }

    func testChecksumDetectsPayloadTampering() throws {
        let url = try directory()
        let coordinator = try ModelIndexCoordinator(directoryURL: url)
        _ = try coordinator.request(spec: .mobileCLIP2S4, checkpointHash: "checkpoint")
        let stateURL = url.appendingPathComponent("model-index-state.json")
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: stateURL)) as? [String: Any])
        json["checksum"] = String(repeating: "0", count: 64)
        try JSONSerialization.data(withJSONObject: json).write(to: stateURL)
        XCTAssertThrowsError(try ModelIndexCoordinator(directoryURL: url))
    }
    func testExplicitCorruptionRecoveryQuarantinesOnlyCorruptManifest() throws {
        let url = try directory()
        let stateURL = url.appendingPathComponent("model-index-state.json")
        let indexURL = url.appendingPathComponent("preserved-index.qemb")
        let damaged = Data("damaged manifest".utf8)
        let index = Data("committed index".utf8)
        try damaged.write(to: stateURL)
        try index.write(to: indexURL)
        let recovered = try ModelIndexCoordinator.recoverCorruptState(directoryURL: url)
        XCTAssertNil(recovered.state.active)
        let quarantines = try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.hasPrefix("model-index-state.corrupt-") }
        XCTAssertEqual(quarantines.count, 1)
        XCTAssertEqual(try Data(contentsOf: XCTUnwrap(quarantines.first)), damaged)
        XCTAssertEqual(try Data(contentsOf: indexURL), index)
        _ = try recovered.request(spec: .mobileCLIP2S4, checkpointHash: "replacement")
        XCTAssertThrowsError(try ModelIndexCoordinator.recoverCorruptState(directoryURL: url))
        XCTAssertNotNil(try ModelIndexCoordinator(directoryURL: url).state.requested)
    }

    func testExplicitCorruptionRecoveryPreservesReadFailuresAndMissingState() throws {
        let url = try directory()
        XCTAssertThrowsError(try ModelIndexCoordinator.recoverCorruptState(directoryURL: url))
        let stateURL = url.appendingPathComponent("model-index-state.json")
        try FileManager.default.createDirectory(at: stateURL, withIntermediateDirectories: false)
        XCTAssertThrowsError(try ModelIndexCoordinator.recoverCorruptState(directoryURL: url))
        var isDirectory: ObjCBool = false
        XCTAssertTrue(FileManager.default.fileExists(atPath: stateURL.path, isDirectory: &isDirectory))
        XCTAssertTrue(isDirectory.boolValue)
        let files = try FileManager.default.contentsOfDirectory(atPath: url.path)
        XCTAssertEqual(files, ["model-index-state.json"])
    }

}
