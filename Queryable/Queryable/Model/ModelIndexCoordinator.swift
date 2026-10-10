import Foundation
import CryptoKit
import Darwin

struct ModelIndexReference: Codable, Equatable, Sendable {
    let modelID: String
    let compatibilityIdentity: String
    let checkpointHash: String
    let generation: UUID

    init(spec: EmbeddingModelSpec, checkpointHash: String, generation: UUID = UUID()) {
        modelID = spec.modelID
        compatibilityIdentity = spec.compatibilityIdentity
        self.checkpointHash = checkpointHash
        self.generation = generation
    }
}

struct ModelIndexState: Codable, Equatable, Sendable {
    enum Phase: String, Codable, Sendable { case idle, building, paused, cancelled, failed }
    var active: ModelIndexReference?
    var requested: ModelIndexReference?
    var previous: ModelIndexReference?
    var phase: Phase = .idle
    var revision: UInt64 = 0
    var error: String?
}

struct ModelIndexActivationEvidence: Equatable, Sendable {
    let reference: ModelIndexReference
    let committedRevision: UInt64
    let complete: Bool
}

enum ModelIndexCoordinatorError: Error, LocalizedError {
    case corruptState
    case staleGeneration
    case incompleteIndex
    case invalidTransition
    case persistence(Int32)

    var errorDescription: String? {
        switch self {
        case .corruptState: return "Saved model recovery state is corrupt. Existing indexes were preserved."
        case .staleGeneration: return "This operation belongs to an earlier model generation."
        case .incompleteIndex: return "The replacement index has not committed all required work."
        case .invalidTransition: return "The requested model recovery action is not available."
        case .persistence(let code): return "Could not save model recovery state (system error \(code))."
        }
    }
}

/// Serialized by its owner (PhotoSearcher is main-actor isolated).
final class ModelIndexCoordinator {
    enum CommitBoundary: CaseIterable { case beforeWrite, afterWrite, afterFileSync, afterRename, afterDirectorySync }
    private struct Envelope: Codable {
        let version: Int
        let payload: Data
        let checksum: String
    }
    private let url: URL
    private let faultInjector: ((CommitBoundary) throws -> Void)?
    private(set) var state: ModelIndexState

    init(directoryURL: URL, faultInjector: ((CommitBoundary) throws -> Void)? = nil) throws {
        url = directoryURL.appendingPathComponent("model-index-state.json")
        self.faultInjector = faultInjector
        state = try Self.read(url) ?? ModelIndexState()
        if state.phase == .building { state.phase = .paused }
    }

    /// Explicit user recovery only. Never invoked by normal startup or selection.
    /// Keep the damaged manifest beside every retained index for later inspection.
    static func recoverCorruptState(directoryURL: URL) throws -> ModelIndexCoordinator {
        let source = directoryURL.appendingPathComponent("model-index-state.json")
        do {
            _ = try read(source)
            throw ModelIndexCoordinatorError.invalidTransition
        } catch ModelIndexCoordinatorError.corruptState {
            let quarantine = directoryURL.appendingPathComponent("model-index-state.corrupt-\(UUID().uuidString).json")
            guard Darwin.rename(source.path, quarantine.path) == 0 else {
                throw ModelIndexCoordinatorError.persistence(errno)
            }
            let descriptor = Darwin.open(directoryURL.path, O_RDONLY)
            guard descriptor >= 0 else { throw ModelIndexCoordinatorError.persistence(errno) }
            defer { Darwin.close(descriptor) }
            guard Darwin.fsync(descriptor) == 0 else { throw ModelIndexCoordinatorError.persistence(errno) }
            return try ModelIndexCoordinator(directoryURL: directoryURL)
        }
    }

    func isCurrentRequest(_ reference: ModelIndexReference) -> Bool { state.requested == reference }
    func isActive(_ reference: ModelIndexReference) -> Bool { state.active == reference }

    @discardableResult
    func request(spec: EmbeddingModelSpec, checkpointHash: String) throws -> ModelIndexReference {
        guard !checkpointHash.isEmpty else { throw ModelIndexCoordinatorError.invalidTransition }
        let reference = ModelIndexReference(spec: spec, checkpointHash: checkpointHash)
        var next = state
        next.requested = reference
        next.phase = .building
        next.error = nil
        try commit(next)
        return reference
    }

    func adoptActive(_ reference: ModelIndexReference) throws {
        guard state.active == nil, state.requested == nil else { throw ModelIndexCoordinatorError.invalidTransition }
        var next = state
        next.active = reference
        try commit(next)
    }

    func pause(_ reference: ModelIndexReference) throws { try transition(reference, to: .paused) }
    func resume(_ reference: ModelIndexReference) throws { try transition(reference, to: .building) }
    func cancel(_ reference: ModelIndexReference) throws { try transition(reference, to: .cancelled) }
    func fail(_ reference: ModelIndexReference, message: String) throws { try transition(reference, to: .failed, error: message) }

    func activate(_ reference: ModelIndexReference, evidence: ModelIndexActivationEvidence) throws {
        guard isCurrentRequest(reference) else { throw ModelIndexCoordinatorError.staleGeneration }
        guard state.phase == .building else { throw ModelIndexCoordinatorError.invalidTransition }
        guard evidence.reference == reference, evidence.committedRevision > 0, evidence.complete else {
            throw ModelIndexCoordinatorError.incompleteIndex
        }
        var next = state
        next.previous = next.active
        next.active = reference
        next.requested = nil
        next.phase = .idle
        next.error = nil
        try commit(next)
    }

    @discardableResult
    func rollback() throws -> ModelIndexReference {
        guard let previous = state.previous else { throw ModelIndexCoordinatorError.invalidTransition }
        var next = state
        next.previous = next.active
        next.active = previous
        next.requested = nil
        next.phase = .idle
        next.error = nil
        try commit(next)
        return previous
    }

    private func transition(_ reference: ModelIndexReference, to phase: ModelIndexState.Phase, error: String? = nil) throws {
        guard isCurrentRequest(reference) else { throw ModelIndexCoordinatorError.staleGeneration }
        var next = state
        next.phase = phase
        next.error = error
        try commit(next)
    }

    private static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func read(_ url: URL) throws -> ModelIndexState? {
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let bytes = try Data(contentsOf: url)
        do {
            let envelope = try JSONDecoder().decode(Envelope.self, from: bytes)
            guard envelope.version == 1, digest(envelope.payload) == envelope.checksum else {
                throw ModelIndexCoordinatorError.corruptState
            }
            let state = try JSONDecoder().decode(ModelIndexState.self, from: envelope.payload)
            let references = [state.active, state.requested, state.previous].compactMap { $0 }
            guard references.allSatisfy({ !$0.modelID.isEmpty && !$0.compatibilityIdentity.isEmpty && !$0.checkpointHash.isEmpty }),
                  (state.phase == .idle) == (state.requested == nil),
                  state.active == nil || state.active != state.requested,
                  state.previous == nil || state.previous != state.active else {
                throw ModelIndexCoordinatorError.corruptState
            }
            return state
        } catch { throw ModelIndexCoordinatorError.corruptState }
    }

    private func commit(_ proposed: ModelIndexState) throws {
        var next = proposed
        guard state.revision < UInt64.max else { throw ModelIndexCoordinatorError.invalidTransition }
        next.revision = state.revision + 1
        let payload = try JSONEncoder().encode(next)
        let bytes = try JSONEncoder().encode(Envelope(version: 1, payload: payload, checksum: Self.digest(payload)))
        let directory = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let temporary = directory.appendingPathComponent(".model-index-state-\(UUID().uuidString).tmp")
        defer { try? FileManager.default.removeItem(at: temporary) }
        var renamed = false
        do {
            try faultInjector?(.beforeWrite)
            try bytes.write(to: temporary)
            try faultInjector?(.afterWrite)
            let handle = try FileHandle(forWritingTo: temporary)
            do { try handle.synchronize(); try handle.close() }
            catch { try? handle.close(); throw error }
            try faultInjector?(.afterFileSync)
            guard Darwin.rename(temporary.path, url.path) == 0 else { throw ModelIndexCoordinatorError.persistence(errno) }
            renamed = true
            try faultInjector?(.afterRename)
            let descriptor = Darwin.open(directory.path, O_RDONLY)
            guard descriptor >= 0 else { throw ModelIndexCoordinatorError.persistence(errno) }
            defer { Darwin.close(descriptor) }
            guard Darwin.fsync(descriptor) == 0 else { throw ModelIndexCoordinatorError.persistence(errno) }
            try faultInjector?(.afterDirectorySync)
            state = next
        } catch {
            if renamed { state = try Self.read(url) ?? state }
            throw error
        }
    }
}
