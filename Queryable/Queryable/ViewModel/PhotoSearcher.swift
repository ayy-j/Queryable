//
//    PhotoSearcher.swift
//    Places
//    Created by Ke Fang on 2022/12/14.

import Foundation
import OSLog
import UIKit
import Photos
import CoreML
import Accelerate

// search result code.
 enum SEARCH_RESULT_CODE: Int {
    case DEFAULT         = -4
    case MODEL_PREPARED  = -3
    case IS_SEARCHING    = -2
    case NEVER_INDEXED   = -1
    case NO_RESULT       = 0
    case HAS_RESULT      = 1
    case SEARCH_ERROR    = 2
}

// build index code.
enum BUILD_INDEX_CODE: Int {
    case DEFAULT             = -3
    case LOADING_PHOTOS      = -2
    case PHOTOS_LOADED       = -1
    case LOADING_MODEL       = 0
    case MODEL_ERROR         = 1
    case IS_BUILDING_INDEX   = 2
    case BUILD_FINISHED      = 3
    case BUILD_INCOMPLETE    = 4
    case BUILD_ERROR         = 5
    case BUILD_PAUSED        = 6
    case BUILD_CANCELLED     = 7
}

/// Injectable I/O boundaries let failure tests exercise the real indexing flow
/// without reading Photos, loading a model, or changing the user's saved index.
struct PhotoIndexingOperations {
    var fetchImage: (PhotoAsset, CGSize) async throws -> UIImage
    var encodeBatch: ([UIImage]) throws -> [MLShapedArray<Float32>]
    var encodeImage: (UIImage) async throws -> MLShapedArray<Float32>
    var save: ([String: MLMultiArray]) throws -> Void
}

enum PhotoIndexingError: Error {
    case encoderNotReady
    case invalidEmbedding
}

@MainActor
class PhotoSearcher: ObservableObject {
    let defaults = UserDefaults.standard
    let photoCollection = PhotoCollection(smartAlbum: .smartAlbumUserLibrary)
    var photoSearchModel: PhotoSearcherModel
    private var modelSpec: EmbeddingModelSpec
    private var indexingSpec: EmbeddingModelSpec
    let KEY_HAS_ACCESS_TO_PHOTOS = "KEY_HAS_ACCESS_TO_PHOTOS"

    // -3: default, -2: Is searching now, -1: Never indexed. 0: No result. 1: Has result.
    @Published var searchResultCode: SEARCH_RESULT_CODE = .DEFAULT
    @Published var buildIndexCode: BUILD_INDEX_CODE = .DEFAULT
    /// Actionable message when model files are missing/unreadable; nil otherwise.
    @Published var modelErrorMessage: String? = nil
    @Published var searchErrorMessage: String?
    @Published var similarPhotoErrorMessage: String?
    @Published var totalUnIndexedPhotosNum: Int = -1
    @Published var curIndexingNums: Int = 0
    @Published var savedIndexingPhotosNum: Int = 0
    @Published var failedIndexingPhotosNum: Int = 0
    @Published var remainingIndexingPhotosNum: Int = 0
    @Published var blankEmbeddingPhotosNum: Int = 0
    @Published var indexingErrorMessage: String?
    @Published var curShowingPhoto: UIImage = UIImage(systemName: "photo")!

    @Published var isFindingSimilarPhotos = false
    @Published var similarPhotoAssets = [PhotoAsset]()
    @Published var searchResultPhotoAssets = [PhotoAsset]()
    @Published var searchString: String = ""

    private(set) var savedEmbedding = [String: MLMultiArray]()
    private(set) var buildingEmbedding = [String: MLMultiArray]()
    private var curIndexingPhoto: UIImage = UIImage(systemName: "photo")!
    private var imageRequestID: PHImageRequestID?
    private var allPhotosId = [String: Int]()
    private var unIndexedPhotos = [PhotoAsset]()
    private var imageEncoder: ImgEncoder?
    private var totalPhotosNum = -1
    private let BUILD_INDEX_FRAGMENT_LENGTH = 100
    private let SAVE_EMBEDDING_EVERY = 5000
    private let indexingOperations: PhotoIndexingOperations?
    private var isBuildingIndex = false
    @Published private(set) var isIndexing = false
    @Published private(set) var activeModelName = "No active index"
    @Published private(set) var requestedModelName: String?
    @Published private(set) var recoveryMessage: String?
    @Published private(set) var canRollback = false
    private var epoch = UUID()
    private var indexingTask: Task<Void, Never>?
    private var coordinator: ModelIndexCoordinator?
    private var activeReference: ModelIndexReference?
    private var targetReference: ModelIndexReference?
    private var targetStore: EmbeddingStore?
    private var targetEmbedding = [String: MLMultiArray]()
    private var checkpoint = IndexingCheckpoint()
    private var activeVersions = [String: Date]()
    private var pendingVersions = [String: Date]()
    private let storageDirectory: URL
    private let persistentTesting: Bool
    private var isLoadingImageEncoder = false
    private var hasLibrarySnapshot = false
    private var fullyAuthorized = false
    private var resourceURL: URL? {
        Bundle.main.url(forResource: "CoreMLModels", withExtension: nil)
    }
    private var workEmbedding: [String: MLMultiArray] {
        targetReference == nil ? savedEmbedding : targetEmbedding
    }

    struct IndexingCheckpoint: Codable {
        var requiredIDs = Set<String>()
        var checkedIDs = Set<String>()
        var versions = [String: Date]()
        var complete = false
    }

    private func checkEpoch(_ captured: UUID) throws {
        try Task.checkCancellation()
        guard captured == epoch else { throw CancellationError() }
    }

    private func publishCoordinatorState() {
        activeModelName = coordinator?.state.active?.modelID ?? "No active index"
        requestedModelName = coordinator?.state.requested?.modelID
        canRollback = coordinator?.state.previous != nil
    }

    private func resolveSpec(_ reference: ModelIndexReference) throws -> EmbeddingModelSpec {
        guard let spec = [EmbeddingModelSpec.mobileCLIP2S4, .mobileCLIPS2].first(where: {
            $0.modelID == reference.modelID && $0.compatibilityIdentity == reference.compatibilityIdentity
        }) else { throw EmbeddingStoreError.notReady }
        return spec
    }

    private func store(for reference: ModelIndexReference, spec: EmbeddingModelSpec) -> EmbeddingStore {
        EmbeddingStore(spec: spec, checkpointHash: reference.checkpointHash,
                       directory: storageDirectory, generation: reference.generation)
    }

    private func restoreTarget(_ reference: ModelIndexReference) throws {
        indexingSpec = try resolveSpec(reference)
        targetReference = reference
        targetStore = store(for: reference, spec: indexingSpec)
        let snapshot = try targetStore!.load()
        targetEmbedding = snapshot?.embeddings ?? [:]
        checkpoint = try snapshot?.checkpoint.map { try JSONDecoder().decode(IndexingCheckpoint.self, from: $0) } ?? IndexingCheckpoint()
        savedIndexingPhotosNum = targetEmbedding.count
        remainingIndexingPhotosNum = checkpoint.requiredIDs.subtracting(targetEmbedding.keys).count
        curIndexingNums = checkpoint.checkedIDs.count
        totalUnIndexedPhotosNum = checkpoint.requiredIDs.count
    }

    private func publishActive(_ reference: ModelIndexReference) throws {
        let spec = try resolveSpec(reference)
        let activeStore = store(for: reference, spec: spec)
        guard let snapshot = try activeStore.load(), snapshot.generation == reference.generation else {
            throw EmbeddingStoreError.notReady
        }
        modelSpec = spec
        photoSearchModel = PhotoSearcherModel(spec: spec)
        activeReference = reference
        embeddingStore = activeStore
        savedEmbedding = snapshot.embeddings
        activeVersions = try snapshot.checkpoint.map { try JSONDecoder().decode(IndexingCheckpoint.self, from: $0).versions } ?? [:]
        gpuSearch = GPUSimilaritySearch(embeddingDimension: spec.embeddingDimension)
        do { try gpuSearch?.buildIndex(from: savedEmbedding) } catch { gpuSearch = nil }
        searchResultPhotoAssets.removeAll()
        similarPhotoAssets.removeAll()
        searchResultCode = savedEmbedding.isEmpty ? .NEVER_INDEXED : .MODEL_PREPARED
        publishCoordinatorState()
    }

    func startIndexing() {
        guard indexingTask == nil else { return }
        indexingTask = Task { [weak self] in
            guard let self else { return }
            await self.buildIndex()
            self.indexingTask = nil
        }
    }

    func pauseIndexing() {
        epoch = UUID()
        indexingTask?.cancel()
        imageEncoder = nil
        photoSearchModel.releaseTextEncoder()
        ImgEncoder.flushBufferPool()
        UIApplication.shared.isIdleTimerDisabled = false
        buildIndexCode = .BUILD_PAUSED
        recoveryMessage = "Paused. Committed photos are safe; resume checks the current photo library."
        if let ref = targetReference {
            do { try coordinator?.pause(ref) } catch { recoveryMessage = "Pause state could not be saved: \(error.localizedDescription). Work has stopped in this session." }
        }
    }

    func pauseForBackground() {
        if isBuildingIndex || buildIndexCode == .LOADING_MODEL { pauseIndexing() }
        photoSearchModel.releaseTextEncoder()
        imageEncoder = nil
        ImgEncoder.flushBufferPool()
    }

    func cancelIndexing() {
        pauseIndexing()
        buildingEmbedding.removeAll()
        buildIndexCode = .BUILD_CANCELLED
        recoveryMessage = "Cancelled. Saved progress and the active index are retained. Resume or start a new rebuild."
        if let ref = targetReference {
            do { try coordinator?.cancel(ref) } catch { recoveryMessage = "Cancellation state could not be saved: \(error.localizedDescription). Work has stopped in this session." }
        }
    }

    func resumeIndexing() async {
        guard !isBuildingIndex else { return }
        do {
            if let ref = targetReference { try coordinator?.resume(ref) }
            await fetchPhotos()
            guard buildIndexCode == .PHOTOS_LOADED else { return }
            await loadImageIncoder()
            if buildIndexCode == .IS_BUILDING_INDEX { startIndexing() }
        } catch { reportRecovery(error) }
    }

    func restartIndexing() async {
        await requestModel(indexingSpec)
    }

    func rollbackModel() async {
        guard let coordinator, let previous = coordinator.state.previous else { return }
        pauseIndexing()
        do {
            let spec = try resolveSpec(previous)
            guard let url = resourceURL,
                  try spec.checkpointHash(resourcesAt: url) == previous.checkpointHash,
                  try store(for: previous, spec: spec).load() != nil else { throw EmbeddingStoreError.notReady }
            let validation = try TextEncoder(resourcesAt: url, spec: spec)
            _ = validation
            let restored = try coordinator.rollback()
            try publishActive(restored)
            activeReference = restored
            targetReference = nil; self.targetStore = nil; targetEmbedding.removeAll(); buildingEmbedding.removeAll()
            indexingSpec = spec
            recoveryMessage = "Previous index restored."
        } catch { reportRecovery(error) }
    }

    private func reportRecovery(_ error: Error) {
        recoveryMessage = "Saved index recovery needs attention: \(error.localizedDescription). Existing files are retained; retry or explicitly rebuild."
        indexingErrorMessage = recoveryMessage
        buildIndexCode = .BUILD_ERROR
    }

    func requestModel(_ spec: EmbeddingModelSpec) async {
        pauseIndexing()
        await indexingTask?.value
        let captured = epoch
        do {
            guard let url = resourceURL else { throw EmbeddingStoreError.notReady }
            let hash = try await Task.detached(priority: .utility) { try spec.checkpointHash(resourcesAt: url) }.value
            try checkEpoch(captured)
            // Validate both tower contracts before recording a requested switch.
            // Scope each validation so towers need not remain resident together.
            do { _ = try ImgEncoder(resourcesAt: url, spec: spec) }
            do { _ = try TextEncoder(resourcesAt: url, spec: spec) }
            try beginRequest(spec: spec, checkpointHash: hash)
            recoveryMessage = "Rebuild requested. The active index remains available until all required photos are saved."
            await resumeIndexing()
        } catch { reportRecovery(error) }
    }

    // Internal seam used by generated-vector restart tests; production verifies artifacts first.
    func beginRequest(spec: EmbeddingModelSpec, checkpointHash: String) throws {
        epoch = UUID()
        if coordinator == nil { coordinator = try ModelIndexCoordinator(directoryURL: storageDirectory) }
        let reference = try coordinator!.request(spec: spec, checkpointHash: checkpointHash)
        buildingEmbedding.removeAll()
        pendingVersions.removeAll()
        try restoreTarget(reference)
        publishCoordinatorState()
    }

    /// GPU-accelerated similarity search (Float16 MPSGraph matmul)
    private var gpuSearch: GPUSimilaritySearch?
    /// Efficient binary embedding storage with incremental saves
    private var embeddingStore: EmbeddingStore?

    @Published var TOPK_SIM: Int {
        didSet {
            UserDefaults.standard.set(TOPK_SIM, forKey: "TOPK_SIM")
        }
    }

    init(modelSpec: EmbeddingModelSpec = .appDefault,
         indexingOperations: PhotoIndexingOperations? = nil,
         storageDirectory: URL? = nil) {
        self.modelSpec = modelSpec
        self.indexingSpec = modelSpec
        self.storageDirectory = storageDirectory ?? URL.documentsDirectory
        self.persistentTesting = storageDirectory != nil
        self.photoSearchModel = PhotoSearcherModel(spec: modelSpec)
        self.indexingOperations = indexingOperations
        let defaultTOPK_SIM = UserDefaults.standard.object(forKey: "TOPK_SIM") as? Int ?? 120
        self.TOPK_SIM = defaultTOPK_SIM
        self.gpuSearch = GPUSimilaritySearch(embeddingDimension: modelSpec.embeddingDimension)
    }

    func changeState(from statusCode1: BUILD_INDEX_CODE, to statusCode2: BUILD_INDEX_CODE) {
        if self.buildIndexCode == statusCode1 {
            self.buildIndexCode = statusCode2
        }
    }


    func prepareModelForSearch() async {
        guard !isBuildingIndex, !isLoadingImageEncoder else { return }
        let captured = epoch
        do {
            if coordinator == nil {
                do {
                    coordinator = try ModelIndexCoordinator(directoryURL: storageDirectory)
                } catch ModelIndexCoordinatorError.corruptState {
                    // Never auto-replace a damaged coordinator file: surface it and
                    // wait for the explicit user recovery action instead.
                    recoveryMessage = "Saved model state is damaged and was preserved. Use Repair to quarantine it and start fresh, or reinstall without deleting indexes."
                    buildIndexCode = .BUILD_ERROR
                    publishCoordinatorState()
                    return
                }
            }
            guard let coordinator else { return }
            if let active = coordinator.state.active {
                let spec = try resolveSpec(active)
                if !persistentTesting {
                    guard let url = resourceURL else { throw EmbeddingStoreError.notReady }
                    let hash = try await Task.detached(priority: .utility) { try spec.checkpointHash(resourcesAt: url) }.value
                    try checkEpoch(captured)
                    guard hash == active.checkpointHash else { throw EmbeddingStoreError.notReady }
                }
                try publishActive(active)
            } else if coordinator.state.requested == nil {
                guard let url = resourceURL else { throw EmbeddingStoreError.notReady }
                let spec = modelSpec
                let hash = try await Task.detached(priority: .utility) { try spec.checkpointHash(resourcesAt: url) }.value
                try checkEpoch(captured)
                let legacy = EmbeddingStore(spec: spec, checkpointHash: hash, directory: storageDirectory)
                if let snapshot = try legacy.load() {
                    let reference = ModelIndexReference(spec: spec, checkpointHash: hash)
                    let destination = store(for: reference, spec: spec)
                    _ = try destination.commit(upserts: snapshot.embeddings, checkpoint: nil,
                                               generation: reference.generation)
                    try coordinator.adoptActive(reference)
                    try publishActive(reference)
                } else { searchResultCode = .NEVER_INDEXED }
            }
            if let requested = coordinator.state.requested {
                try restoreTarget(requested)
                buildIndexCode = coordinator.state.phase == .cancelled ? .BUILD_CANCELLED : .BUILD_PAUSED
                recoveryMessage = "Saved rebuild found. Resume to check the current library and continue."
            }
            publishCoordinatorState()
        } catch is CancellationError { return }
        catch { reportRecovery(error) }
    }

    /// Explicit user recovery for a damaged coordinator file. Quarantines only
    /// the corrupt manifest beside the retained indexes, then resets to empty state.
    func repairCorruptModelState() {
        pauseIndexing()
        do {
            coordinator = try ModelIndexCoordinator.recoverCorruptState(directoryURL: storageDirectory)
            activeReference = nil
            targetReference = nil
            targetStore = nil
            targetEmbedding.removeAll()
            buildingEmbedding.removeAll()
            checkpoint = IndexingCheckpoint()
            searchResultCode = .NEVER_INDEXED
            buildIndexCode = .DEFAULT
            recoveryMessage = "Damaged model state was quarantined; indexes were preserved. Select a model to start a fresh rebuild."
            publishCoordinatorState()
        } catch {
            reportRecovery(error)
        }
    }

    func fetchPhotos() async {
        let captured = epoch
        buildIndexCode = .LOADING_PHOTOS
        await photoCollection.cache.requestOptions.isNetworkAccessAllowed = false
        let authorized = await PhotoLibrary.checkAuthorization()
        guard captured == epoch else { return }
        guard authorized else {
            hasLibrarySnapshot = false
            recoveryMessage = "Photo access is unavailable. Restore access in Settings, then resume. Saved records are retained."
            buildIndexCode = .BUILD_PAUSED
            return
        }
        do {
            try await photoCollection.load()
            try checkEpoch(captured)
            fullyAuthorized = PHPhotoLibrary.authorizationStatus(for: .readWrite) == .authorized
            hasLibrarySnapshot = true
            defaults.set(true, forKey: KEY_HAS_ACCESS_TO_PHOTOS)
            totalPhotosNum = photoCollection.photoAssets.count
            try await fetchUnIndexedPhotos()
            buildIndexCode = .PHOTOS_LOADED
        } catch is CancellationError { return }
        catch { reportRecovery(error) }
    }

    func loadImageIncoder() async {
        guard !isLoadingImageEncoder, !isBuildingIndex else { return }
        isLoadingImageEncoder = true
        defer { isLoadingImageEncoder = false }
        let captured = epoch
        buildIndexCode = .LOADING_MODEL
        modelErrorMessage = nil
        do {
            guard let url = resourceURL else { throw EmbeddingStoreError.notReady }
            if targetReference == nil {
                let spec = modelSpec
                let hash = try await Task.detached(priority: .utility) { try spec.checkpointHash(resourcesAt: url) }.value
                try checkEpoch(captured)
                try beginRequest(spec: spec, checkpointHash: hash)
                // Updating the current model retains its already committed vectors
                // and their durable checkpoint (required IDs, versions, completion).
                // Without the checkpoint bytes, resume would lose requiredIDs and
                // could declare completion while photos are still missing.
                if let activeStore = embeddingStore, let committed = try activeStore.load() {
                    targetEmbedding = committed.embeddings
                    if let data = committed.checkpoint {
                        checkpoint = try JSONDecoder().decode(IndexingCheckpoint.self, from: data)
                    } else {
                        checkpoint.versions = activeVersions
                    }
                    if !targetEmbedding.isEmpty {
                        _ = try targetStore!.commit(upserts: targetEmbedding,
                                                    checkpoint: try JSONEncoder().encode(checkpoint),
                                                    generation: targetReference!.generation)
                    }
                } else {
                    targetEmbedding = savedEmbedding
                    checkpoint.versions = activeVersions
                }
            }
            guard let ref = targetReference else { throw EmbeddingStoreError.notReady }
            let loadingEpoch = epoch
            let spec = indexingSpec
            let hash = try await Task.detached(priority: .utility) { try spec.checkpointHash(resourcesAt: url) }.value
            try checkEpoch(loadingEpoch)
            guard hash == ref.checkpointHash else { throw EmbeddingStoreError.incompatible }
            try coordinator?.resume(ref)
            photoSearchModel.releaseTextEncoder()
            imageEncoder = try ImgEncoder(resourcesAt: url, spec: indexingSpec)
            try await fetchUnIndexedPhotos()
            buildIndexCode = .IS_BUILDING_INDEX
        } catch is CancellationError { return }
        catch {
            imageEncoder = nil
            modelErrorMessage = error.localizedDescription
            buildIndexCode = .MODEL_ERROR
        }
    }

    private func validatedEmbedding(_ embedding: MLShapedArray<Float32>) throws -> MLMultiArray {
        let values = embedding.scalars
        guard values.count == indexingSpec.embeddingDimension,
              embedding.shape == [1, indexingSpec.embeddingDimension] || embedding.shape == [indexingSpec.embeddingDimension],
              values.allSatisfy({ $0.isFinite }), values.contains(where: { $0 != 0 }) else {
            throw PhotoIndexingError.invalidEmbedding
        }
        // Copy scalar values rather than retaining the prediction's IOSurface.
        return MLMultiArray(MLShapedArray<Float32>(scalars: values, shape: [1, indexingSpec.embeddingDimension]))
    }

    private func hasUsableEmbedding(for id: String) -> Bool {
        guard let value = workEmbedding[id], value.dataType == .float32,
              value.count == indexingSpec.embeddingDimension else { return false }
        let values = MLShapedArray<Float32>(converting: value).scalars
        return values.allSatisfy { $0.isFinite } && values.contains { $0 != 0 }
    }

    func batchBuildIndex(assets: [PhotoAsset]) async throws {
        let captured = epoch
        let targetSize = CGSize(width: indexingSpec.imageSize, height: indexingSpec.imageSize)
        let operations: PhotoIndexingOperations
        if let injected = indexingOperations {
            operations = injected
        } else {
            guard let encoder = imageEncoder else { throw PhotoIndexingError.encoderNotReady }
            let cache = photoCollection.cache
            operations = PhotoIndexingOperations(
                fetchImage: { try await cache.imageForIndexing(for: $0, targetSize: $1) },
                encodeBatch: { try encoder.encodeBatch(images: $0) },
                encodeImage: { try await encoder.encode(image: $0) },
                save: { _ in }
            )
        }
        await photoCollection.cache.startCaching(for: assets, targetSize: targetSize)
        do {
            for batchStart in stride(from: 0, to: assets.count, by: 32) {
                try checkEpoch(captured)
                if indexingOperations == nil && (ProcessInfo.processInfo.isLowPowerModeEnabled || ProcessInfo.processInfo.thermalState == .serious || ProcessInfo.processInfo.thermalState == .critical) {
                    pauseIndexing()
                    throw CancellationError()
                }
                let batchAssets = Array(assets[batchStart..<min(batchStart + 32, assets.count)])
                var images = [(PhotoAsset, UIImage)]()
                try await withThrowingTaskGroup(of: (PhotoAsset, UIImage?).self) { group in
                    for asset in batchAssets {
                        group.addTask {
                            do { return (asset, try await operations.fetchImage(asset, targetSize)) }
                            catch is CancellationError { throw CancellationError() }
                            catch { return (asset, nil) }
                        }
                    }
                    for try await (asset, image) in group {
                        try checkEpoch(captured)
                        if let image { images.append((asset, image)) }
                        else { self.failedIndexingPhotosNum += 1 }
                    }
                }
                try checkEpoch(captured)
                if !images.isEmpty {
                    do {
                        let embeddings = try autoreleasepool {
                            try operations.encodeBatch(images.map { $0.1 })
                        }
                        guard embeddings.count == images.count else { throw PhotoIndexingError.invalidEmbedding }
                        for (index, (asset, _)) in images.enumerated() {
                            do {
                                buildingEmbedding[asset.id] = try validatedEmbedding(embeddings[index])
                                if let date = asset.phAsset?.modificationDate { pendingVersions[asset.id] = date }
                            }
                            catch { failedIndexingPhotosNum += 1 }
                        }
                    } catch {
                        // A batch failure must not discard images that work individually.
                        for (asset, image) in images {
                            try checkEpoch(captured)
                            do {
                                let embedding = try await operations.encodeImage(image)
                                try checkEpoch(captured)
                                buildingEmbedding[asset.id] = try validatedEmbedding(embedding)
                                if let date = asset.phAsset?.modificationDate { pendingVersions[asset.id] = date }
                            } catch is CancellationError { throw CancellationError() }
                            catch { failedIndexingPhotosNum += 1 }
                        }
                    }
                    curIndexingPhoto = images.last!.1
                }
                checkpoint.checkedIDs.formUnion(batchAssets.map(\.id))
                curIndexingNums += batchAssets.count
                curShowingPhoto = curIndexingPhoto
                images.removeAll()
                ImgEncoder.flushBufferPool()
            }
        } catch {
            await photoCollection.cache.stopCaching(for: assets, targetSize: targetSize)
            ImgEncoder.flushBufferPool()
            throw error
        }
        await photoCollection.cache.stopCaching(for: assets, targetSize: targetSize)
    }

    func fetchUnIndexedPhotos() async throws {
        let startingTime = Date()
        self.unIndexedPhotos = [PhotoAsset]()
        self.allPhotosId.removeAll()
        self.blankEmbeddingPhotosNum = 0
        var photoIdSet = Set<String>(minimumCapacity: photoCollection.photoAssets.count)

        for idx in 0..<self.photoCollection.photoAssets.count {
            let asset = self.photoCollection.photoAssets[idx]
            self.allPhotosId[asset.id] = 1
            photoIdSet.insert(asset.id)
            let knownVersion = targetReference == nil ? activeVersions[asset.id] : checkpoint.versions[asset.id]
            if let knownVersion, let current = asset.phAsset?.modificationDate, current != knownVersion {
                try invalidateEditedPhotos([asset.id])
            }
            if !hasUsableEmbedding(for: asset.id) {
                self.unIndexedPhotos.append(asset)
                if workEmbedding[asset.id] != nil { blankEmbeddingPhotosNum += 1 }
            }
        }

        if hasLibrarySnapshot {
            try reconcileLibrary(ids: photoIdSet, fullAccess: fullyAuthorized)
        }

        self.totalUnIndexedPhotosNum = self.unIndexedPhotos.count
        print("\(startingTime.timeIntervalSinceNow * -1) seconds used for filter unindex photos")
    }

    private func invalidateEditedPhotos(_ ids: Set<String>) throws {
        if let ref = targetReference, let targetStore {
            for id in ids { checkpoint.versions.removeValue(forKey: id) }
            checkpoint.complete = false
            _ = try targetStore.commit(deleting: ids, checkpoint: try JSONEncoder().encode(checkpoint), generation: ref.generation)
            for id in ids { targetEmbedding.removeValue(forKey: id) }
        }
        if let activeStore = embeddingStore, let activeRef = activeReference {
            _ = try activeStore.commit(deleting: Set(ids), checkpoint: nil, generation: activeRef.generation)
            guard let snapshot = try activeStore.load(), snapshot.generation == activeRef.generation else {
                throw EmbeddingStoreError.notReady
            }
            for id in ids { savedEmbedding.removeValue(forKey: id); activeVersions.removeValue(forKey: id) }
            gpuSearch?.removeEmbeddings(ids)
        }
    }

    /// Reconciliation is also a test seam for additions, removals, and limited authorization.
    func reconcileLibrary(ids: Set<String>, fullAccess: Bool) throws {
        hasLibrarySnapshot = true
        fullyAuthorized = fullAccess
        allPhotosId = Dictionary(uniqueKeysWithValues: ids.map { ($0, 1) })
        let deleted = fullAccess ? Set(workEmbedding.keys).subtracting(ids) : []
        buildingEmbedding = buildingEmbedding.filter { ids.contains($0.key) }
        checkpoint.requiredIDs = ids
        checkpoint.complete = false
        if let ref = targetReference, let targetStore {
            _ = try targetStore.commit(deleting: deleted, checkpoint: try JSONEncoder().encode(checkpoint),
                                       generation: ref.generation)
            for id in deleted { targetEmbedding.removeValue(forKey: id) }
        }
        // Deletions affect the old active index as well; permissions never do.
        let activeDeleted = fullAccess ? Set(savedEmbedding.keys).subtracting(ids) : []
        if !activeDeleted.isEmpty, let activeStore = embeddingStore, let activeRef = activeReference {
            _ = try activeStore.commit(deleting: activeDeleted, checkpoint: nil, generation: activeRef.generation)
            guard let snapshot = try activeStore.load(), snapshot.generation == activeRef.generation else {
                throw EmbeddingStoreError.notReady
            }
            for id in activeDeleted { savedEmbedding.removeValue(forKey: id) }
            gpuSearch?.removeEmbeddings(activeDeleted)
        }
        searchResultPhotoAssets.removeAll { !ids.contains($0.id) }
        similarPhotoAssets.removeAll { !ids.contains($0.id) }
    }

    func deleteEmbeddingByAsset(asset: PhotoAsset) async {
        // Confirm deletion through a fresh library snapshot; failed Photos deletes
        // must not silently remove a searchable vector.
        await fetchPhotos()
    }

    func updateEmbedding(new_indexed_results: [String: MLMultiArray]) throws {
        guard let activeStore = embeddingStore, let activeRef = activeReference else { throw EmbeddingStoreError.notReady }
        print("Before update, embedding count=\(self.savedEmbedding.count)")

        // Typed commit is the single visibility boundary; verify the committed
        // revision before publishing in memory so a failed write never mutates state.
        let revision = try activeStore.commit(upserts: new_indexed_results, checkpoint: nil, generation: activeRef.generation)
        guard let snapshot = try activeStore.load(), snapshot.revision == revision,
              snapshot.generation == activeRef.generation else { throw EmbeddingStoreError.writeFailed }

        for (key, value) in new_indexed_results {
            self.savedEmbedding[key] = value
        }
        // The journal is already saved. If acceleration fails, keep that work
        // searchable through the existing CPU path instead of reporting a save failure.
        do {
            try gpuSearch?.addEmbeddings(new_indexed_results)
        } catch {
            gpuSearch = nil
            logger.error("GPU index update failed; using CPU search: \(error.localizedDescription)")
        }
        print("After update, embedding count=\(self.savedEmbedding.count)")
        print("Embedding saved (incremental)")
    }

    /**
     Build index
     */
    func buildIndex() async {
        await buildIndex(assets: unIndexedPhotos.filter { !hasUsableEmbedding(for: $0.id) })
    }

    func buildIndex(assets: [PhotoAsset]) async {
        guard !isBuildingIndex else { return }
        isBuildingIndex = true
        isIndexing = true
        let captured = epoch
        UIApplication.shared.isIdleTimerDisabled = true
        defer {
            isIndexing = false
            UIApplication.shared.isIdleTimerDisabled = false
            isBuildingIndex = false
            imageEncoder = nil
            ImgEncoder.flushBufferPool()
        }
        buildIndexCode = .IS_BUILDING_INDEX
        indexingErrorMessage = nil
        curIndexingNums = targetReference == nil ? 0 : checkpoint.checkedIDs.count
        savedIndexingPhotosNum = targetReference == nil ? 0 : targetEmbedding.count
        failedIndexingPhotosNum = 0
        totalUnIndexedPhotosNum = assets.count
        remainingIndexingPhotosNum = assets.count
        if !hasLibrarySnapshot { checkpoint.requiredIDs = Set(assets.map(\.id)).union(workEmbedding.keys) }
        checkpoint.complete = false

        do {
            if let ref = targetReference { try coordinator?.resume(ref) }
            // A save retry commits retained work before attempting more photos.
            let retainedCount = buildingEmbedding.count
            try persistBuildingEmbeddings()
            curIndexingNums += retainedCount
            let pendingAssets = assets.filter { !hasUsableEmbedding(for: $0.id) }
            for index in stride(from: 0, to: pendingAssets.count, by: BUILD_INDEX_FRAGMENT_LENGTH) {
                let end = min(index + BUILD_INDEX_FRAGMENT_LENGTH, pendingAssets.count)
                try await batchBuildIndex(assets: Array(pendingAssets[index..<end]))
                try checkEpoch(captured)
                if targetReference != nil || buildingEmbedding.count >= SAVE_EMBEDDING_EVERY {
                    try persistBuildingEmbeddings()
                }
                // Give the Neural Engine runtime a chance to reclaim allocations.
                if curIndexingNums % 500 < BUILD_INDEX_FRAGMENT_LENGTH {
                    try await Task.sleep(nanoseconds: 100_000_000)
                }
            }
            try checkEpoch(captured)
            try persistBuildingEmbeddings()
            if indexingOperations == nil {
                let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
                guard status == .authorized || status == .limited else { throw CancellationError() }
                try await photoCollection.load()
                try checkEpoch(captured)
                let currentStatus = PHPhotoLibrary.authorizationStatus(for: .readWrite)
                guard currentStatus == .authorized || currentStatus == .limited else { throw CancellationError() }
                fullyAuthorized = status == .authorized && currentStatus == .authorized
                hasLibrarySnapshot = true
                try await fetchUnIndexedPhotos()
            }
            remainingIndexingPhotosNum = checkpoint.requiredIDs.filter { !hasUsableEmbedding(for: $0) }.count
            if remainingIndexingPhotosNum == 0 {
                try completeTarget()
                buildIndexCode = .BUILD_FINISHED
                totalUnIndexedPhotosNum = 0
            } else {
                buildIndexCode = .BUILD_INCOMPLETE
            }
            if let activeStore = embeddingStore, let activeRef = activeReference,
               activeStore.needsCompaction() {
                // Compaction is optional: the committed segments are durable.
                do {
                    _ = try activeStore.commit(upserts: savedEmbedding, checkpoint: nil,
                                               generation: activeRef.generation, replacing: true)
                } catch {
                    logger.error("Index compaction failed; retaining committed segments.")
                }
            }
        } catch {
            guard captured == epoch else { return }
            remainingIndexingPhotosNum = assets.filter { !hasUsableEmbedding(for: $0.id) }.count
            indexingErrorMessage = error is CancellationError
                ? "Indexing was interrupted. Saved photos are safe; retry to continue."
                : "The index could not be saved or completed. Retry to keep going."
            buildIndexCode = .BUILD_ERROR
            logger.error("Indexing stopped with \(self.buildingEmbedding.count) unsaved entries: \(error.localizedDescription)")
            // Keep unsaved embeddings in memory so a save retry need not re-encode.
        }
    }

    private func persistBuildingEmbeddings() throws {
        guard !buildingEmbedding.isEmpty else { return }
        if hasLibrarySnapshot {
            buildingEmbedding = buildingEmbedding.filter { allPhotosId[$0.key] != nil }
        }
        if indexingOperations == nil {
            let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
            guard status == .authorized || status == .limited else { throw CancellationError() }
            let current = PHAsset.fetchAssets(withLocalIdentifiers: Array(buildingEmbedding.keys), options: nil)
            var usable = Set<String>()
            current.enumerateObjects { asset, _, _ in
                if self.pendingVersions[asset.localIdentifier] == asset.modificationDate { usable.insert(asset.localIdentifier) }
            }
            buildingEmbedding = buildingEmbedding.filter { usable.contains($0.key) }
        }
        let count = buildingEmbedding.count
        for id in buildingEmbedding.keys { if let date = pendingVersions[id] { checkpoint.versions[id] = date } }
        if let ref = targetReference, let targetStore {
            guard coordinator?.isCurrentRequest(ref) == true else { throw CancellationError() }
            _ = try targetStore.commit(upserts: buildingEmbedding,
                                       checkpoint: try JSONEncoder().encode(checkpoint), generation: ref.generation)
            targetEmbedding.merge(buildingEmbedding) { _, new in new }
        } else if let operations = indexingOperations {
            try operations.save(buildingEmbedding)
            savedEmbedding.merge(buildingEmbedding) { _, new in new }
        } else {
            try updateEmbedding(new_indexed_results: buildingEmbedding)
        }
        savedIndexingPhotosNum += count
        buildingEmbedding.removeAll()
        pendingVersions.removeAll()
    }

    private func completeTarget() throws {
        guard let ref = targetReference, let targetStore, let coordinator else { return }
        guard checkpoint.requiredIDs.allSatisfy({ hasUsableEmbedding(for: $0) }) else { throw EmbeddingStoreError.notReady }
        checkpoint.complete = true
        let revision = try targetStore.commit(checkpoint: try JSONEncoder().encode(checkpoint), generation: ref.generation)
        guard let committed = try targetStore.load(), committed.revision == revision,
              committed.generation == ref.generation,
              let data = committed.checkpoint,
              try JSONDecoder().decode(IndexingCheckpoint.self, from: data).complete else { throw EmbeddingStoreError.notReady }
        imageEncoder = nil
        ImgEncoder.flushBufferPool()
        if indexingOperations == nil {
            guard let url = resourceURL, try indexingSpec.checkpointHash(resourcesAt: url) == ref.checkpointHash else { throw EmbeddingStoreError.notReady }
            do { _ = try TextEncoder(resourcesAt: url, spec: indexingSpec) }
        }
        do {
            try coordinator.activate(ref, evidence: ModelIndexActivationEvidence(reference: ref, committedRevision: revision, complete: true))
        } catch {
            // A rename may have committed even if the following durability check failed.
            if coordinator.state.active == ref {
                activeReference = ref
                try publishActive(ref)
            }
            throw error
        }
        try publishActive(ref)
        targetReference = nil; self.targetStore = nil; targetEmbedding.removeAll()
        activeReference = ref
        recoveryMessage = nil
        publishCoordinatorState()
    }

    private func getDocumentsDirectory() -> URL {
        URL.documentsDirectory
    }


    /**
     Search Part — GPU-accelerated similarity search
     */
    func search(with query: String) async {
        // Capture the generation so a model switch during encoding cannot
        // publish another generation's results into this query.
        let captured = epoch
        let searchingSpec = modelSpec
        let searchingEmbeddings = savedEmbedding
        self.searchString = query
        self.searchErrorMessage = nil
        self.searchResultPhotoAssets = [PhotoAsset]()

        self.searchResultCode = .IS_SEARCHING

        if searchingEmbeddings.isEmpty {
            print("Never indexed.")
            self.searchResultCode = .NEVER_INDEXED
            return
        }

        print("Has indexed data, now begin to search.")

        do {
            if indexingOperations == nil {
                guard let url = resourceURL else { throw PhotoSearchError.encoderNotReady }
                try photoSearchModel.load_text_encoder(resourcesAt: url, spec: searchingSpec)
            }
            defer { photoSearchModel.releaseTextEncoder() }
            let embedding = try photoSearchModel.text_embedding(prompt: query)
            try checkEpoch(captured)
            guard searchingSpec.compatibilityIdentity == modelSpec.compatibilityIdentity else {
                throw CancellationError()
            }
            let ids = try rankedPhotoIDs(query: embedding, embeddings: searchingEmbeddings, spec: searchingSpec)
            try checkEpoch(captured)
            searchResultPhotoAssets = ids.map { PhotoAsset(identifier: $0) }
            searchResultCode = ids.isEmpty ? .NO_RESULT : .HAS_RESULT
        } catch is CancellationError {
            // A stale query after a model switch must not overwrite fresh state.
            guard captured == epoch else { return }
            searchResultCode = .MODEL_PREPARED
        } catch {
            searchErrorMessage = searchFailureMessage(for: error)
            searchResultCode = .SEARCH_ERROR
            logger.error("Search failed: \(error.localizedDescription)")
        }
    }

    func similarPhoto(with photoAsset: PhotoAsset) async {
        let captured = epoch
        let searchingSpec = modelSpec
        let searchingEmbeddings = savedEmbedding
        isFindingSimilarPhotos = true
        similarPhotoAssets.removeAll()
        similarPhotoErrorMessage = nil
        defer { isFindingSimilarPhotos = false }

        do {
            guard let embedding = searchingEmbeddings[photoAsset.id] else {
                throw PhotoSearchError.referencePhotoMissing
            }
            // Validate before converting so alternate scalar/layout inputs cannot bypass the contract.
            _ = try SimilarityVectorValidation.norm(of: embedding, dimension: searchingSpec.embeddingDimension, id: photoAsset.id)
            let query = MLShapedArray<Float32>(converting: embedding)
            try checkEpoch(captured)
            guard searchingSpec.compatibilityIdentity == modelSpec.compatibilityIdentity else {
                throw CancellationError()
            }
            similarPhotoAssets = try rankedPhotoIDs(query: query, embeddings: searchingEmbeddings, spec: searchingSpec).map { PhotoAsset(identifier: $0) }
            try checkEpoch(captured)
        } catch is CancellationError {
            guard captured == epoch else { return }
            similarPhotoErrorMessage = "The search model changed. Try again."
            logger.error("Similar-photo search superseded by a model change.")
        } catch {
            similarPhotoErrorMessage = searchFailureMessage(for: error)
            logger.error("Similar-photo search failed: \(error.localizedDescription)")
        }
    }

    /// Both search surfaces share validation, backend selection, and result publication rules.
    /// Callers pass the snapshot they searched so a concurrent model switch cannot
    /// mix one generation's query with another generation's index.
    private func rankedPhotoIDs(query: MLShapedArray<Float32>, embeddings: [String: MLMultiArray]? = nil, spec: EmbeddingModelSpec? = nil) throws -> [String] {
        let searchingSpec = spec ?? modelSpec
        let searchingEmbeddings = embeddings ?? savedEmbedding
        _ = try SimilarityVectorValidation.norm(of: query, dimension: searchingSpec.embeddingDimension)
        let scores: [String: Float]
        if let gpu = gpuSearch, gpu.count > 0 {
            guard gpu.embeddingDimension == searchingSpec.embeddingDimension else {
                throw SimilaritySearchError.indexUnavailable
            }
            do {
                scores = try gpu.search(queryEmbedding: query)
            } catch SimilaritySearchError.indexUnavailable {
                gpuSearch = nil
                logger.error("GPU index unavailable; using validated CPU search.")
                scores = try photoSearchModel.similarityScores(query: query, embeddings: searchingEmbeddings)
            } catch SimilaritySearchError.resultsUnavailable {
                gpuSearch = nil
                logger.error("GPU returned no scores; using validated CPU search.")
                scores = try photoSearchModel.similarityScores(query: query, embeddings: searchingEmbeddings)
            }
        } else {
            scores = try photoSearchModel.similarityScores(query: query, embeddings: searchingEmbeddings)
        }
        var visibleIDs: Set<String>?
        if indexingOperations == nil {
            let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
            guard status == .authorized || status == .limited else { throw PhotoSearchError.photoAccessUnavailable }
            let visible = PHAsset.fetchAssets(withLocalIdentifiers: Array(scores.keys), options: nil)
            var ids = Set<String>()
            visible.enumerateObjects { asset, _, _ in ids.insert(asset.localIdentifier) }
            visibleIDs = ids
        }
        return scores.filter { (visibleIDs?.contains($0.key) ?? true) && (!hasLibrarySnapshot || allPhotosId[$0.key] != nil) }.sorted {
            $0.value == $1.value ? $0.key < $1.key : $0.value > $1.value
        }.prefix(max(0, TOPK_SIM)).map(\.key)
    }

    private func searchFailureMessage(for error: Error) -> String {
        if let error = error as? PhotoSearchError { return error.localizedDescription }
        if error is SimilaritySearchError {
            return "The search data could not be used. Update the index and try again."
        }
        return "The search model could not complete this search. Try again."
    }


}


public func clearCache(){
    URLCache.shared.removeAllCachedResponses()

    do {
        let tmpDirURL = FileManager.default.temporaryDirectory
        let tmpDirectory = try FileManager.default.contentsOfDirectory(atPath: tmpDirURL.path)
        try tmpDirectory.forEach { file in
            let fileUrl = tmpDirURL.appendingPathComponent(file)
            print("File to be removed: \(fileUrl)")
            try FileManager.default.removeItem(atPath: fileUrl.path)
        }
    } catch {
        //catch the error somehow
    }
}


fileprivate let logger = Logger(subsystem: "com.mazzystar.Queryable", category: "PhotoSearcher")
