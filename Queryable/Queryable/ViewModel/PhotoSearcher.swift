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
    private let modelSpec: EmbeddingModelSpec
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
         indexingOperations: PhotoIndexingOperations? = nil) {
        self.modelSpec = modelSpec
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
        print("Clear cache..")
        clearCache()
        print("Cache cleared.")

        self.searchResultCode = .DEFAULT
        self.searchErrorMessage = nil
        self.modelErrorMessage = nil
        guard let path = Bundle.main.path(forResource: "CoreMLModels", ofType: nil, inDirectory: nil) else {
            logger.error("Failed to find the CoreML models.")
            self.modelErrorMessage = ModelArtifactError.missingArtifact("CoreMLModels").localizedDescription
            self.searchResultCode = .NEVER_INDEXED
            return
        }
        let resourceURL = URL(fileURLWithPath: path)
        let modelSpec = self.modelSpec

        let store: EmbeddingStore
        do {
            let checkpointHash = try await Task.detached(priority: .utility) {
                try modelSpec.checkpointHash(resourcesAt: resourceURL)
            }.value
            try self.photoSearchModel.load_text_encoder(resourcesAt: resourceURL, spec: modelSpec)
            store = EmbeddingStore(spec: modelSpec, checkpointHash: checkpointHash)
            self.embeddingStore = store
        } catch {
            logger.error("Failed to load model contract: \(error.localizedDescription)")
            self.modelErrorMessage = (error as? ModelArtifactError)?.localizedDescription
                ?? (error as? EmbeddingModelSpecError)?.localizedDescription
                ?? error.localizedDescription
            self.searchResultCode = .NEVER_INDEXED
            return
        }
        print("Text encoder loaded.")

        // Load embeddings from binary store (background I/O)
        let loaded = await Task.detached {
            store.loadAll()
        }.value

        if let loaded, !loaded.isEmpty {
            self.savedEmbedding = loaded
            print("Photos embedding loaded. total \(self.savedEmbedding.count)")

            // Build GPU search index
            do {
                try gpuSearch?.buildIndex(from: self.savedEmbedding)
            } catch {
                logger.error("Rejected embedding index: \(error.localizedDescription)")
                self.savedEmbedding.removeAll()
                self.searchResultCode = .NEVER_INDEXED
            }
        } else {
            self.searchResultCode = .NEVER_INDEXED
            print("No compatible nonempty saved photo index loaded. Indexing is required.")
        }

        // set network authorization
        await self.photoCollection.cache.requestOptions.isNetworkAccessAllowed = false

        // Get the current authorization state.
        let status = PHPhotoLibrary.authorizationStatus()
        if (status == .authorized) {
            defaults.set(true, forKey: self.KEY_HAS_ACCESS_TO_PHOTOS)
            print("KEY_HAS_ACCESS_TO_PHOTOS has been updated to true.")
        }

        if !self.savedEmbedding.isEmpty {
            self.searchResultCode = .MODEL_PREPARED
        }
    }

    func fetchPhotos() async {
        self.buildIndexCode = .LOADING_PHOTOS

        // set network authorization
        await self.photoCollection.cache.requestOptions.isNetworkAccessAllowed = false

        let authorized = await PhotoLibrary.checkAuthorization()
        guard authorized else {
            logger.error("Photo library access was not authorized.")
            return
        }

        do {
            try await self.photoCollection.load()
            print("Total \(photoCollection.photoAssets.count) photos loaded.")
            defaults.set(true, forKey: self.KEY_HAS_ACCESS_TO_PHOTOS)
            self.totalPhotosNum = photoCollection.photoAssets.count
            try await self.fetchUnIndexedPhotos()
        } catch let error {
            logger.error("Failed to load photo collection: \(error.localizedDescription)")
        }

        if self.totalPhotosNum > 0 {
            self.buildIndexCode = .PHOTOS_LOADED
        }
    }

    func loadImageIncoder() async {
        self.buildIndexCode = .LOADING_MODEL
        self.modelErrorMessage = nil
        guard let path = Bundle.main.path(forResource: "CoreMLModels", ofType: nil, inDirectory: nil) else {
            logger.error("Failed to find the CoreML models.")
            self.modelErrorMessage = ModelArtifactError.missingArtifact("CoreMLModels").localizedDescription
            self.buildIndexCode = .MODEL_ERROR
            return
        }
        let resourceURL = URL(fileURLWithPath: path)

        do {
            let startingTime = Date()
            let imgEncoder = try ImgEncoder(resourcesAt: resourceURL, spec: self.modelSpec)
            print("\(startingTime.timeIntervalSinceNow * -1) seconds used for loading img encoder")
            self.imageEncoder = imgEncoder

            // Ensure embeddingStore is ready before buildIndex runs.
            // prepareModelForSearch() may not have been called (e.g. user went straight
            // to the Build Index tab) or may have failed, leaving embeddingStore nil.
            if self.embeddingStore == nil {
                let modelSpec = self.modelSpec
                let checkpointHash = try await Task.detached(priority: .utility) {
                    try modelSpec.checkpointHash(resourcesAt: resourceURL)
                }.value
                self.embeddingStore = EmbeddingStore(spec: modelSpec, checkpointHash: checkpointHash)
            }

            self.buildIndexCode = .IS_BUILDING_INDEX
        } catch let error {
            logger.error("Failed to load model: \(error.localizedDescription)")
            self.modelErrorMessage = (error as? ModelArtifactError)?.localizedDescription ?? error.localizedDescription
            self.buildIndexCode = .MODEL_ERROR
        }
    }

    private func validatedEmbedding(_ embedding: MLShapedArray<Float32>) throws -> MLMultiArray {
        let values = embedding.scalars
        guard values.count == modelSpec.embeddingDimension,
              embedding.shape == [1, modelSpec.embeddingDimension] || embedding.shape == [modelSpec.embeddingDimension],
              values.allSatisfy({ $0.isFinite }), values.contains(where: { $0 != 0 }) else {
            throw PhotoIndexingError.invalidEmbedding
        }
        // Copy scalar values rather than retaining the prediction's IOSurface.
        return MLMultiArray(MLShapedArray<Float32>(scalars: values, shape: [1, modelSpec.embeddingDimension]))
    }

    private func hasUsableEmbedding(for id: String) -> Bool {
        guard let value = savedEmbedding[id], value.dataType == .float32,
              value.count == modelSpec.embeddingDimension else { return false }
        let values = MLShapedArray<Float32>(converting: value).scalars
        return values.allSatisfy { $0.isFinite } && values.contains { $0 != 0 }
    }

    func batchBuildIndex(assets: [PhotoAsset]) async throws {
        let targetSize = CGSize(width: modelSpec.imageSize, height: modelSpec.imageSize)
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
                try Task.checkCancellation()
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
                        if let image { images.append((asset, image)) }
                        else { self.failedIndexingPhotosNum += 1 }
                    }
                }
                try Task.checkCancellation()
                if !images.isEmpty {
                    do {
                        let embeddings = try autoreleasepool {
                            try operations.encodeBatch(images.map { $0.1 })
                        }
                        guard embeddings.count == images.count else { throw PhotoIndexingError.invalidEmbedding }
                        for (index, (asset, _)) in images.enumerated() {
                            do { buildingEmbedding[asset.id] = try validatedEmbedding(embeddings[index]) }
                            catch { failedIndexingPhotosNum += 1 }
                        }
                    } catch {
                        // A batch failure must not discard images that work individually.
                        for (asset, image) in images {
                            try Task.checkCancellation()
                            do {
                                let embedding = try await operations.encodeImage(image)
                                buildingEmbedding[asset.id] = try validatedEmbedding(embedding)
                            } catch is CancellationError { throw CancellationError() }
                            catch { failedIndexingPhotosNum += 1 }
                        }
                    }
                    curIndexingPhoto = images.last!.1
                }
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
            if !hasUsableEmbedding(for: asset.id) {
                self.unIndexedPhotos.append(asset)
                if savedEmbedding[asset.id] != nil { blankEmbeddingPhotosNum += 1 }
            }
        }

        // Orphan detection: embeddings whose photos have been deleted
        var orphanedIds = [String]()
        for embeddingId in self.savedEmbedding.keys {
            if !photoIdSet.contains(embeddingId) {
                orphanedIds.append(embeddingId)
            }
        }
        if !orphanedIds.isEmpty {
            let orphanRatio = Double(orphanedIds.count) / Double(self.savedEmbedding.count)
            if orphanRatio > 0.2 {
                print("[Startup] orphanCleanup: skipping — \(orphanedIds.count)/\(self.savedEmbedding.count) (\(String(format: "%.0f", orphanRatio * 100))%) exceeds 20% threshold")
            } else {
                print("[Startup] orphanCleanup: removing \(orphanedIds.count) orphaned embeddings")
                for id in orphanedIds {
                    self.savedEmbedding.removeValue(forKey: id)
                }
                embeddingStore?.markDeleted(orphanedIds)
                gpuSearch?.removeEmbeddings(Set(orphanedIds))
            }
        }

        self.totalUnIndexedPhotosNum = self.unIndexedPhotos.count
        print("\(startingTime.timeIntervalSinceNow * -1) seconds used for filter unindex photos")
    }

    func deleteEmbeddingByAsset(asset: PhotoAsset) async {
        if self.savedEmbedding[asset.id] != nil {
            self.savedEmbedding.removeValue(forKey: asset.id)
            embeddingStore?.markDeleted([asset.id])
            gpuSearch?.removeEmbeddings(Set([asset.id]))
            print("\(asset.id) deleted.")
        }
    }

    func updateEmbedding(new_indexed_results: [String: MLMultiArray]) throws {
        guard let embeddingStore else { throw EmbeddingStoreError.notReady }
        print("Before update, embedding count=\(self.savedEmbedding.count)")

        // Incremental save: only write new embeddings to journal
        guard embeddingStore.appendNew(new_indexed_results) else {
            throw EmbeddingStoreError.writeFailed
        }

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
        defer {
            isBuildingIndex = false
            imageEncoder = nil
            ImgEncoder.flushBufferPool()
        }
        buildIndexCode = .IS_BUILDING_INDEX
        indexingErrorMessage = nil
        curIndexingNums = 0
        savedIndexingPhotosNum = 0
        failedIndexingPhotosNum = 0
        totalUnIndexedPhotosNum = assets.count
        remainingIndexingPhotosNum = assets.count

        do {
            // A save retry commits retained work before attempting more photos.
            let retainedCount = buildingEmbedding.count
            try persistBuildingEmbeddings()
            curIndexingNums += retainedCount
            let pendingAssets = assets.filter { !hasUsableEmbedding(for: $0.id) }
            for index in stride(from: 0, to: pendingAssets.count, by: BUILD_INDEX_FRAGMENT_LENGTH) {
                let end = min(index + BUILD_INDEX_FRAGMENT_LENGTH, pendingAssets.count)
                try await batchBuildIndex(assets: Array(pendingAssets[index..<end]))
                if buildingEmbedding.count >= SAVE_EMBEDDING_EVERY {
                    try persistBuildingEmbeddings()
                }
                // Give the Neural Engine runtime a chance to reclaim allocations.
                if curIndexingNums % 500 < BUILD_INDEX_FRAGMENT_LENGTH {
                    try await Task.sleep(nanoseconds: 100_000_000)
                }
            }
            try Task.checkCancellation()
            try persistBuildingEmbeddings()
            remainingIndexingPhotosNum = assets.filter { !hasUsableEmbedding(for: $0.id) }.count
            if remainingIndexingPhotosNum == 0 {
                buildIndexCode = .BUILD_FINISHED
                totalUnIndexedPhotosNum = 0
            } else {
                buildIndexCode = .BUILD_INCOMPLETE
            }
            if embeddingStore?.needsCompaction() == true {
                // Compaction is optional: the successful journal save is durable.
                if embeddingStore?.compact(savedEmbedding) == false {
                    logger.error("Index compaction failed; retaining the saved journal.")
                }
            }
        } catch {
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
        let count = buildingEmbedding.count
        if let operations = indexingOperations {
            try operations.save(buildingEmbedding)
            savedEmbedding.merge(buildingEmbedding) { _, new in new }
        } else {
            try updateEmbedding(new_indexed_results: buildingEmbedding)
        }
        savedIndexingPhotosNum += count
        buildingEmbedding.removeAll()
    }

    private func getDocumentsDirectory() -> URL {
        URL.documentsDirectory
    }


    /**
     Search Part — GPU-accelerated similarity search
     */
    func search(with query: String) async {
        self.searchString = query
        self.searchErrorMessage = nil
        self.searchResultPhotoAssets = [PhotoAsset]()

        self.searchResultCode = .IS_SEARCHING

        if self.savedEmbedding.isEmpty {
            print("Never indexed.")
            self.searchResultCode = .NEVER_INDEXED
            return
        }

        print("Has indexed data, now begin to search.")

        // Filter deleted photos
        if !self.allPhotosId.isEmpty {
            let startingTime = Date()
            var deletedKeys = [String]()
            for key in self.savedEmbedding.keys {
                if self.allPhotosId[key] == nil {
                    deletedKeys.append(key)
                }
            }

            if !deletedKeys.isEmpty {
                for key in deletedKeys {
                    self.savedEmbedding.removeValue(forKey: key)
                }
                embeddingStore?.markDeleted(deletedKeys)
                gpuSearch?.removeEmbeddings(Set(deletedKeys))
                print("\(deletedKeys.count) keys in savedEmbedding has been deleted.")
            }
            print("\(startingTime.timeIntervalSinceNow * -1) seconds used for cleanup.")
        }

        do {
            let embedding = try photoSearchModel.text_embedding(prompt: query)
            let ids = try rankedPhotoIDs(query: embedding)
            searchResultPhotoAssets = ids.map { PhotoAsset(identifier: $0) }
            searchResultCode = ids.isEmpty ? .NO_RESULT : .HAS_RESULT
        } catch {
            searchErrorMessage = searchFailureMessage(for: error)
            searchResultCode = .SEARCH_ERROR
            logger.error("Search failed: \(error.localizedDescription)")
        }
    }

    func similarPhoto(with photoAsset: PhotoAsset) async {
        isFindingSimilarPhotos = true
        similarPhotoAssets.removeAll()
        similarPhotoErrorMessage = nil
        defer { isFindingSimilarPhotos = false }

        do {
            guard let embedding = savedEmbedding[photoAsset.id] else {
                throw PhotoSearchError.referencePhotoMissing
            }
            // Validate before converting so alternate scalar/layout inputs cannot bypass the contract.
            _ = try SimilarityVectorValidation.norm(of: embedding, dimension: modelSpec.embeddingDimension, id: photoAsset.id)
            let query = MLShapedArray<Float32>(converting: embedding)
            similarPhotoAssets = try rankedPhotoIDs(query: query).map { PhotoAsset(identifier: $0) }
        } catch {
            similarPhotoErrorMessage = searchFailureMessage(for: error)
            logger.error("Similar-photo search failed: \(error.localizedDescription)")
        }
    }

    /// Both search surfaces share validation, backend selection, and result publication rules.
    private func rankedPhotoIDs(query: MLShapedArray<Float32>) throws -> [String] {
        _ = try SimilarityVectorValidation.norm(of: query, dimension: modelSpec.embeddingDimension)
        let scores: [String: Float]
        if let gpu = gpuSearch, gpu.count > 0 {
            do {
                scores = try gpu.search(queryEmbedding: query)
            } catch SimilaritySearchError.indexUnavailable {
                gpuSearch = nil
                logger.error("GPU index unavailable; using validated CPU search.")
                scores = try photoSearchModel.similarityScores(query: query, embeddings: savedEmbedding)
            } catch SimilaritySearchError.resultsUnavailable {
                gpuSearch = nil
                logger.error("GPU returned no scores; using validated CPU search.")
                scores = try photoSearchModel.similarityScores(query: query, embeddings: savedEmbedding)
            }
        } else {
            scores = try photoSearchModel.similarityScores(query: query, embeddings: savedEmbedding)
        }
        return scores.sorted {
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
