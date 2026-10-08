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
}


@MainActor
class PhotoSearcher: ObservableObject {
    let defaults = UserDefaults.standard
    let photoCollection = PhotoCollection(smartAlbum: .smartAlbumUserLibrary)
    var photoSearchModel = PhotoSearcherModel()
    private let modelSpec: EmbeddingModelSpec
    let KEY_HAS_ACCESS_TO_PHOTOS = "KEY_HAS_ACCESS_TO_PHOTOS"

    // -3: default, -2: Is searching now, -1: Never indexed. 0: No result. 1: Has result.
    @Published var searchResultCode: SEARCH_RESULT_CODE = .DEFAULT
    @Published var buildIndexCode: BUILD_INDEX_CODE = .DEFAULT
    /// Actionable message when model files are missing/unreadable; nil otherwise.
    @Published var modelErrorMessage: String? = nil
    @Published var totalUnIndexedPhotosNum: Int = -1
    @Published var curIndexingNums: Int = -1
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

    /// GPU-accelerated similarity search (Float16 MPSGraph matmul)
    private var gpuSearch: GPUSimilaritySearch?
    /// Efficient binary embedding storage with incremental saves
    private var embeddingStore: EmbeddingStore?

    @Published var TOPK_SIM: Int {
        didSet {
            UserDefaults.standard.set(TOPK_SIM, forKey: "TOPK_SIM")
        }
    }

    init(modelSpec: EmbeddingModelSpec = .mobileCLIPS2) {
        self.modelSpec = modelSpec
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
            if let artifactError = error as? ModelArtifactError {
                self.modelErrorMessage = artifactError.localizedDescription
            }
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
            print("Load photos embedding failure.")
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
            self.buildIndexCode = .IS_BUILDING_INDEX
        } catch let error {
            logger.error("Failed to load model: \(error.localizedDescription)")
            self.modelErrorMessage = (error as? ModelArtifactError)?.localizedDescription ?? error.localizedDescription
            self.buildIndexCode = .MODEL_ERROR
        }
    }

    func defaultEmbedding() -> MLShapedArray<Float32> {
        // Zero embedding: 0 cosine similarity with any query, won't appear in results
        let zeros = [Float32](repeating: 0, count: modelSpec.embeddingDimension)
        return MLShapedArray<Float32>(scalars: zeros, shape: [1, modelSpec.embeddingDimension])
    }

    func batchBuildIndex(assets: [PhotoAsset]) async throws {
        let targetSize = CGSize(width: modelSpec.imageSize, height: modelSpec.imageSize)
        await photoCollection.cache.startCaching(for: assets, targetSize: targetSize)

        let BATCH_SIZE = 32

        for batchStart in stride(from: 0, to: assets.count, by: BATCH_SIZE) {
            let batchEnd = min(batchStart + BATCH_SIZE, assets.count)
            let batchAssets = Array(assets[batchStart..<batchEnd])

            // Phase 1: Parallel image fetching
            var images = [(PhotoAsset, UIImage)]()
            await withTaskGroup(of: (PhotoAsset, UIImage?).self) { group in
                for asset in batchAssets {
                    group.addTask {
                        var fetchedImage: UIImage? = nil
                        _ = await self.photoCollection.cache.requestImage(for: asset, targetSize: targetSize) { result in
                            fetchedImage = result?.image
                        }
                        return (asset, fetchedImage)
                    }
                }
                for await (asset, image) in group {
                    if let image = image {
                        images.append((asset, image))
                    } else {
                        self.buildingEmbedding[asset.id] = MLMultiArray(self.defaultEmbedding())
                    }
                }
            }

            // Phase 2: Batch CoreML prediction
            var batchFailed = false
            if !images.isEmpty {
                autoreleasepool {
                    do {
                        let uiImages = images.map { $0.1 }
                        let embeddings = try self.imageEncoder!.encodeBatch(images: uiImages)

                        for (i, (asset, _)) in images.enumerated() {
                            self.buildingEmbedding[asset.id] = ImgEncoder.detachFromIOSurface(embeddings[i])
                        }
                    } catch {
                        print("[BatchIndex] Batch encoding failed, falling back to single: \(error)")
                        batchFailed = true
                    }
                }
                // Fallback outside autoreleasepool (encode is async)
                if batchFailed {
                    for (asset, image) in images {
                        do {
                            let emb = try await self.imageEncoder!.encode(image: image)
                            self.buildingEmbedding[asset.id] = ImgEncoder.detachFromIOSurface(emb)
                        } catch {
                            self.buildingEmbedding[asset.id] = MLMultiArray(self.defaultEmbedding())
                        }
                    }
                }
            }

            // Capture last image for UI before clearing
            let lastImage = images.last?.1

            // Explicitly release UIImages before flush
            images.removeAll()

            // Release pooled CVPixelBuffers so the OS can reclaim their IOSurfaces
            ImgEncoder.flushBufferPool()

            // Update UI with last image from batch
            if let lastImage {
                self.curIndexingPhoto = lastImage
            }
        }

        await photoCollection.cache.stopCaching(for: assets, targetSize: targetSize)
    }

    func fetchUnIndexedPhotos() async throws {
        let startingTime = Date()
        self.unIndexedPhotos = [PhotoAsset]()
        var photoIdSet = Set<String>(minimumCapacity: photoCollection.photoAssets.count)

        for idx in 0..<self.photoCollection.photoAssets.count {
            let asset = self.photoCollection.photoAssets[idx]
            self.allPhotosId[asset.id] = 1
            photoIdSet.insert(asset.id)
            if self.savedEmbedding[asset.id] == nil {
                self.unIndexedPhotos.append(asset)
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

        // Update GPU index with new embeddings
        try gpuSearch?.addEmbeddings(new_indexed_results)
        for (key, value) in new_indexed_results {
            self.savedEmbedding[key] = value
        }
        print("After update, embedding count=\(self.savedEmbedding.count)")
        print("Embedding saved (incremental)")
    }

    /**
     Build index
     */
    func buildIndex() async {
        self.buildingEmbedding = [String: MLMultiArray]()

        // set network authorization
        await self.photoCollection.cache.requestOptions.isNetworkAccessAllowed = false

        do {
             for idx in stride(from: 0, to: self.unIndexedPhotos.count, by: self.BUILD_INDEX_FRAGMENT_LENGTH) {
                 let endPoint = min(idx+self.BUILD_INDEX_FRAGMENT_LENGTH, self.unIndexedPhotos.count)
                 let assets = Array(self.unIndexedPhotos[idx..<endPoint])

                 try await self.batchBuildIndex(assets: assets)
                 self.curIndexingNums += assets.count
                 self.curShowingPhoto = self.curIndexingPhoto

                 // Yield every ~500 images so the Neural Engine runtime
                 // can reclaim internal IOSurface allocations.
                 if self.curIndexingNums % 500 < self.BUILD_INDEX_FRAGMENT_LENGTH {
                     try? await Task.sleep(nanoseconds: 100_000_000) // 100ms
                 }

                 if curIndexingNums > 0 && self.buildingEmbedding.count >= SAVE_EMBEDDING_EVERY {
                     print("Save index for \(curIndexingNums) images.")
                     try self.updateEmbedding(new_indexed_results: self.buildingEmbedding)
                     self.buildingEmbedding = [String: MLMultiArray]()
                 }
            }

            // save embedding
            try self.updateEmbedding(new_indexed_results: self.buildingEmbedding)

        } catch {
            print("Build index error: \(error). Saving \(self.buildingEmbedding.count) embeddings indexed so far.")
            do {
               try self.updateEmbedding(new_indexed_results: self.buildingEmbedding)
            } catch {
               logger.error("Failed to save indexed embeddings: \(error.localizedDescription)")
            }
        }

        // Compact storage if needed
        if embeddingStore?.needsCompaction() == true {
            embeddingStore?.compact(self.savedEmbedding)
        }

        // delete large memory usage
        self.buildingEmbedding = [String: MLMultiArray]()
        self.imageEncoder = nil
        self.buildIndexCode = .BUILD_FINISHED
        self.totalUnIndexedPhotosNum = 0

        clearCache()
    }

    private func getDocumentsDirectory() -> URL {
        URL.documentsDirectory
    }


    /**
     Search Part — GPU-accelerated similarity search
     */
    func search(with query: String) async {
        self.searchString = query
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

        print("Searching query = \(query)")
        let _text_emb = self.photoSearchModel.text_embedding(prompt: query)

        let startingTime = Date()
        let FINAL_TOP_K = min(self.TOPK_SIM, self.savedEmbedding.count)

        // GPU search path
        if let gpu = self.gpuSearch, gpu.count > 0 {
            let simDict: [String: Float]
            do {
                simDict = try gpu.search(queryEmbedding: _text_emb)
            } catch {
                logger.error("Search rejected an incompatible embedding: \(error.localizedDescription)")
                self.searchResultCode = .NO_RESULT
                return
            }
            let topK = simDict.sorted { $0.value > $1.value }.prefix(FINAL_TOP_K)
            print("\(startingTime.timeIntervalSinceNow * -1) seconds used for GPU search \(self.savedEmbedding.count) embeddings.")

            for photo in topK {
                let _asset = PhotoAsset(identifier: photo.key)
                self.searchResultPhotoAssets.append(_asset)
            }
        } else {
            // CPU fallback
            print("GPU search unavailable, using CPU fallback")
            var simDict = [String: Float]()
            for (key, cur_img_emb) in self.savedEmbedding {
                let img_emb = MLShapedArray<Float32>(converting: cur_img_emb)
                simDict[key] = await self.photoSearchModel.cosine_similarity(A: _text_emb, B: img_emb)
            }
            let topK = simDict.sorted { $0.value > $1.value }.prefix(FINAL_TOP_K)
            for photo in topK {
                let _asset = PhotoAsset(identifier: photo.key)
                self.searchResultPhotoAssets.append(_asset)
            }
            print("\(startingTime.timeIntervalSinceNow * -1) seconds used for CPU search \(self.savedEmbedding.count) embeddings.")
        }

        self.searchResultCode = .HAS_RESULT
    }


    /**
     Similarity ranking — GPU-accelerated
     */
    func similarPhoto(with photoAsset: PhotoAsset) async {
        self.isFindingSimilarPhotos = true
        self.similarPhotoAssets = [PhotoAsset]()

        guard let embML = self.savedEmbedding[photoAsset.id] else {
            self.isFindingSimilarPhotos = false
            return
        }

        let _img_emb = MLShapedArray<Float32>(converting: embML)
        let startingTime = Date()
        let FINAL_TOP_K = min(self.TOPK_SIM, self.savedEmbedding.count)

        if let gpu = self.gpuSearch, gpu.count > 0 {
            let simDict: [String: Float]
            do {
                simDict = try gpu.search(queryEmbedding: _img_emb)
            } catch {
                logger.error("Similar-photo search rejected an incompatible embedding: \(error.localizedDescription)")
                self.isFindingSimilarPhotos = false
                return
            }
            let topK = simDict.sorted { $0.value > $1.value }.prefix(FINAL_TOP_K)
            print("\(startingTime.timeIntervalSinceNow * -1) seconds used for GPU similar search.")

            for photo in topK {
                let _asset = PhotoAsset(identifier: photo.key)
                self.similarPhotoAssets.append(_asset)
            }
        } else {
            // CPU fallback
            var simDict = [String: Float]()
            for (key, cur_img_emb) in self.savedEmbedding {
                let img_emb = MLShapedArray<Float32>(converting: cur_img_emb)
                simDict[key] = await self.photoSearchModel.cosine_similarity(A: _img_emb, B: img_emb)
            }
            let topK = simDict.sorted { $0.value > $1.value }.prefix(FINAL_TOP_K)
            for photo in topK {
                let _asset = PhotoAsset(identifier: photo.key)
                self.similarPhotoAssets.append(_asset)
            }
            print("\(startingTime.timeIntervalSinceNow * -1) seconds used for CPU similar search.")
        }

        self.isFindingSimilarPhotos = false
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
