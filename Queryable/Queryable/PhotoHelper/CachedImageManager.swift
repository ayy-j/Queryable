/*
See the License.txt file for this sample’s licensing information.
*/

import UIKit
import Photos
import SwiftUI
import os.log

actor CachedImageManager {
    
    private let imageManager = PHCachingImageManager()
    
    private var imageContentMode = PHImageContentMode.aspectFit
    
    enum CachedImageManagerError: LocalizedError {
        case error(Error)
        case cancelled
        case failed
    }
    
    private var cachedAssetIdentifiers = [String : Bool]()
    
    lazy var requestOptions: PHImageRequestOptions = {
        let options = PHImageRequestOptions()
        options.isNetworkAccessAllowed = false
        options.deliveryMode = .opportunistic
        return options
    }()
    
    init() {
        imageManager.allowsCachingHighQualityImages = false
    }
    
    var cachedImageCount: Int {
        cachedAssetIdentifiers.keys.count
    }
    
    func startCaching(for assets: [PhotoAsset], targetSize: CGSize) {
        let phAssets = assets.compactMap { $0.phAsset }
        phAssets.forEach {
            cachedAssetIdentifiers[$0.localIdentifier] = true
        }
        imageManager.startCachingImages(for: phAssets, targetSize: targetSize, contentMode: imageContentMode, options: requestOptions)
    }

    func stopCaching(for assets: [PhotoAsset], targetSize: CGSize) {
        let phAssets = assets.compactMap { $0.phAsset }
        phAssets.forEach {
            cachedAssetIdentifiers.removeValue(forKey: $0.localIdentifier)
        }
        imageManager.stopCachingImages(for: phAssets, targetSize: targetSize, contentMode: imageContentMode, options: requestOptions)
    }
    
    func stopCaching() {
        imageManager.stopCachingImagesForAllAssets()
    }
    
    @discardableResult
    func requestImage(for asset: PhotoAsset, targetSize: CGSize, completion: @escaping ((image: UIImage?, isLowerQuality: Bool)?) -> Void) -> PHImageRequestID? {
        guard let phAsset = asset.phAsset else {
            completion(nil)
            return nil
        }
        
        let requestID = imageManager.requestImage(for: phAsset, targetSize: targetSize, contentMode: imageContentMode, options: requestOptions) { image, info in
            if let error = info?[PHImageErrorKey] as? Error {
                logger.error("CachedImageManager requestImage error: \(error.localizedDescription)")
                completion(nil)
            } else if let cancelled = (info?[PHImageCancelledKey] as? NSNumber)?.boolValue, cancelled {
                logger.debug("CachedImageManager request canceled")
                completion(nil)
            } else if let image = image {
                let isLowerQualityImage = (info?[PHImageResultIsDegradedKey] as? NSNumber)?.boolValue ?? false
                let result = (image: image, isLowerQuality: isLowerQualityImage)
                completion(result)
            } else {
                completion(nil)
            }
        }
        return requestID
    }
    
    func cancelImageRequest(for requestID: PHImageRequestID) {
        imageManager.cancelImageRequest(requestID)
    }

    /// Index only the final image. Display requests remain opportunistic.
    func imageForIndexing(for asset: PhotoAsset, targetSize: CGSize) async throws -> UIImage {
        guard let phAsset = asset.phAsset else { throw CachedImageManagerError.failed }
        let options = PHImageRequestOptions()
        options.isNetworkAccessAllowed = false
        options.deliveryMode = .highQualityFormat
        let manager = imageManager
        let contentMode = imageContentMode
        return try await IndexingImageRequest.image { completion in
            manager.requestImage(for: phAsset, targetSize: targetSize, contentMode: contentMode,
                                 options: options, resultHandler: completion)
        } cancel: { requestID in
            manager.cancelImageRequest(requestID)
        }
    }
}

/// Photos can call back before requestImage returns, more than once, or after
/// cancellation. Protect the continuation and request ID together in every case.
final class IndexingImageRequest: @unchecked Sendable {
    typealias Completion = (UIImage?, [AnyHashable: Any]?) -> Void
    private let lock = NSLock()
    private var continuation: CheckedContinuation<UIImage, Error>?
    private var result: Result<UIImage, Error>?
    private var requestID: PHImageRequestID?
    private var wasCancelled = false

    static func image(start: (@escaping Completion) -> PHImageRequestID,
                      cancel: @escaping (PHImageRequestID) -> Void) async throws -> UIImage {
        let request = IndexingImageRequest()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                guard request.install(continuation) else { return }
                let id = start { image, info in
                    if (info?[PHImageCancelledKey] as? NSNumber)?.boolValue == true {
                        request.finish(.failure(CancellationError()))
                    } else if let error = info?[PHImageErrorKey] as? Error {
                        request.finish(.failure(error))
                    } else if image == nil && (info?[PHImageResultIsInCloudKey] as? NSNumber)?.boolValue == true {
                        request.finish(.failure(CachedImageManager.CachedImageManagerError.failed))
                    } else if (info?[PHImageResultIsDegradedKey] as? NSNumber)?.boolValue != true {
                        request.finish(image.map { .success($0) }
                            ?? .failure(CachedImageManager.CachedImageManagerError.failed))
                    }
                }
                request.setRequestID(id, cancel: cancel)
            }
        } onCancel: {
            request.cancel(using: cancel)
        }
    }

    private func install(_ continuation: CheckedContinuation<UIImage, Error>) -> Bool {
        lock.lock()
        if let result {
            lock.unlock()
            continuation.resume(with: result)
            return false
        }
        self.continuation = continuation
        lock.unlock()
        return true
    }

    private func finish(_ result: Result<UIImage, Error>) {
        lock.lock()
        guard self.result == nil else { lock.unlock(); return }
        self.result = result
        let continuation = self.continuation
        self.continuation = nil
        lock.unlock()
        continuation?.resume(with: result)
    }

    private func setRequestID(_ id: PHImageRequestID, cancel: (PHImageRequestID) -> Void) {
        lock.lock()
        requestID = id
        let shouldCancel = wasCancelled
        lock.unlock()
        if shouldCancel { cancel(id) }
    }

    private func cancel(using cancel: (PHImageRequestID) -> Void) {
        lock.lock()
        wasCancelled = true
        let id = requestID
        let continuation = result == nil ? self.continuation : nil
        if result == nil {
            result = .failure(CancellationError())
            self.continuation = nil
        }
        lock.unlock()
        continuation?.resume(throwing: CancellationError())
        if let id { cancel(id) }
    }
}

fileprivate let logger = Logger(subsystem: "com.apple.swiftplaygroundscontent.capturingphotos", category: "CachedImageManager")
