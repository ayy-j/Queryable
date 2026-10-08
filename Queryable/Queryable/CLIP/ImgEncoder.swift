//
//  ImgEncoder.swift
//  Queryable
//
//  Created by Ke Fang on 2022/12/08.
//

import Foundation
import CoreML
import CoreImage
import UIKit

public struct ImgEncoder {
    var model: MLModel
    let spec: EmbeddingModelSpec

    /// Shared CIContext for GPU-accelerated image processing
    private static let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    private struct BufferPoolKey: Hashable {
        let width: Int
        let height: Int
        let pixelFormat: OSType
    }

    private static let bufferPoolLock = NSLock()
    private static var bufferPools = [BufferPoolKey: CVPixelBufferPool]()

    /// Flush idle buffers from every pool so the OS can reclaim their IOSurfaces.
    static func flushBufferPool() {
        bufferPoolLock.lock()
        let pools = Array(bufferPools.values)
        bufferPoolLock.unlock()

        for pool in pools {
            CVPixelBufferPoolFlush(pool, CVPixelBufferPoolFlushFlags(rawValue: 0))
        }
    }

    /// Pool of recycled IOSurface-backed buffers for a given input geometry.
    /// Pools are keyed by size and pixel format so a model with a different
    /// input resolution (e.g. 384 px SigLIP) still recycles its buffers.
    static func pixelBufferPool(size: CGSize, pixelFormat: OSType) -> CVPixelBufferPool? {
        let key = BufferPoolKey(
            width: Int(size.width),
            height: Int(size.height),
            pixelFormat: pixelFormat
        )
        bufferPoolLock.lock()
        defer { bufferPoolLock.unlock() }

        if let pool = bufferPools[key] {
            return pool
        }

        let poolAttrs: [String: Any] = [
            kCVPixelBufferPoolMinimumBufferCountKey as String: 4
        ]
        let bufferAttrs: [String: Any] = [
            kCVPixelBufferWidthKey as String: key.width,
            kCVPixelBufferHeightKey as String: key.height,
            kCVPixelBufferPixelFormatTypeKey as String: key.pixelFormat,
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        var pool: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(
            kCFAllocatorDefault,
            poolAttrs as CFDictionary,
            bufferAttrs as CFDictionary,
            &pool
        ) == kCVReturnSuccess, let pool else {
            return nil
        }

        bufferPools[key] = pool
        return pool
    }

    /// Deep-copy an MLShapedArray's scalar data into a fresh, heap-backed MLMultiArray.
    /// CoreML prediction outputs are backed by IOSurface memory; MLShapedArray(converting:)
    /// and MLMultiArray(shapedArray) may share that IOSurface storage rather than copying.
    /// Storing those wrappers in savedEmbedding means each embedding retains an IOSurface,
    /// hitting the per-process 16384 IOSurface limit at ~15800 embeddings.
    /// This method breaks that chain by memcpy-ing the floats into plain heap memory.
    static func detachFromIOSurface(_ shapedArray: MLShapedArray<Float32>) -> MLMultiArray {
        let count = shapedArray.scalarCount
        let heapArray = try! MLMultiArray(shape: [1, NSNumber(value: count)], dataType: .float32)
        let dst = heapArray.dataPointer.assumingMemoryBound(to: Float32.self)
        shapedArray.withUnsafeShapedBufferPointer { ptr, _, _ in
            dst.update(from: ptr.baseAddress!, count: count)
        }
        return heapArray
    }

    init(resourcesAt baseURL: URL,
         spec: EmbeddingModelSpec = .mobileCLIPS2,
         configuration config: MLModelConfiguration = .init()
    ) throws {
        self.spec = spec
        guard spec.imageOutputType == .multiArrayFloat32 else {
            throw ImageEncodingError.unsupportedFeatureType
        }
        try spec.validateImageRuntimePreprocessing()
        let imgEncoderURL = baseURL.appending(path: spec.imageModelName)
        let imgEncoderModel = try MLModel(contentsOf: imgEncoderURL, configuration: config)
        try spec.validate(imageModel: imgEncoderModel)
        self.model = imgEncoderModel
    }

    public func computeImgEmbedding(img: UIImage) async throws -> MLShapedArray<Float32> {
        let imgEmbedding = try await self.encode(image: img)
        return imgEmbedding
    }

    /// Prediction queue
    let queue = DispatchQueue(label: "imgencoder.predict")

    public func encode(image: UIImage) async throws -> MLShapedArray<Float32> {
        do {
            let inputSize = CGSize(width: spec.imageSize, height: spec.imageSize)
            guard let buffer = Self.resizeAndConvertToBuffer(
                image: image,
                size: inputSize,
                preprocessing: spec.imagePreprocessing
            ) else {
                throw ImageEncodingError.bufferConversionError
            }

            guard let inputFeatures = try? MLDictionaryFeatureProvider(dictionary: [spec.imageInputName: buffer]) else {
                throw ImageEncodingError.featureProviderError
            }

            let result = try queue.sync { try model.prediction(from: inputFeatures) }
            guard let embeddingFeature = result.featureValue(for: spec.imageOutputName),
                  let multiArray = embeddingFeature.multiArrayValue,
                  multiArray.dataType == .float32,
                  multiArray.count == spec.embeddingDimension else {
                throw ImageEncodingError.predictionError
            }

            return MLShapedArray<Float32>(converting: multiArray)
        } catch {
            print("Error in encoding: \(error)")
            throw error
        }
    }

    /// Batch prediction: encode multiple images in one CoreML call.
    /// Uses MLArrayBatchProvider for efficient Neural Engine pipelining.
    /// All CoreML intermediates are scoped inside autoreleasepool to release
    /// Neural Engine IOSurface allocations promptly between batches.
    public func encodeBatch(images: [UIImage]) throws -> [MLShapedArray<Float32>] {
        let targetSize = CGSize(width: spec.imageSize, height: spec.imageSize)

        var embeddings = [MLShapedArray<Float32>]()
        embeddings.reserveCapacity(images.count)

        // autoreleasepool ensures CoreML's IOSurface-backed MLMultiArrays
        // and Espresso intermediates are released before the next batch
        try autoreleasepool {
            var featureProviders = [MLFeatureProvider]()
            featureProviders.reserveCapacity(images.count)

            for image in images {
                guard let buffer = Self.resizeAndConvertToBuffer(
                    image: image,
                    size: targetSize,
                    preprocessing: spec.imagePreprocessing
                ) else {
                    throw ImageEncodingError.bufferConversionError
                }
                let features = try MLDictionaryFeatureProvider(dictionary: [spec.imageInputName: buffer])
                featureProviders.append(features)
            }

            let batchProvider = MLArrayBatchProvider(array: featureProviders)

            // Single batch prediction call — Neural Engine handles pipelining
            let batchResults = try queue.sync { try model.predictions(fromBatch: batchProvider) }

            for i in 0..<batchResults.count {
                let result = batchResults.features(at: i)
                guard let embeddingFeature = result.featureValue(for: spec.imageOutputName),
                      let multiArray = embeddingFeature.multiArrayValue,
                      multiArray.dataType == .float32,
                      multiArray.count == spec.embeddingDimension else {
                    throw ImageEncodingError.predictionError
                }
                embeddings.append(MLShapedArray<Float32>(converting: multiArray))
            }
        }

        return embeddings
    }

    /// GPU-accelerated image resize using CoreImage CILanczosScaleTransform,
    /// then render directly to a pooled CVPixelBuffer.
    private static func resizeAndConvertToBuffer(
        image: UIImage,
        size: CGSize,
        preprocessing: ImagePreprocessing
    ) -> CVPixelBuffer? {
        guard preprocessing.pixelFormat == "32ARGB" else { return nil }
        guard let cgImage = image.cgImage else { return nil }

        let ciImage = CIImage(cgImage: cgImage)
        let scaleX = size.width / ciImage.extent.width
        let scaleY = size.height / ciImage.extent.height

        guard let filter = CIFilter(name: preprocessing.resizeFilter) else { return nil }
        filter.setValue(ciImage, forKey: kCIInputImageKey)
        filter.setValue(scaleY, forKey: kCIInputScaleKey)
        filter.setValue(scaleX / scaleY, forKey: kCIInputAspectRatioKey)

        guard let outputImage = filter.outputImage else { return nil }

        // Get a recycled buffer from the model-sized pool.
        var pixelBuffer: CVPixelBuffer?
        if let pool = pixelBufferPool(size: size, pixelFormat: kCVPixelFormatType_32ARGB) {
            let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pixelBuffer)
            guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }
            ciContext.render(outputImage, to: buffer)
            return buffer
        }

        // Fallback: create standalone buffer if pool init failed
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(size.width), Int(size.height),
            kCVPixelFormatType_32ARGB,
            attrs as CFDictionary,
            &pixelBuffer
        )
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }
        ciContext.render(outputImage, to: buffer)
        return buffer
    }
}

// Define the custom errors
enum ImageEncodingError: Error {
    case resizeError
    case bufferConversionError
    case featureProviderError
    case predictionError
    case unsupportedFeatureType
}
