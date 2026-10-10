//
//  ImgEncoder.swift
//  Queryable
//
//  Created by Ke Fang on 2022/12/08.
//

import Foundation
import CoreML
import CoreImage
#if canImport(UIKit)
import UIKit
#endif

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
            CVPixelBufferPoolFlush(pool, .excessBuffers)
        }
    }

    /// Pool of recycled IOSurface-backed buffers for a given input geometry.
    /// Pools are keyed by size and pixel format so a model with a different
    /// input resolution (e.g. 384 px SigLIP) still recycles its buffers.
    static func pixelBufferPool(size: CGSize, pixelFormat: OSType) -> CVPixelBufferPool? {
        guard let width = Int(exactly: size.width), let height = Int(exactly: size.height),
              width > 0, height > 0 else { return nil }
        let key = BufferPoolKey(
            width: width,
            height: height,
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
    /// This method breaks that chain by copying logical scalars into plain heap memory.
    static func detachFromIOSurface(_ shapedArray: MLShapedArray<Float32>) -> MLMultiArray {
        let count = shapedArray.scalarCount
        let heapArray = try! MLMultiArray(shape: [1, NSNumber(value: count)], dataType: .float32)
        let dst = heapArray.dataPointer.assumingMemoryBound(to: Float32.self)
        // Prediction arrays may be strided; a raw contiguous copy would include padding.
        shapedArray.scalars.withUnsafeBufferPointer { ptr in
            if let source = ptr.baseAddress {
                dst.update(from: source, count: count)
            }
        }
        return heapArray
    }

    init(resourcesAt baseURL: URL,
         spec: EmbeddingModelSpec = .appDefault,
         configuration config: MLModelConfiguration = .init()
    ) throws {
        self.spec = spec
        guard spec.imageOutputType == .multiArrayFloat32 else {
            throw ImageEncodingError.unsupportedFeatureType
        }
        try spec.validateImageRuntimePreprocessing()
        let missing = spec.missingArtifacts(resourcesAt: baseURL)
        if let firstMissing = missing.first {
            throw ModelArtifactError.missingArtifact(firstMissing)
        }
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
            let inputFeatures = try Self.imageFeatureProvider(for: image, spec: spec)
            let result = try queue.sync { try model.prediction(from: inputFeatures) }
            return try Self.validatedEmbedding(from: result, spec: spec)
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
        guard !images.isEmpty else { return [] }

        // autoreleasepool ensures CoreML's IOSurface-backed MLMultiArrays
        // and Espresso intermediates are released before the next batch
        return try autoreleasepool {
            var featureProviders = [MLFeatureProvider]()
            featureProviders.reserveCapacity(images.count)

            for image in images {
                featureProviders.append(try Self.imageFeatureProvider(for: image, spec: spec))
            }

            let batchProvider = MLArrayBatchProvider(array: featureProviders)

            // Single batch prediction call — Neural Engine handles pipelining
            let batchResults = try queue.sync { try model.predictions(fromBatch: batchProvider) }

            return try Self.validatedEmbeddings(from: batchResults, expectedCount: images.count, spec: spec)
        }
    }

    /// Shared single/batch input path. Reject unsupported preprocessing before Core Image KVC.
    static func imageFeatureProvider(for image: UIImage, spec: EmbeddingModelSpec) throws -> MLFeatureProvider {
        try spec.validateImageRuntimePreprocessing()
        guard let buffer = resizeAndConvertToBuffer(
            image: image,
            size: CGSize(width: spec.imageSize, height: spec.imageSize),
            preprocessing: spec.imagePreprocessing
        ) else {
            throw ImageEncodingError.bufferConversionError
        }
        return try MLDictionaryFeatureProvider(dictionary: [spec.imageInputName: buffer])
    }

    /// Validate the runtime output, including providers whose metadata passed model validation.
    /// Preserve the model's shape and values; normalization remains the caller's responsibility.
    static func validatedEmbedding(
        from result: MLFeatureProvider,
        spec: EmbeddingModelSpec
    ) throws -> MLShapedArray<Float32> {
        guard spec.imageOutputType == .multiArrayFloat32,
              result.featureNames.contains(spec.imageOutputName),
              let feature = result.featureValue(for: spec.imageOutputName),
              feature.type == .multiArray,
              let multiArray = feature.multiArrayValue,
              multiArray.dataType == .float32,
              multiArray.count == spec.embeddingDimension else {
            throw ImageEncodingError.predictionError
        }
        let shape = multiArray.shape.map { $0.intValue }
        guard shape == [spec.embeddingDimension] || shape == [1, spec.embeddingDimension] else {
            throw ImageEncodingError.predictionError
        }

        let embedding = MLShapedArray<Float32>(converting: multiArray)
        var squaredNorm: Double = 0
        for value in embedding.scalars {
            guard value.isFinite else { throw ImageEncodingError.predictionError }
            let scalar = Double(value)
            squaredNorm += scalar * scalar
        }
        // Match the search layer's minimum L2 norm without Float32 overflow/underflow.
        guard squaredNorm.isFinite, squaredNorm.squareRoot() > 1e-8 else {
            throw ImageEncodingError.predictionError
        }
        return embedding
    }

    /// Return the complete validated batch or throw; no partially validated batch escapes.
    static func validatedEmbeddings(
        from results: MLBatchProvider,
        expectedCount: Int,
        spec: EmbeddingModelSpec
    ) throws -> [MLShapedArray<Float32>] {
        guard expectedCount >= 0, results.count == expectedCount else {
            throw ImageEncodingError.predictionError
        }
        return try (0..<expectedCount).map {
            try validatedEmbedding(from: results.features(at: $0), spec: spec)
        }
    }

    /// GPU-accelerated image resize using CoreImage CILanczosScaleTransform,
    /// then render directly to a pooled CVPixelBuffer.
    private static func resizeAndConvertToBuffer(
        image: UIImage,
        size: CGSize,
        preprocessing: ImagePreprocessing
    ) -> CVPixelBuffer? {
        guard let width = Int(exactly: size.width), let height = Int(exactly: size.height),
              width > 0, height > 0 else { return nil }
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
            width, height,
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
