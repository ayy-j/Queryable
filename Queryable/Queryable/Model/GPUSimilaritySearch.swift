//
//  GPUSimilaritySearch.swift
//  Queryable
//
//  GPU-accelerated similarity search using MPSGraph matrix multiplication.
//  Replaces per-embedding CPU cosine similarity with a single [N,D]×[D,1] GPU matmul.
//
//  Performance strategy:
//  - Pre-allocate MTLBuffer for the embedding matrix (avoids ~27MB copy per search)
//  - Cache compiled MPSGraphExecutable (avoids graph recompilation per search)
//  - Only allocate a tiny buffer for the query vector on each search
//

import Foundation
import CoreML
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import Accelerate

class GPUSimilaritySearch {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    /// Photo IDs in the same order as rows in the embedding matrix
    private(set) var ids: [String] = []

    /// Pre-allocated GPU buffer for the embedding matrix
    private var matrixBuffer: MTLBuffer?

    /// Cached compiled graph + placeholders (invalidated when index changes)
    private var cachedGraph: CachedGraph?

    private let embeddingDim: Int

    var count: Int { ids.count }
    var embeddingDimension: Int { embeddingDim }

    private struct CachedGraph {
        let graph: MPSGraph
        let matrixPlaceholder: MPSGraphTensor
        let queryPlaceholder: MPSGraphTensor
        let resultTensor: MPSGraphTensor
        let n: Int
    }

    init?(embeddingDimension: Int) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              embeddingDimension > 0 else {
            return nil
        }
        self.device = device
        self.commandQueue = commandQueue
        self.embeddingDim = embeddingDimension
    }

    /// Build the GPU index from the in-memory embedding dictionary.
    /// Embeddings are L2-normalized and converted to Float16.
    func buildIndex(from embeddings: [String: MLMultiArray]) throws {
        let norms = try validate(embeddings)
        let startTime = Date()
        let n = embeddings.count
        guard n > 0 else {
            ids.removeAll()
            matrixBuffer = nil
            cachedGraph = nil
            return
        }
        let newMatrixBuffer = try makeMatrixBuffer(embeddingCount: n)
        let destination = newMatrixBuffer.contents().assumingMemoryBound(to: Float16.self)
        var normalizedBuf = [Float32](repeating: 0, count: embeddingDim)
        var newIDs = [String]()
        newIDs.reserveCapacity(n)
        for (row, (id, mlArray)) in embeddings.enumerated() {
            newIDs.append(id)
            writeNormalizedEmbedding(
                mlArray,
                norm: norms[id]!,
                to: destination.advanced(by: row * embeddingDim),
                using: &normalizedBuf
            )
        }

        ids = newIDs
        matrixBuffer = newMatrixBuffer
        rebuildGraph()

        print("[GPUSearch] Built index: \(n) embeddings in \(String(format: "%.3f", Date().timeIntervalSince(startTime)))s")
    }

    /// Upsert embeddings, preserving the row positions of existing IDs.
    /// Validate and stage the whole batch before publishing any index changes.
    func addEmbeddings(_ newEmbeddings: [String: MLMultiArray]) throws {
        let norms = try validate(newEmbeddings)
        guard !newEmbeddings.isEmpty else { return }
        let existingRows = Dictionary(uniqueKeysWithValues: ids.enumerated().map { ($0.element, $0.offset) })
        let addedCount = newEmbeddings.keys.reduce(into: 0) { count, id in
            if existingRows[id] == nil { count += 1 }
        }
        let (newCount, countOverflow) = ids.count.addingReportingOverflow(addedCount)
        guard !countOverflow else { throw SimilaritySearchError.bufferAllocationFailed }
        let newMatrixBuffer = try makeMatrixBuffer(embeddingCount: newCount)
        let destination = newMatrixBuffer.contents().assumingMemoryBound(to: Float16.self)
        if let matrixBuffer, !ids.isEmpty {
            let source = matrixBuffer.contents().assumingMemoryBound(to: Float16.self)
            destination.update(from: source, count: ids.count * embeddingDim)
        }

        var normalizedBuf = [Float32](repeating: 0, count: embeddingDim)
        var newIDs = ids
        newIDs.reserveCapacity(newCount)
        for (id, mlArray) in newEmbeddings {
            let row: Int
            if let existingRow = existingRows[id] {
                row = existingRow
            } else {
                row = newIDs.count
                newIDs.append(id)
            }
            writeNormalizedEmbedding(
                mlArray,
                norm: norms[id]!,
                to: destination.advanced(by: row * embeddingDim),
                using: &normalizedBuf
            )
        }

        ids = newIDs
        matrixBuffer = newMatrixBuffer
        rebuildGraph()
    }

    /// Remove embeddings by their IDs, copying retained rows directly between Metal buffers.
    func removeEmbeddings(_ idsToRemove: Set<String>) {
        guard !idsToRemove.isEmpty else { return }

        let retainedCount = ids.reduce(into: 0) { count, id in
            if !idsToRemove.contains(id) { count += 1 }
        }
        var newIds = [String]()
        newIds.reserveCapacity(retainedCount)
        guard retainedCount > 0 else {
            ids.removeAll()
            matrixBuffer = nil
            cachedGraph = nil
            return
        }
        guard let newMatrixBuffer = try? makeMatrixBuffer(embeddingCount: retainedCount),
              let oldMatrixBuffer = matrixBuffer else {
            ids.removeAll()
            matrixBuffer = nil
            cachedGraph = nil
            return
        }
        let source = oldMatrixBuffer.contents().assumingMemoryBound(to: Float16.self)
        let destination = newMatrixBuffer.contents().assumingMemoryBound(to: Float16.self)

        var destinationRow = 0
        for (i, id) in ids.enumerated() {
            if !idsToRemove.contains(id) {
                newIds.append(id)
                let sourceOffset = i * embeddingDim
                let destinationOffset = destinationRow * embeddingDim
                destination.advanced(by: destinationOffset)
                    .update(from: source.advanced(by: sourceOffset), count: embeddingDim)
                destinationRow += 1
            }
        }

        ids = newIds
        matrixBuffer = newMatrixBuffer
        rebuildGraph()
    }

    // MARK: - GPU Buffer Management

    private func makeMatrixBuffer(embeddingCount: Int) throws -> MTLBuffer {
        let (elementCount, elementOverflow) = embeddingCount.multipliedReportingOverflow(by: embeddingDim)
        let (byteCount, byteOverflow) = elementCount.multipliedReportingOverflow(by: MemoryLayout<Float16>.size)
        guard !elementOverflow, !byteOverflow,
              let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw SimilaritySearchError.bufferAllocationFailed
        }
        return buffer
    }

    private func writeNormalizedEmbedding(
        _ embedding: MLMultiArray,
        norm: Double,
        to destination: UnsafeMutablePointer<Float16>,
        using normalizedBuffer: inout [Float32]
    ) {
        let source = embedding.dataPointer.assumingMemoryBound(to: Float32.self)
        // Validation has established a finite, nonzero norm and contiguous layout.
        // Double accumulation/division avoids Float32 overflow for finite inputs.
        for index in 0..<embeddingDim {
            normalizedBuffer[index] = Float32(Double(source[index]) / norm)
        }
        normalizedBuffer.withUnsafeBufferPointer { sourceBuffer in
            var sourceImage = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: sourceBuffer.baseAddress!),
                height: 1,
                width: vImagePixelCount(embeddingDim),
                rowBytes: embeddingDim * MemoryLayout<Float32>.size
            )
            var destinationImage = vImage_Buffer(
                data: UnsafeMutableRawPointer(destination),
                height: 1,
                width: vImagePixelCount(embeddingDim),
                rowBytes: embeddingDim * MemoryLayout<Float16>.size
            )
            vImageConvert_PlanarFtoPlanar16F(&sourceImage, &destinationImage, 0)
        }
    }

    /// Build the cached graph for the current Metal buffer.
    private func rebuildGraph() {
        let n = ids.count
        guard n > 0 else {
            matrixBuffer = nil
            cachedGraph = nil
            return
        }
        guard matrixBuffer != nil else {
            cachedGraph = nil
            return
        }

        // Build and cache the MPSGraph for this matrix size
        let graph = MPSGraph()
        let matrixShape: [NSNumber] = [NSNumber(value: n), NSNumber(value: embeddingDim)]
        let queryShape: [NSNumber] = [NSNumber(value: embeddingDim), NSNumber(value: 1)]

        let matrixPlaceholder = graph.placeholder(shape: matrixShape, dataType: .float16, name: "embeddings")
        let queryPlaceholder = graph.placeholder(shape: queryShape, dataType: .float16, name: "query")
        // Retain compact Float16 storage, but accumulate dense dot products in
        // Float32. Float16 accumulation drifts measurably at larger dimensions.
        let matrixFloat32 = graph.cast(matrixPlaceholder, to: .float32, name: "embeddingsFloat32")
        let queryFloat32 = graph.cast(queryPlaceholder, to: .float32, name: "queryFloat32")
        let resultTensor = graph.matrixMultiplication(
            primary: matrixFloat32,
            secondary: queryFloat32,
            name: "similarity"
        )

        cachedGraph = CachedGraph(
            graph: graph,
            matrixPlaceholder: matrixPlaceholder,
            queryPlaceholder: queryPlaceholder,
            resultTensor: resultTensor,
            n: n
        )
    }

    // MARK: - Search

    /// Compute similarity scores for a query embedding against all stored embeddings.
    /// Returns [photoID: similarity_score].
    func search(queryEmbedding: MLShapedArray<Float32>) throws -> [String: Float] {
        let queryNorm = try SimilarityVectorValidation.norm(of: queryEmbedding, dimension: embeddingDim)
        let n = ids.count
        guard n > 0 else { return [:] }
        guard let cached = cachedGraph,
              let matBuf = matrixBuffer,
              cached.n == n else {
            throw SimilaritySearchError.indexUnavailable
        }

        let queryFloat16 = queryEmbedding.scalars.map { Float16(Double($0) / queryNorm) }

        // Create tensor data from pre-allocated matrix buffer (no copy)
        let matrixShape: [NSNumber] = [NSNumber(value: n), NSNumber(value: embeddingDim)]
        let matrixTensorData = MPSGraphTensorData(
            matBuf,
            shape: matrixShape,
            dataType: .float16
        )

        // Only the query vector needs a fresh buffer (~1KB)
        let queryShape: [NSNumber] = [NSNumber(value: embeddingDim), NSNumber(value: 1)]
        let queryTensorData = queryFloat16.withUnsafeBufferPointer { ptr in
            MPSGraphTensorData(
                device: MPSGraphDevice(mtlDevice: device),
                data: Data(buffer: ptr),
                shape: queryShape,
                dataType: .float16
            )
        }

        // Execute on GPU using cached graph
        let results = cached.graph.run(
            with: commandQueue,
            feeds: [cached.matrixPlaceholder: matrixTensorData,
                    cached.queryPlaceholder: queryTensorData],
            targetTensors: [cached.resultTensor],
            targetOperations: nil
        )

        guard let resultTensorData = results[cached.resultTensor] else {
            throw SimilaritySearchError.resultsUnavailable
        }

        // The graph accumulates and returns Float32 similarity scores.
        var resultFloat32 = [Float32](repeating: 0, count: n)
        resultTensorData.mpsndarray().readBytes(&resultFloat32, strideBytes: nil)

        guard resultFloat32.allSatisfy(\.isFinite) else {
            throw SimilaritySearchError.nonFiniteVector
        }

        // Build result dictionary
        var simDict = [String: Float](minimumCapacity: n)
        for i in 0..<n {
            simDict[ids[i]] = resultFloat32[i]
        }

        return simDict
    }

    private func validate(_ embeddings: [String: MLMultiArray]) throws -> [String: Double] {
        var norms = [String: Double](minimumCapacity: embeddings.count)
        for (id, embedding) in embeddings {
            norms[id] = try SimilarityVectorValidation.norm(of: embedding, dimension: embeddingDim, id: id)
        }
        return norms
    }
}

/// Validation is independent of Metal so malformed inputs can be checked on every host.
/// Only [D] and [1,D] contiguous Float32 vectors are supported by the raw-pointer path.
enum SimilarityVectorValidation {
    static func norm(of embedding: MLMultiArray, dimension: Int, id: String) throws -> Double {
        guard embedding.dataType == .float32, embedding.count == dimension else {
            throw SimilaritySearchError.invalidEmbedding(id: id, expected: dimension, actual: embedding.count)
        }
        let shape = embedding.shape.map(\.intValue)
        let strides = embedding.strides.map(\.intValue)
        guard (shape == [dimension] || shape == [1, dimension]), strides.last == 1 else {
            throw SimilaritySearchError.unsupportedLayout
        }
        // Do not read dataPointer until scalar type, shape, and strides are validated.
        let source = embedding.dataPointer.assumingMemoryBound(to: Float32.self)
        var squares = 0.0
        for index in 0..<dimension {
            guard source[index].isFinite else { throw SimilaritySearchError.nonFiniteVector }
            squares += Double(source[index]) * Double(source[index])
        }
        return try checkedNorm(squares)
    }

    static func norm(of embedding: MLShapedArray<Float32>, dimension: Int) throws -> Double {
        guard embedding.scalarCount == dimension else {
            throw SimilaritySearchError.dimensionMismatch(expected: dimension, actual: embedding.scalarCount)
        }
        guard embedding.shape == [dimension] || embedding.shape == [1, dimension] else {
            throw SimilaritySearchError.unsupportedLayout
        }
        var squares = 0.0
        for value in embedding.scalars {
            guard value.isFinite else { throw SimilaritySearchError.nonFiniteVector }
            squares += Double(value) * Double(value)
        }
        return try checkedNorm(squares)
    }

    private static func checkedNorm(_ squares: Double) throws -> Double {
        let norm = sqrt(squares)
        guard norm > 1e-8 else { throw SimilaritySearchError.zeroNormVector }
        return norm
    }
}

enum SimilaritySearchError: Error, Equatable {
    case dimensionMismatch(expected: Int, actual: Int)
    case invalidEmbedding(id: String, expected: Int, actual: Int)
    case unsupportedLayout
    case nonFiniteVector
    case zeroNormVector
    case indexUnavailable
    case resultsUnavailable
    case bufferAllocationFailed
}
