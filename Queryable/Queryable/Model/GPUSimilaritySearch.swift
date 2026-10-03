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
        try validate(embeddings)
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
                to: destination.advanced(by: row * embeddingDim),
                using: &normalizedBuf
            )
        }

        ids = newIDs
        matrixBuffer = newMatrixBuffer
        rebuildGraph()

        print("[GPUSearch] Built index: \(n) embeddings in \(String(format: "%.3f", Date().timeIntervalSince(startTime)))s")
    }

    /// Add new embeddings to the existing index.
    func addEmbeddings(_ newEmbeddings: [String: MLMultiArray]) throws {
        try validate(newEmbeddings)
        guard !newEmbeddings.isEmpty else { return }
        let (newCount, countOverflow) = ids.count.addingReportingOverflow(newEmbeddings.count)
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
            let row = newIDs.count
            newIDs.append(id)
            writeNormalizedEmbedding(
                mlArray,
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
        to destination: UnsafeMutablePointer<Float16>,
        using normalizedBuffer: inout [Float32]
    ) {
        let source = embedding.dataPointer.assumingMemoryBound(to: Float32.self)
        var sumSquares: Float32 = 0
        vDSP_svesq(source, 1, &sumSquares, vDSP_Length(embeddingDim))
        let norm = sqrt(sumSquares)

        if norm > 1e-8 {
            var inverseNorm = 1.0 / norm
            vDSP_vsmul(source, 1, &inverseNorm, &normalizedBuffer, 1, vDSP_Length(embeddingDim))
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
        } else {
            for index in 0..<embeddingDim {
                destination[index] = 0
            }
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
        let resultTensor = graph.matrixMultiplication(
            primary: matrixPlaceholder,
            secondary: queryPlaceholder,
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
        guard queryEmbedding.scalarCount == embeddingDim else {
            throw SimilaritySearchError.dimensionMismatch(expected: embeddingDim, actual: queryEmbedding.scalarCount)
        }
        let n = ids.count
        guard n > 0,
              let cached = cachedGraph,
              let matBuf = matrixBuffer,
              cached.n == n else {
            return [:]
        }

        // L2-normalize query and convert to Float16
        let queryScalars = queryEmbedding.scalars
        let queryNorm = sqrt(vDSP.sumOfSquares(queryScalars))
        var queryFloat16 = [Float16](repeating: 0, count: embeddingDim)
        if queryNorm > 1e-8 {
            for j in 0..<embeddingDim {
                queryFloat16[j] = Float16(queryScalars[j] / queryNorm)
            }
        }

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

        guard let resultTensorData = results[cached.resultTensor] else { return [:] }

        // Read results back as Float16
        var resultFloat16 = [Float16](repeating: 0, count: n)
        resultTensorData.mpsndarray().readBytes(&resultFloat16, strideBytes: nil)

        // Build result dictionary
        var simDict = [String: Float](minimumCapacity: n)
        for i in 0..<n {
            simDict[ids[i]] = Float(resultFloat16[i])
        }

        return simDict
    }

    private func validate(_ embeddings: [String: MLMultiArray]) throws {
        for (id, embedding) in embeddings {
            guard embedding.dataType == .float32, embedding.count == embeddingDim else {
                throw SimilaritySearchError.invalidEmbedding(
                    id: id,
                    expected: embeddingDim,
                    actual: embedding.count
                )
            }
        }
    }
}

enum SimilaritySearchError: Error {
    case dimensionMismatch(expected: Int, actual: Int)
    case invalidEmbedding(id: String, expected: Int, actual: Int)
    case bufferAllocationFailed
}
