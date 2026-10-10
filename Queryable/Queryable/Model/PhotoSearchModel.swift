//
//  PhotoSearcherModel.swift
//  Queryable
//
import CoreML
import Foundation

struct PhotoSearcherModel {
    private var textEncoder: TextEncoder?
    private let textEmbeddingProvider: ((String) throws -> MLShapedArray<Float32>)?
    private(set) var spec: EmbeddingModelSpec

    /// The optional provider exercises prediction failures without loading model files.
    init(spec: EmbeddingModelSpec = .appDefault,
         textEmbeddingProvider: ((String) throws -> MLShapedArray<Float32>)? = nil) {
        self.spec = spec
        self.textEmbeddingProvider = textEmbeddingProvider
    }

    mutating func load_text_encoder(resourcesAt resourceURL: URL, spec: EmbeddingModelSpec) throws {
        let encoder = try TextEncoder(resourcesAt: resourceURL, spec: spec)
        textEncoder = encoder
        self.spec = spec
    }

    mutating func releaseTextEncoder() { textEncoder = nil }

    func text_embedding(prompt: String) throws -> MLShapedArray<Float32> {
        let embedding: MLShapedArray<Float32>
        if let provider = textEmbeddingProvider {
            embedding = try provider(prompt)
        } else {
            guard let textEncoder else { throw PhotoSearchError.encoderNotReady }
            embedding = try textEncoder.computeTextEmbedding(prompt: prompt)
        }
        do {
            _ = try SimilarityVectorValidation.norm(of: embedding, dimension: spec.embeddingDimension)
        } catch {
            throw PhotoSearchError.invalidTextEmbedding
        }
        return embedding
    }

    /// Validate the complete input set before publishing any CPU search results.
    func similarityScores(query: MLShapedArray<Float32>, embeddings: [String: MLMultiArray]) throws -> [String: Float] {
        let dimension = spec.embeddingDimension
        let queryNorm = try SimilarityVectorValidation.norm(of: query, dimension: dimension)
        let queryValues = query.scalars
        var scores = [String: Float](minimumCapacity: embeddings.count)
        for (id, embedding) in embeddings {
            let norm = try SimilarityVectorValidation.norm(of: embedding, dimension: dimension, id: id)
            // The shared validator has checked type, shape, strides, and finite values.
            let values = embedding.dataPointer.assumingMemoryBound(to: Float32.self)
            var score = 0.0
            for index in 0..<dimension {
                score += (Double(queryValues[index]) / queryNorm) * (Double(values[index]) / norm)
            }
            scores[id] = Float(min(1, max(-1, score)))
        }
        return scores
    }

    func cosine_similarity(A: MLShapedArray<Float32>, B: MLShapedArray<Float32>) throws -> Float {
        let aNorm = try SimilarityVectorValidation.norm(of: A, dimension: spec.embeddingDimension)
        let bNorm = try SimilarityVectorValidation.norm(of: B, dimension: spec.embeddingDimension)
        let dot = zip(A.scalars, B.scalars).reduce(0.0) {
            $0 + (Double($1.0) / aNorm) * (Double($1.1) / bNorm)
        }
        return Float(min(1, max(-1, dot)))
    }

    func spherical_dist_loss(A: MLShapedArray<Float32>, B: MLShapedArray<Float32>) throws -> Float {
        let cosine = try cosine_similarity(A: A, B: B)
        let angle = acos(Double(cosine))
        return Float(angle * angle / 2)
    }
}

enum PhotoSearchError: Error, LocalizedError {
    case encoderNotReady
    case invalidTextEmbedding
    case referencePhotoMissing
    case photoAccessUnavailable

    var errorDescription: String? {
        switch self {
        case .encoderNotReady:
            return "The search model is not ready. Reopen the app and try again."
        case .invalidTextEmbedding:
            return "The search model returned unusable data. Reopen the app and try again."
        case .photoAccessUnavailable:
            return "Photo access is unavailable. Restore access in Settings and retry."
        case .referencePhotoMissing:
            return "This photo is not in the saved index. Update the index and try again."
        }
    }
}
