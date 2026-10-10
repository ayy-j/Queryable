import XCTest
import CoreML
import UIKit
@testable import Queryable

private func searchVector(_ dimension: Int = 768, _ values: [Float32] = [1, 0]) -> MLShapedArray<Float32> {
    MLShapedArray(scalars: values + Array(repeating: 0, count: dimension - values.count), shape: [1, dimension])
}

final class PhotoSearchModelTests: XCTestCase {
    private func spec(_ dimension: Int) throws -> EmbeddingModelSpec {
        switch dimension {
        case 512: return .mobileCLIPS2
        case 768: return .mobileCLIP2S4
        default:
            return try .sigLIPSo400m(revision: "test", imageModelName: "image.mlmodelc",
                                    textModelName: "text.mlmodelc", tokenizerAssets: ["tokenizer.json"],
                                    imagePreprocessing: EmbeddingModelSpec.mobileCLIP2S4.imagePreprocessing,
                                    imageInputName: "image", imageOutputName: "embedding",
                                    textInputName: "tokens", textInputType: .multiArrayInt32,
                                    textOutputName: "embedding", embeddingDimension: dimension)
        }
    }

    func testKnownCPUScoringAndRankingAtEveryRequiredDimension() throws {
        // Analytic cosine reference, tolerance declared independently of GPU Float16 scoring.
        for dimension in [512, 768, 1152, 1536] {
            let model = PhotoSearcherModel(spec: try spec(dimension))
            let query = searchVector(dimension)
            let scores = try model.similarityScores(query: query, embeddings: [
                "same": MLMultiArray(searchVector(dimension, [3, 0])),
                "diagonal": MLMultiArray(searchVector(dimension, [1, 1])),
                "orthogonal": MLMultiArray(searchVector(dimension, [0, 2])),
                "opposite": MLMultiArray(searchVector(dimension, [-4, 0]))
            ])
            XCTAssertEqual(try XCTUnwrap(scores["same"]), 1, accuracy: 1e-6)
            XCTAssertEqual(try XCTUnwrap(scores["diagonal"]), Float(1 / sqrt(2.0)), accuracy: 1e-6)
            XCTAssertEqual(try XCTUnwrap(scores["orthogonal"]), 0, accuracy: 1e-6)
            XCTAssertEqual(try XCTUnwrap(scores["opposite"]), -1, accuracy: 1e-6)
            XCTAssertEqual(scores.sorted { $0.value > $1.value }.map(\.key), ["same", "diagonal", "orthogonal", "opposite"])
        }
    }

    func testCPUScorerRejectsInvalidQueriesEvenWithEmptyIndex() {
        let model = PhotoSearcherModel()
        for query in [searchVector(767), searchVector(768, [0]), searchVector(768, [.nan]), searchVector(768, [.infinity])] {
            XCTAssertThrowsError(try model.similarityScores(query: query, embeddings: [:]))
        }
        XCTAssertEqual(try model.similarityScores(query: searchVector(), embeddings: [:]), [:])
    }

    func testCPUScorerRejectsInvalidIndexInsteadOfPublishingPartialResults() throws {
        let model = PhotoSearcherModel()
        let wrongType = try MLMultiArray(shape: [1, 768], dataType: .double)
        let matrix = MLMultiArray(MLShapedArray<Float32>(scalars: Array(repeating: 1, count: 768), shape: [2, 384]))
        let backing = UnsafeMutablePointer<Float32>.allocate(capacity: 1536)
        backing.initialize(repeating: 1, count: 1536)
        let strided = try MLMultiArray(dataPointer: backing, shape: [1, 768], dataType: .float32,
                                      strides: [1536, 2], deallocator: { $0.assumingMemoryBound(to: Float32.self).deallocate() })
        for invalid in [MLMultiArray(searchVector(767)), MLMultiArray(searchVector(768, [0])),
                        MLMultiArray(searchVector(768, [.nan])), MLMultiArray(searchVector(768, [.infinity])),
                        wrongType, matrix, strided] {
            XCTAssertThrowsError(try model.similarityScores(query: searchVector(), embeddings: [
                "valid": MLMultiArray(searchVector()), "invalid": invalid
            ]))
        }
    }

    func testCosineAndSphericalDistanceRejectInvalidInputsAndStayFinite() throws {
        let model = PhotoSearcherModel()
        XCTAssertEqual(try model.cosine_similarity(A: searchVector(), B: searchVector(768, [-1])), -1)
        XCTAssertEqual(try model.spherical_dist_loss(A: searchVector(), B: searchVector(768, [-1])), Float(Double.pi * Double.pi / 2), accuracy: 1e-6)
        XCTAssertThrowsError(try model.cosine_similarity(A: searchVector(), B: searchVector(768, [0])))
        XCTAssertThrowsError(try model.spherical_dist_loss(A: searchVector(), B: searchVector(767)))
        let large = searchVector(768, [.greatestFiniteMagnitude, .greatestFiniteMagnitude])
        XCTAssertEqual(try model.cosine_similarity(A: large, B: large), 1, accuracy: 1e-6)
        let scores = try model.similarityScores(query: large, embeddings: ["large": MLMultiArray(large)])
        XCTAssertEqual(try XCTUnwrap(scores["large"]), 1, accuracy: 1e-6)
    }

    func testUnloadedTextEncoderThrowsInsteadOfCrashing() {
        XCTAssertThrowsError(try PhotoSearcherModel().text_embedding(prompt: "test")) { error in
            guard case PhotoSearchError.encoderNotReady = error else { return XCTFail("Unexpected error: \(error)") }
        }
    }

    func testTextPredictionErrorsPropagateAndInvalidOutputsAreRejected() {
        let failure = PhotoSearcherModel(textEmbeddingProvider: { _ in throw CocoaError(.fileReadUnknown) })
        XCTAssertThrowsError(try failure.text_embedding(prompt: "test")) { error in
            XCTAssertEqual((error as NSError).code, CocoaError.fileReadUnknown.rawValue)
        }
        for output in [searchVector(767), searchVector(768, [0]), searchVector(768, [.nan])] {
            let model = PhotoSearcherModel(textEmbeddingProvider: { _ in output })
            XCTAssertThrowsError(try model.text_embedding(prompt: "test")) { error in
                guard case PhotoSearchError.invalidTextEmbedding = error else {
                    return XCTFail("Invalid text output must identify a model failure: \(error)")
                }
            }
        }
    }
}

@MainActor
final class SearchReliabilityTests: XCTestCase {
    private func searcher(spec: EmbeddingModelSpec = .mobileCLIP2S4) async -> PhotoSearcher {
        let dimension = spec.embeddingDimension
        let searcher = PhotoSearcher(modelSpec: spec, indexingOperations: PhotoIndexingOperations(
            fetchImage: { _, _ in UIImage() },
            encodeBatch: { images in images.map { _ in searchVector(dimension) } },
            encodeImage: { _ in searchVector(dimension) },
            save: { _ in }
        ))
        // Keep this test independent of the simulator's persisted result-count preference.
        let priorLimit = UserDefaults.standard.object(forKey: "TOPK_SIM")
        searcher.TOPK_SIM = 10
        if let priorLimit {
            UserDefaults.standard.set(priorLimit, forKey: "TOPK_SIM")
        } else {
            UserDefaults.standard.removeObject(forKey: "TOPK_SIM")
        }
        // The injected save path leaves the GPU empty and exercises the real CPU fallback.
        await searcher.buildIndex(assets: [PhotoAsset(identifier: "search-test-photo", phAsset: nil)])
        return searcher
    }

    func testUnloadedEncoderProducesSearchErrorAndClearsStaleResults() async {
        let searcher = await searcher()
        searcher.searchResultPhotoAssets = [PhotoAsset(identifier: "stale", phAsset: nil)]
        await searcher.search(with: "test")
        XCTAssertEqual(searcher.searchResultCode, .SEARCH_ERROR)
        XCTAssertNotNil(searcher.searchErrorMessage)
        XCTAssertTrue(searcher.searchResultPhotoAssets.isEmpty)
    }

    func testPredictionFailureThenRetryProducesResultsAndClearsError() async {
        let searcher = await searcher()
        var shouldFail = true
        searcher.photoSearchModel = PhotoSearcherModel(textEmbeddingProvider: { _ in
            if shouldFail { throw CocoaError(.fileReadUnknown) }
            return searchVector()
        })
        await searcher.search(with: "test")
        XCTAssertEqual(searcher.searchResultCode, .SEARCH_ERROR)
        XCTAssertNotNil(searcher.searchErrorMessage)
        XCTAssertTrue(searcher.searchResultPhotoAssets.isEmpty)
        shouldFail = false
        await searcher.search(with: "retry")
        XCTAssertEqual(searcher.searchResultCode, .HAS_RESULT)
        XCTAssertNil(searcher.searchErrorMessage)
        XCTAssertEqual(searcher.searchResultPhotoAssets.map(\.id), ["search-test-photo"])
    }

    func testInvalidPredictionNeverProducesResults() async {
        let searcher = await searcher()
        for query in [searchVector(767), searchVector(768, [0]), searchVector(768, [.nan])] {
            searcher.photoSearchModel = PhotoSearcherModel(textEmbeddingProvider: { _ in query })
            await searcher.search(with: "test")
            XCTAssertEqual(searcher.searchResultCode, .SEARCH_ERROR)
            XCTAssertTrue(searcher.searchResultPhotoAssets.isEmpty)
        }
    }

    func testMissingSimilarPhotoStopsSpinnerAndRetryClearsError() async {
        let searcher = await searcher()
        searcher.similarPhotoAssets = [PhotoAsset(identifier: "stale", phAsset: nil)]
        await searcher.similarPhoto(with: PhotoAsset(identifier: "missing", phAsset: nil))
        XCTAssertFalse(searcher.isFindingSimilarPhotos)
        XCTAssertNotNil(searcher.similarPhotoErrorMessage)
        XCTAssertTrue(searcher.similarPhotoAssets.isEmpty)
        await searcher.similarPhoto(with: PhotoAsset(identifier: "search-test-photo", phAsset: nil))
        XCTAssertFalse(searcher.isFindingSimilarPhotos)
        XCTAssertNil(searcher.similarPhotoErrorMessage)
        XCTAssertEqual(searcher.similarPhotoAssets.map(\.id), ["search-test-photo"])
    }

    func testS2CPUFallbackUsesRequestedDimension() async {
        let searcher = await searcher(spec: .mobileCLIPS2)
        XCTAssertEqual(searcher.photoSearchModel.spec, .mobileCLIPS2)
        searcher.photoSearchModel = PhotoSearcherModel(spec: .mobileCLIPS2, textEmbeddingProvider: { _ in searchVector(512) })
        await searcher.search(with: "test")
        XCTAssertEqual(searcher.searchResultCode, .HAS_RESULT)
        await searcher.similarPhoto(with: PhotoAsset(identifier: "search-test-photo", phAsset: nil))
        XCTAssertNil(searcher.similarPhotoErrorMessage)
        XCTAssertEqual(searcher.similarPhotoAssets.map(\.id), ["search-test-photo"])
    }

    func testEmptyIndexRemainsNeverIndexedWithoutLoadingTextModel() async {
        let searcher = PhotoSearcher()
        await searcher.search(with: "test")
        XCTAssertEqual(searcher.searchResultCode, .NEVER_INDEXED)
        XCTAssertNil(searcher.searchErrorMessage)
    }
}
