import XCTest
import CoreML
@testable import Queryable

final class SimilarityValidationTests: XCTestCase {
    private let dimensions = [512, 768, 1152, 1536]

    private func vector(_ dimension: Int, value: Float32 = 1, shape: [Int]? = nil) -> MLShapedArray<Float32> {
        MLShapedArray(scalars: Array(repeating: value, count: dimension), shape: shape ?? [1, dimension])
    }

    private func assertError<T>(_ expected: SimilaritySearchError,
                                _ operation: () throws -> T,
                                file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try operation(), file: file, line: line) { error in
            XCTAssertEqual(error as? SimilaritySearchError, expected, file: file, line: line)
        }
    }

    func testContiguousVectorsAndQueriesAtEveryRequiredDimension() throws {
        for dimension in dimensions {
            for shape in [[dimension], [1, dimension]] {
                let shaped = vector(dimension, shape: shape)
                let expected = sqrt(Double(dimension))
                XCTAssertEqual(try SimilarityVectorValidation.norm(of: shaped, dimension: dimension), expected)
                XCTAssertEqual(try SimilarityVectorValidation.norm(of: MLMultiArray(shaped), dimension: dimension, id: "photo"), expected)
            }
        }
    }

    func testShortAndLongVectorsAreRejected() {
        for dimension in dimensions {
            for length in [dimension - 1, dimension + 1] {
                let shaped = vector(length)
                assertError(.dimensionMismatch(expected: dimension, actual: length)) {
                    try SimilarityVectorValidation.norm(of: shaped, dimension: dimension)
                }
                assertError(.invalidEmbedding(id: "photo", expected: dimension, actual: length)) {
                    try SimilarityVectorValidation.norm(of: MLMultiArray(shaped), dimension: dimension, id: "photo")
                }
            }
        }
    }

    func testSameCountMatrixShapeIsRejected() {
        for dimension in dimensions {
            let shaped = vector(dimension, shape: [2, dimension / 2])
            assertError(.unsupportedLayout) {
                try SimilarityVectorValidation.norm(of: shaped, dimension: dimension)
            }
            assertError(.unsupportedLayout) {
                try SimilarityVectorValidation.norm(of: MLMultiArray(shaped), dimension: dimension, id: "photo")
            }
        }
    }

    func testNoncontiguousFloat32AndOtherScalarTypesAreRejected() throws {
        for dimension in dimensions {
            let backing = UnsafeMutablePointer<Float32>.allocate(capacity: dimension * 2)
            backing.initialize(repeating: 1, count: dimension * 2)
            let strided = try MLMultiArray(dataPointer: backing, shape: [1, NSNumber(value: dimension)],
                                          dataType: .float32, strides: [NSNumber(value: dimension * 2), 2],
                                          deallocator: { $0.assumingMemoryBound(to: Float32.self).deallocate() })
            assertError(.unsupportedLayout) {
                try SimilarityVectorValidation.norm(of: strided, dimension: dimension, id: "photo")
            }
            for type: MLMultiArrayDataType in [.double, .int32, .float16] {
                let array = try MLMultiArray(shape: [1, NSNumber(value: dimension)], dataType: type)
                assertError(.invalidEmbedding(id: "photo", expected: dimension, actual: dimension)) {
                    try SimilarityVectorValidation.norm(of: array, dimension: dimension, id: "photo")
                }
            }
        }
    }

    func testBlankNearZeroAndNonfiniteValuesAreRejected() {
        for dimension in dimensions {
            for (value, error): (Float32, SimilaritySearchError) in [
                (0, .zeroNormVector), (1e-20, .zeroNormVector),
                (.nan, .nonFiniteVector), (.infinity, .nonFiniteVector), (-.infinity, .nonFiniteVector)
            ] {
                var shaped = vector(dimension, value: 0)
                shaped[scalarAt: 0, dimension - 1] = value
                assertError(error) { try SimilarityVectorValidation.norm(of: shaped, dimension: dimension) }
                assertError(error) {
                    try SimilarityVectorValidation.norm(of: MLMultiArray(shaped), dimension: dimension, id: "photo")
                }
            }
        }
    }

    func testLargeFiniteVectorsKeepFiniteNonzeroNorms() throws {
        for dimension in dimensions {
            let shaped = vector(dimension, value: .greatestFiniteMagnitude)
            let norm = try SimilarityVectorValidation.norm(of: shaped, dimension: dimension)
            XCTAssertTrue(norm.isFinite)
            let normalized = Float16(Double(Float32.greatestFiniteMagnitude) / norm)
            XCTAssertTrue(normalized.isFinite)
            XCTAssertGreaterThan(normalized, 0)
            XCTAssertEqual(try SimilarityVectorValidation.norm(of: MLMultiArray(shaped), dimension: dimension, id: "photo"), norm)
        }
    }

    func testEmptyGPUIndexStillRejectsInvalidQueriesAndMutations() throws {
        guard let gpu = GPUSimilaritySearch(embeddingDimension: 768) else {
            throw XCTSkip("Metal unavailable; device-independent validation is covered separately")
        }
        XCTAssertTrue(try gpu.search(queryEmbedding: vector(768)).isEmpty)
        assertError(.zeroNormVector) { try gpu.search(queryEmbedding: vector(768, value: 0)) }
        assertError(.nonFiniteVector) { try gpu.search(queryEmbedding: vector(768, value: .nan)) }
        assertError(.zeroNormVector) { try gpu.buildIndex(from: ["blank": MLMultiArray(vector(768, value: 0))]) }
        assertError(.nonFiniteVector) { try gpu.addEmbeddings(["nan": MLMultiArray(vector(768, value: .nan))]) }
        XCTAssertEqual(gpu.count, 0)
        XCTAssertTrue(gpu.ids.isEmpty)
    }
}
