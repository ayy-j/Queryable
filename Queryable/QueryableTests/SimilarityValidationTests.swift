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

    // MPSGraph execution crashes in Apple's simulator runtime on some hosts.
    // Keep state/validation coverage there and execute score checks on a native Mac or device.
    private func requireGPUExecution() throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("MPSGraph execution requires a physical device or native macOS test runner")
        #endif
    }

    private func makeGPU(_ dimension: Int) throws -> GPUSimilaritySearch {
        guard let gpu = GPUSimilaritySearch(embeddingDimension: dimension) else {
            throw XCTSkip("Metal unavailable")
        }
        return gpu
    }

    /// Dense vectors with a known component along the all-ones query and an
    /// orthogonal alternating component; every required dimension is even.
    private func knownVector(_ dimension: Int, along: Float32, across: Float32) -> MLShapedArray<Float32> {
        MLShapedArray(scalars: (0..<dimension).map { along + ($0.isMultiple(of: 2) ? across : -across) },
                      shape: [1, dimension])
    }

    private func cpuCosine(_ lhs: [Float32], _ rhs: [Float32]) -> Double {
        let dot = zip(lhs, rhs).reduce(0.0) { $0 + Double($1.0) * Double($1.1) }
        let lhsNorm = sqrt(lhs.reduce(0.0) { $0 + Double($1) * Double($1) })
        let rhsNorm = sqrt(rhs.reduce(0.0) { $0 + Double($1) * Double($1) })
        return dot / (lhsNorm * rhsNorm)
    }

    private func assertScores(_ gpu: GPUSimilaritySearch,
                              expected: [String: MLShapedArray<Float32>],
                              query: MLShapedArray<Float32>,
                              file: StaticString = #filePath, line: UInt = #line) throws {
        // Declared before execution: allow 0.002 absolute error for Float16
        // inputs and Float32 matmul. Scores are separated by much more than this.
        let absoluteTolerance = 2e-3
        let scores = try gpu.search(queryEmbedding: query)
        let reference = expected.mapValues { cpuCosine($0.scalars, query.scalars) }
        XCTAssertEqual(Set(scores.keys), Set(expected.keys), file: file, line: line)
        for (id, expectedScore) in reference {
            XCTAssertEqual(Double(try XCTUnwrap(scores[id], file: file, line: line)), expectedScore,
                           accuracy: absoluteTolerance, "ID: \(id)", file: file, line: line)
        }
        XCTAssertEqual(scores.keys.sorted { scores[$0]! > scores[$1]! },
                       reference.keys.sorted { reference[$0]! > reference[$1]! }, file: file, line: line)
    }

    func testUpsertsKeepUniqueIDsAndExistingRowPositionsWithoutExecutingGraph() throws {
        for dimension in dimensions {
            let gpu = try makeGPU(dimension)
            try gpu.buildIndex(from: ["replace": MLMultiArray(vector(dimension)),
                                      "untouched": MLMultiArray(vector(dimension))])
            let originalIDs = gpu.ids
            try gpu.addEmbeddings(["replace": MLMultiArray(vector(dimension, value: -1))])
            XCTAssertEqual(gpu.ids, originalIDs)
            XCTAssertEqual(gpu.count, 2)
            try gpu.addEmbeddings(["replace": MLMultiArray(vector(dimension)),
                                   "added": MLMultiArray(vector(dimension))])
            XCTAssertEqual(Array(gpu.ids.prefix(2)), originalIDs)
            XCTAssertEqual(gpu.ids.last, "added")
            for _ in 0..<3 {
                try gpu.addEmbeddings(["replace": MLMultiArray(vector(dimension))])
            }
            XCTAssertEqual(gpu.count, 3)
            XCTAssertEqual(Set(gpu.ids).count, 3)
            let beforeInvalid = gpu.ids
            assertError(.zeroNormVector) {
                try gpu.addEmbeddings(["replace": MLMultiArray(vector(dimension, value: -1)),
                                       "invalid": MLMultiArray(vector(dimension, value: 0))])
            }
            XCTAssertEqual(gpu.ids, beforeInvalid)
            gpu.removeEmbeddings(["replace", "missing"])
            XCTAssertEqual(Set(gpu.ids), ["untouched", "added"])
            try gpu.buildIndex(from: [:])
            XCTAssertEqual(gpu.count, 0)
            XCTAssertTrue(try gpu.search(queryEmbedding: vector(dimension)).isEmpty)
        }
    }

    func testGPUUpsertsMatchIndependentCPUCosineAtEveryRequiredDimension() throws {
        try requireGPUExecution()
        for dimension in dimensions {
            let gpu = try makeGPU(dimension)
            let query = vector(dimension)
            var expected = ["replace": knownVector(dimension, along: -0.8, across: 0.6),
                            "untouched": knownVector(dimension, along: 0.6, across: 0.8)]
            try gpu.buildIndex(from: expected.mapValues { MLMultiArray($0) })
            try assertScores(gpu, expected: expected, query: query)
            let originalIDs = gpu.ids

            expected["replace"] = knownVector(dimension, along: 0.9, across: 0.3)
            try gpu.addEmbeddings(["replace": MLMultiArray(expected["replace"]!)])
            XCTAssertEqual(gpu.ids, originalIDs)
            XCTAssertEqual(gpu.count, 2)
            try assertScores(gpu, expected: expected, query: query)

            expected["replace"] = knownVector(dimension, along: -0.3, across: 0.9)
            expected["added"] = knownVector(dimension, along: 0.8, across: 0.6)
            try gpu.addEmbeddings(["replace": MLMultiArray(expected["replace"]!),
                                   "added": MLMultiArray(expected["added"]!)])
            XCTAssertEqual(Array(gpu.ids.prefix(2)), originalIDs)
            XCTAssertEqual(gpu.count, 3)
            try assertScores(gpu, expected: expected, query: query)

            for along: Float32 in [-0.9, 0.2, 0.95] {
                expected["replace"] = knownVector(dimension, along: along, across: 0.4)
                try gpu.addEmbeddings(["replace": MLMultiArray(expected["replace"]!)])
                XCTAssertEqual(gpu.count, 3)
                XCTAssertEqual(Set(gpu.ids).count, 3)
                try assertScores(gpu, expected: expected, query: query)
            }
        }
    }

    func testGPUInvalidBatchesPreservePreviousUsableIndex() throws {
        try requireGPUExecution()
        for dimension in dimensions {
            let gpu = try makeGPU(dimension)
            let query = vector(dimension)
            let expected = ["existing": knownVector(dimension, along: 0.6, across: 0.8)]
            try gpu.buildIndex(from: expected.mapValues { MLMultiArray($0) })
            let originalIDs = gpu.ids
            let invalidVectors: [(MLShapedArray<Float32>, SimilaritySearchError)] = [
                (vector(dimension, value: 0), .zeroNormVector),
                (vector(dimension, value: .nan), .nonFiniteVector),
                (vector(dimension - 1), .invalidEmbedding(id: "invalid", expected: dimension, actual: dimension - 1)),
                (vector(dimension, shape: [2, dimension / 2]), .unsupportedLayout)
            ]
            for (invalid, error) in invalidVectors {
                assertError(error) {
                    try gpu.addEmbeddings(["existing": MLMultiArray(vector(dimension, value: -1)),
                                           "new": MLMultiArray(vector(dimension)),
                                           "invalid": MLMultiArray(invalid)])
                }
                XCTAssertEqual(gpu.ids, originalIDs)
                XCTAssertEqual(gpu.count, 1)
                try assertScores(gpu, expected: expected, query: query)
                assertError(error) { try gpu.buildIndex(from: ["invalid": MLMultiArray(invalid)]) }
                try assertScores(gpu, expected: expected, query: query)
            }
            try gpu.addEmbeddings([:])
            try assertScores(gpu, expected: expected, query: query)
        }
    }

    func testGPURemovalRebuildAndEmptyTransitionsMatchCPU() throws {
        try requireGPUExecution()
        for dimension in dimensions {
            let gpu = try makeGPU(dimension)
            let query = vector(dimension)
            var expected = ["a": knownVector(dimension, along: 0.8, across: 0.6),
                            "b": knownVector(dimension, along: -0.6, across: 0.8),
                            "c": knownVector(dimension, along: 0.3, across: 0.9)]
            try gpu.addEmbeddings(expected.mapValues { MLMultiArray($0) })
            try assertScores(gpu, expected: expected, query: query)
            let retainedOrder = gpu.ids.filter { $0 != "b" }
            gpu.removeEmbeddings(["b", "missing"])
            expected.removeValue(forKey: "b")
            XCTAssertEqual(gpu.ids, retainedOrder)
            try assertScores(gpu, expected: expected, query: query)
            gpu.removeEmbeddings(["missing"])
            try assertScores(gpu, expected: expected, query: query)
            gpu.removeEmbeddings(Set(expected.keys))
            XCTAssertEqual(gpu.count, 0)
            XCTAssertTrue(try gpu.search(queryEmbedding: query).isEmpty)
            expected = ["rebuilt": knownVector(dimension, along: -0.8, across: 0.6)]
            try gpu.buildIndex(from: expected.mapValues { MLMultiArray($0) })
            try assertScores(gpu, expected: expected, query: query)
            try gpu.buildIndex(from: [:])
            XCTAssertTrue(gpu.ids.isEmpty)
            XCTAssertTrue(try gpu.search(queryEmbedding: query).isEmpty)
            try gpu.addEmbeddings(expected.mapValues { MLMultiArray($0) })
            try assertScores(gpu, expected: expected, query: query)
        }
    }

}
