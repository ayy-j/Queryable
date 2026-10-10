import XCTest
@testable import Queryable

/// Parity gates for issue #5: golden-vector and tokenizer checks that every
/// Core ML release must pass before promotion (issues #16, #18, #21).
final class ParityMetricsTests: XCTestCase {

    // MARK: - Metrics

    func testCosineSimilarityIdenticalVectorsIsOne() throws {
        let v: [Float] = [0.5, -0.25, 0.75, 0.1]
        XCTAssertEqual(try ParityMetrics.cosineSimilarity(v, v), 1.0, accuracy: 1e-6)
        XCTAssertEqual(try ParityMetrics.cosineDistance(v, v), 0.0, accuracy: 1e-6)
    }

    func testCosineSimilarityOrthogonalVectorsIsZero() throws {
        XCTAssertEqual(
            try ParityMetrics.cosineSimilarity([1, 0], [0, 1]),
            0.0,
            accuracy: 1e-6
        )
    }

    func testCosineSimilarityRejectsDimensionMismatch() {
        XCTAssertThrowsError(try ParityMetrics.cosineSimilarity([1, 0], [1]))
    }

    func testCosineSimilarityRejectsNonFiniteValues() {
        XCTAssertThrowsError(try ParityMetrics.cosineSimilarity([1, .nan], [1, 0]))
        XCTAssertThrowsError(try ParityMetrics.cosineSimilarity([1, .infinity], [1, 0]))
    }

    func testL2NormOfNormalizedVectorIsOne() throws {
        let norm = 1.0 / sqrt(3.0)
        let v = [Float(norm), Float(norm), Float(norm)]
        XCTAssertEqual(try ParityMetrics.l2Norm(v), 1.0, accuracy: 1e-6)
    }

    func testSpearmanCorrelationIdenticalOrderIsOne() throws {
        let scores: [Float] = [0.9, 0.5, 0.7, 0.1]
        XCTAssertEqual(try ParityMetrics.spearmanRankCorrelation(scores, scores), 1.0, accuracy: 1e-5)
    }

    func testSpearmanCorrelationReversedOrderIsNegativeOne() throws {
        let a: [Float] = [1, 2, 3, 4]
        let b: [Float] = [4, 3, 2, 1]
        XCTAssertEqual(try ParityMetrics.spearmanRankCorrelation(a, b), -1.0, accuracy: 1e-5)
    }

    func testValidateVectorRejectsWrongDimension() {
        XCTAssertThrowsError(
            try ParityMetrics.validateVector([1, 2, 3], expectedDimension: 512)
        )
    }

    // MARK: - Gate

    func testGatePassesIdenticalVectors() {
        // Towers L2-normalize inside, so gate inputs must be unit-norm.
        let v = [Float](repeating: 0.5, count: 4)
        let result = ParityGate.evaluate(reference: v, candidate: v, tolerances: .fp32)
        XCTAssertTrue(result.passed)
        XCTAssertTrue(result.failures.isEmpty)
    }

    func testGateFailsBeyondTolerance() {
        let reference = [Float](repeating: 1.0, count: 4)
        let candidate: [Float] = [1, 0, 0, 0]
        let result = ParityGate.evaluate(reference: reference, candidate: candidate, tolerances: .fp32)
        XCTAssertFalse(result.passed)
        XCTAssertFalse(result.failures.isEmpty)
    }

    func testGateFailsDimensionMismatchLoudly() {
        let result = ParityGate.evaluate(
            reference: [1, 2, 3],
            candidate: [1, 2],
            tolerances: .fp32
        )
        XCTAssertFalse(result.passed)
    }

    func testGateChecksRankCorrelationWhenScoresProvided() {
        let dim = 4
        let unit = [Float](repeating: 0.5, count: dim)
        let refScores: [Float] = [0.9, 0.7, 0.5, 0.1]
        let candScores: [Float] = [0.1, 0.5, 0.7, 0.9]
        let result = ParityGate.evaluate(
            reference: unit,
            candidate: unit,
            tolerances: .fp32,
            referenceScores: refScores,
            candidateScores: candScores
        )
        XCTAssertFalse(result.passed)
        XCTAssertTrue(result.failures.contains { $0.contains("rank-correlation") })
    }

    // MARK: - Tokenizer golden fixtures

    private var fixtureURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("ParityFixtures/clip-bpe-tokens-v1.json")
    }

    func testTokenFixtureFileLoadsAndMatchesSpecContract() throws {
        let file = try ParityFixtureLoader.loadTokenFixtures(from: fixtureURL)
        XCTAssertEqual(file.manifest.fixtureVersion, "clip-bpe-tokens-v1")
        XCTAssertEqual(file.manifest.modelID, "mobileclip-s2")
        XCTAssertEqual(file.manifest.embeddingDimension, 512)
        XCTAssertEqual(file.manifest.contextLength, 77)
        XCTAssertFalse(file.cases.isEmpty)
    }

    func testTokenFixturesCoverRequiredEdgeCases() throws {
        let file = try ParityFixtureLoader.loadTokenFixtures(from: fixtureURL)
        let ids = Set(file.cases.map(\.id))
        for required in ["empty", "punctuation", "unicode-diacritics", "emoji",
                         "whitespace", "max-length-truncation"] {
            XCTAssertTrue(ids.contains(required), "missing fixture case: \(required)")
        }
    }

    func testSwiftTokenizerMatchesGoldenTokenIDs() throws {
        let file = try ParityFixtureLoader.loadTokenFixtures(from: fixtureURL)
        let resources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Queryable/CoreMLModels")
        let tokenizer = try BPETokenizer(
            mergesAt: resources.appendingPathComponent("merges.txt"),
            vocabularyAt: resources.appendingPathComponent("vocab.json")
        )
        for fixture in file.cases {
            let (_, ids) = tokenizer.tokenize(input: fixture.prompt, minCount: fixture.paddedLength)
            XCTAssertTrue(
                ParityFixtureLoader.checkTokenIDs(actual: ids, expected: fixture.expectedTokenIDs),
                "token mismatch for case '\(fixture.id)'"
            )
        }
    }

    func testPaddedFixturesMatchContextLength() throws {
        let file = try ParityFixtureLoader.loadTokenFixtures(from: fixtureURL)
        for fixture in file.cases where fixture.paddedLength != nil {
            if fixture.id == "max-length-truncation" {
                // Raw regression fixtures remain unbounded. TextEncoder uses
                // the separate fixed-context API, preserving EOS at slot 76.
                XCTAssertGreaterThan(fixture.expectedTokenIDs.count, fixture.paddedLength!)
            } else {
                XCTAssertEqual(fixture.expectedTokenIDs.count, fixture.paddedLength)
            }
        }
    }

    // MARK: - Fixture schemas

    func testVectorFixtureFileRoundTrips() throws {
        let manifest = ParityFixtureManifest(
            fixtureVersion: "vectors-v1",
            modelID: "mobileclip-s2",
            modelRevision: "mobileclip-s2-v1",
            tokenizerAssets: ["vocab.json", "merges.txt"],
            preprocessingFingerprint: "ci-lanczos-argb-256-v1",
            embeddingDimension: 4,
            contextLength: 77,
            createdAt: "2026-10-08"
        )
        let file = VectorFixtureFile(
            manifest: manifest,
            precision: "fp32",
            vectors: [VectorFixture(id: "probe-1", kind: "text", source: "hello", vector: [0.5, 0.5, 0.5, 0.5])]
        )
        let data = try JSONEncoder().encode(file)
        let decoded = try JSONDecoder().decode(VectorFixtureFile.self, from: data)
        XCTAssertEqual(decoded, file)
    }

    func testParityReportRoundTrips() throws {
        let report = ParityReport(
            fixtureVersion: "clip-bpe-tokens-v1",
            modelID: "mobileclip-s2",
            modelRevision: "mobileclip-s2-v1",
            tolerancesID: "fp32",
            maxCosineDistance: 1e-4,
            minRankCorrelation: 0.999,
            results: [ParityCaseResult(
                fixtureID: "probe-1",
                cosineDistance: 1e-5,
                rankCorrelation: 1.0,
                passed: true,
                failures: []
            )],
            passed: true
        )
        let data = try JSONEncoder().encode(report)
        let decoded = try JSONDecoder().decode(ParityReport.self, from: data)
        XCTAssertEqual(decoded, report)
    }
}
