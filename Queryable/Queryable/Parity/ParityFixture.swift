import Foundation

/// Codable fixture/report schemas for issue #5.
///
/// Every fixture file carries its own provenance so a parity report can
/// identify fixture version, model/checkpoint, tokenizer assets,
/// preprocessing, tolerances, and runtime — never conversation history.
public struct ParityFixtureManifest: Codable, Equatable, Sendable {
    public let fixtureVersion: String
    public let modelID: String
    public let modelRevision: String
    public let checkpointHash: String?
    public let tokenizerAssets: [String]
    public let preprocessingFingerprint: String
    public let embeddingDimension: Int
    public let contextLength: Int
    public let createdAt: String

    public init(
        fixtureVersion: String,
        modelID: String,
        modelRevision: String,
        checkpointHash: String? = nil,
        tokenizerAssets: [String],
        preprocessingFingerprint: String,
        embeddingDimension: Int,
        contextLength: Int,
        createdAt: String
    ) {
        self.fixtureVersion = fixtureVersion
        self.modelID = modelID
        self.modelRevision = modelRevision
        self.checkpointHash = checkpointHash
        self.tokenizerAssets = tokenizerAssets
        self.preprocessingFingerprint = preprocessingFingerprint
        self.embeddingDimension = embeddingDimension
        self.contextLength = contextLength
        self.createdAt = createdAt
    }
}

/// One tokenizer golden case. `expectedTokenIDs` must match the pinned
/// reference exactly — no fuzzy matching.
public struct TokenFixture: Codable, Equatable, Sendable {
    public let id: String
    public let prompt: String
    public let paddedLength: Int?
    public let expectedTokenIDs: [Int]

    public init(id: String, prompt: String, paddedLength: Int? = nil, expectedTokenIDs: [Int]) {
        self.id = id
        self.prompt = prompt
        self.paddedLength = paddedLength
        self.expectedTokenIDs = expectedTokenIDs
    }
}

public struct TokenFixtureFile: Codable, Equatable, Sendable {
    public let manifest: ParityFixtureManifest
    public let cases: [TokenFixture]

    public init(manifest: ParityFixtureManifest, cases: [TokenFixture]) {
        self.manifest = manifest
        self.cases = cases
    }
}

/// One reference vector produced by the pinned PyTorch runner
/// (`tools/generate_parity_reference.py`).
public struct VectorFixture: Codable, Equatable, Sendable {
    public let id: String
    public let kind: String
    public let source: String
    public let vector: [Float]

    public init(id: String, kind: String, source: String, vector: [Float]) {
        self.id = id
        self.kind = kind
        self.source = source
        self.vector = vector
    }
}

public struct VectorFixtureFile: Codable, Equatable, Sendable {
    public let manifest: ParityFixtureManifest
    public let precision: String
    public let vectors: [VectorFixture]

    public init(manifest: ParityFixtureManifest, precision: String, vectors: [VectorFixture]) {
        self.manifest = manifest
        self.precision = precision
        self.vectors = vectors
    }
}

/// Per-vector gate outcome inside a parity report.
public struct ParityCaseResult: Codable, Equatable, Sendable {
    public let fixtureID: String
    public let cosineDistance: Float?
    public let rankCorrelation: Float?
    public let passed: Bool
    public let failures: [String]

    public init(fixtureID: String, cosineDistance: Float?, rankCorrelation: Float?, passed: Bool, failures: [String]) {
        self.fixtureID = fixtureID
        self.cosineDistance = cosineDistance
        self.rankCorrelation = rankCorrelation
        self.passed = passed
        self.failures = failures
    }
}

/// Reproducible parity report. Stored next to the release candidate;
/// public-safe (no private photos, prompts, or library metadata).
public struct ParityReport: Codable, Equatable, Sendable {
    public let reportVersion: String
    public let fixtureVersion: String
    public let modelID: String
    public let modelRevision: String
    public let artifactChecksums: [String: String]
    public let tolerancesID: String
    public let maxCosineDistance: Float
    public let minRankCorrelation: Float
    public let runtime: [String: String]
    public let results: [ParityCaseResult]
    public let passed: Bool

    public init(
        reportVersion: String = "parity-report-v1",
        fixtureVersion: String,
        modelID: String,
        modelRevision: String,
        artifactChecksums: [String: String] = [:],
        tolerancesID: String,
        maxCosineDistance: Float,
        minRankCorrelation: Float,
        runtime: [String: String] = [:],
        results: [ParityCaseResult],
        passed: Bool
    ) {
        self.reportVersion = reportVersion
        self.fixtureVersion = fixtureVersion
        self.modelID = modelID
        self.modelRevision = modelRevision
        self.artifactChecksums = artifactChecksums
        self.tolerancesID = tolerancesID
        self.maxCosineDistance = maxCosineDistance
        self.minRankCorrelation = minRankCorrelation
        self.runtime = runtime
        self.results = results
        self.passed = passed
    }
}

public enum ParityFixtureLoader {
    public static func loadTokenFixtures(from url: URL) throws -> TokenFixtureFile {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(TokenFixtureFile.self, from: data)
    }

    public static func loadVectorFixtures(from url: URL) throws -> VectorFixtureFile {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(VectorFixtureFile.self, from: data)
    }

    /// Exact token-ID comparison. Any mismatch fails loudly — tokenizers
    /// must never silently fall back (issue #17 requires explicit failure).
    public static func checkTokenIDs(actual: [Int], expected: [Int]) -> Bool {
        actual == expected
    }
}

/// Evaluates already captured outputs; inference and reference export must be
/// performed separately with the same public probes and pinned model contract.
public enum ParityFixtureGate {
    public enum ValidationError: Error, Equatable {
        case invalidManifest(String)
        case invalidFixtures(String)
        case candidateIDsMismatch
    }

    public static func evaluate(
        reference: VectorFixtureFile,
        candidates: [String: [Float]],
        tolerancesID: String,
        artifactChecksums: [String: String],
        runtime: [String: String]
    ) throws -> ParityReport {
        let tolerances: ParityTolerances
        switch tolerancesID {
        case "fp32": tolerances = .fp32
        case "fp16": tolerances = .fp16
        default: throw ValidationError.invalidManifest("unknown tolerance ID")
        }
        let manifest = reference.manifest
        guard reference.precision == "fp32", manifest.embeddingDimension > 0,
              manifest.contextLength > 0,
              !manifest.fixtureVersion.isEmpty, !manifest.modelID.isEmpty,
              !manifest.modelRevision.isEmpty,
              !manifest.preprocessingFingerprint.isEmpty,
              !manifest.tokenizerAssets.isEmpty,
              let checkpoint = manifest.checkpointHash,
              checkpoint.count == 64,
              checkpoint.allSatisfy({ $0.isHexDigit }) else {
            throw ValidationError.invalidManifest("pinned FP32 reference contract required")
        }
        guard !artifactChecksums.isEmpty,
              artifactChecksums.values.allSatisfy({ $0.count == 64 && $0.allSatisfy(\.isHexDigit) }),
              ["device", "backend"].allSatisfy({ !(runtime[$0] ?? "").isEmpty }) else {
            throw ValidationError.invalidManifest("artifact SHA-256 checksums and device/backend required")
        }
        let ids = reference.vectors.map(\.id)
        guard !ids.isEmpty, ids.allSatisfy({ !$0.isEmpty }), Set(ids).count == ids.count,
              reference.vectors.allSatisfy({ ["image", "text"].contains($0.kind) }) else {
            throw ValidationError.invalidFixtures("nonempty unique IDs and image/text kinds required")
        }
        guard Set(candidates.keys) == Set(ids) else {
            throw ValidationError.candidateIDsMismatch
        }
        // Release parity requires cross-modal retrieval ordering, not just a
        // collection of individually matching vectors.
        let images = reference.vectors.filter { $0.kind == "image" }
        let texts = reference.vectors.filter { $0.kind == "text" }
        guard images.count >= 2, texts.count >= 2 else {
            throw ValidationError.invalidFixtures("at least two image and two text probes required")
        }
        for fixture in reference.vectors {
            try ParityMetrics.validateVector(fixture.vector,
                expectedDimension: manifest.embeddingDimension,
                normTolerance: ParityTolerances.fp32.normTolerance)
        }
        let results = try reference.vectors.map { fixture -> ParityCaseResult in
            let candidate = candidates[fixture.id]!
            let probes = fixture.kind == "text" ? images : texts
            // Invalid candidate scores are passed into the numerical gate so
            // they produce a failed, serializable case rather than aborting report creation.
            let refScores = try probes.map { try ParityMetrics.cosineSimilarity(fixture.vector, $0.vector) }
            let candScores = probes.map {
                (try? ParityMetrics.cosineSimilarity(candidate, candidates[$0.id]!)) ?? .nan
            }
            let outcome = ParityGate.evaluate(reference: fixture.vector,
                candidate: candidate, tolerances: tolerances,
                referenceScores: refScores, candidateScores: candScores)
            return ParityCaseResult(fixtureID: fixture.id,
                cosineDistance: outcome.cosineDistance.isFinite ? outcome.cosineDistance : nil,
                rankCorrelation: outcome.rankCorrelation,
                passed: outcome.passed, failures: outcome.failures)
        }
        return ParityReport(fixtureVersion: manifest.fixtureVersion,
            modelID: manifest.modelID, modelRevision: manifest.modelRevision,
            artifactChecksums: artifactChecksums, tolerancesID: tolerancesID,
            maxCosineDistance: tolerances.maxCosineDistance,
            minRankCorrelation: tolerances.minRankCorrelation,
            runtime: runtime, results: results, passed: results.allSatisfy(\.passed))
    }
}
