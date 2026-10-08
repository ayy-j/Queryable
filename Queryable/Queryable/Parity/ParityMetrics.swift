import Foundation

/// Numerical parity metrics for issue #5.
///
/// Pure-Foundation implementation (no CoreML/UIKit) so the same code runs in
/// XCTest and in the SwiftPM substitute harness described in
/// `docs/test-infrastructure-plan.md`.
public enum ParityMetrics {

    public enum ParityError: Error, Equatable {
        case dimensionMismatch(expected: Int, actual: Int)
        case emptyVector
        case nonFiniteValue
    }

    /// L2 norm of a vector. Throws on empty input or non-finite values.
    public static func l2Norm(_ vector: [Float]) throws -> Float {
        guard !vector.isEmpty else { throw ParityError.emptyVector }
        var sum: Double = 0
        for v in vector {
            guard v.isFinite else { throw ParityError.nonFiniteValue }
            sum += Double(v) * Double(v)
        }
        return Float(sqrt(sum))
    }

    /// Cosine similarity in [-1, 1]. Throws on dimension mismatch, empty
    /// input, non-finite values, or zero-norm vectors.
    public static func cosineSimilarity(_ a: [Float], _ b: [Float]) throws -> Float {
        guard a.count == b.count else {
            throw ParityError.dimensionMismatch(expected: a.count, actual: b.count)
        }
        guard !a.isEmpty else { throw ParityError.emptyVector }
        var dot: Double = 0
        var na: Double = 0
        var nb: Double = 0
        for i in 0..<a.count {
            let x = a[i], y = b[i]
            guard x.isFinite && y.isFinite else { throw ParityError.nonFiniteValue }
            dot += Double(x) * Double(y)
            na += Double(x) * Double(x)
            nb += Double(y) * Double(y)
        }
        guard na > 0 && nb > 0 else { throw ParityError.emptyVector }
        return Float(dot / (sqrt(na) * sqrt(nb)))
    }

    /// Cosine distance (1 - similarity). Small values mean agreement.
    public static func cosineDistance(_ a: [Float], _ b: [Float]) throws -> Float {
        1 - (try cosineSimilarity(a, b))
    }

    /// Rank positions of each index when sorting scores descending.
    /// Ties are broken by index so the result is deterministic.
    public static func ranks(of scores: [Float]) -> [Int] {
        let order = scores.indices.sorted {
            if scores[$0] != scores[$1] { return scores[$0] > scores[$1] }
            return $0 < $1
        }
        var rank = [Int](repeating: 0, count: scores.count)
        for (position, index) in order.enumerated() {
            rank[index] = position
        }
        return rank
    }

    /// Spearman rank correlation in [-1, 1] between two score lists.
    /// Used to verify that a converted tower preserves retrieval ordering
    /// even when absolute cosine values shift slightly.
    public static func spearmanRankCorrelation(_ a: [Float], _ b: [Float]) throws -> Float {
        guard a.count == b.count else {
            throw ParityError.dimensionMismatch(expected: a.count, actual: b.count)
        }
        guard a.count >= 2 else { throw ParityError.emptyVector }
        for v in a + b { guard v.isFinite else { throw ParityError.nonFiniteValue } }
        let ra = ranks(of: a).map(Double.init)
        let rb = ranks(of: b).map(Double.init)
        let n = Double(a.count)
        let meanA = ra.reduce(0, +) / n
        let meanB = rb.reduce(0, +) / n
        var cov: Double = 0
        var va: Double = 0
        var vb: Double = 0
        for i in 0..<a.count {
            let da = ra[i] - meanA
            let db = rb[i] - meanB
            cov += da * db
            va += da * da
            vb += db * db
        }
        guard va > 0 && vb > 0 else { return 1.0 }
        return Float(cov / (sqrt(va) * sqrt(vb)))
    }

    /// Validates a single embedding: expected dimension, all finite,
    /// and (for normalized towers) L2 norm within `normTolerance` of 1.
    public static func validateVector(
        _ vector: [Float],
        expectedDimension: Int,
        normTolerance: Float? = nil
    ) throws {
        guard vector.count == expectedDimension else {
            throw ParityError.dimensionMismatch(expected: expectedDimension, actual: vector.count)
        }
        for v in vector { guard v.isFinite else { throw ParityError.nonFiniteValue } }
        if let tolerance = normTolerance {
            let norm = try l2Norm(vector)
            guard abs(norm - 1) <= tolerance else { throw ParityError.nonFiniteValue }
        }
    }
}

/// Pass/fail tolerances for one precision/backend combination.
///
/// Thresholds are declared before any release candidate is measured
/// (issue #5 acceptance criteria); they must not be tuned after seeing results.
public struct ParityTolerances: Equatable, Sendable {
    public let maxCosineDistance: Float
    public let minRankCorrelation: Float
    public let normTolerance: Float

    public init(maxCosineDistance: Float, minRankCorrelation: Float, normTolerance: Float) {
        self.maxCosineDistance = maxCosineDistance
        self.minRankCorrelation = minRankCorrelation
        self.normTolerance = normTolerance
    }

    /// FP32 reference vs FP32 Core ML on the same machine.
    public static let fp32 = ParityTolerances(
        maxCosineDistance: 1e-4,
        minRankCorrelation: 0.999,
        normTolerance: 1e-3
    )

    /// FP16 Core ML tower vs FP32 PyTorch reference. Allows for the
    /// image-encoder precision error noted in README.md.
    public static let fp16 = ParityTolerances(
        maxCosineDistance: 5e-3,
        minRankCorrelation: 0.99,
        normTolerance: 1e-2
    )
}

/// Outcome of gating one candidate vector pair against tolerances.
public struct ParityGateResult: Equatable, Sendable {
    public let cosineDistance: Float
    public let rankCorrelation: Float?
    public let passed: Bool
    public let failures: [String]

    public init(cosineDistance: Float, rankCorrelation: Float?, passed: Bool, failures: [String]) {
        self.cosineDistance = cosineDistance
        self.rankCorrelation = rankCorrelation
        self.passed = passed
        self.failures = failures
    }
}

public enum ParityGate {
    /// Gates a single reference/candidate vector pair.
    /// - Parameters:
    ///   - reference: Pinned PyTorch reference vector.
    ///   - candidate: Swift/Core ML output vector.
    ///   - referenceScores: Optional retrieval scores over a shared probe set
    ///     from the reference tower (for rank correlation).
    ///   - candidateScores: Matching scores from the candidate tower.
    public static func evaluate(
        reference: [Float],
        candidate: [Float],
        tolerances: ParityTolerances,
        referenceScores: [Float]? = nil,
        candidateScores: [Float]? = nil
    ) -> ParityGateResult {
        var failures = [String]()
        let distance: Float
        do {
            try ParityMetrics.validateVector(reference, expectedDimension: reference.count)
            try ParityMetrics.validateVector(candidate, expectedDimension: reference.count)
            distance = try ParityMetrics.cosineDistance(reference, candidate)
        } catch {
            return ParityGateResult(
                cosineDistance: .nan,
                rankCorrelation: nil,
                passed: false,
                failures: ["vector-validation: \(error)"]
            )
        }
        if distance > tolerances.maxCosineDistance {
            failures.append("cosine-distance \(distance) exceeds \(tolerances.maxCosineDistance)")
        }
        var correlation: Float?
        if let refScores = referenceScores, let candScores = candidateScores {
            do {
                correlation = try ParityMetrics.spearmanRankCorrelation(refScores, candScores)
                if correlation! < tolerances.minRankCorrelation {
                    failures.append("rank-correlation \(correlation!) below \(tolerances.minRankCorrelation)")
                }
            } catch {
                failures.append("rank-correlation: \(error)")
            }
        }
        do {
            let norm = try ParityMetrics.l2Norm(candidate)
            if abs(norm - 1) > tolerances.normTolerance {
                failures.append("candidate L2 norm \(norm) outside \(tolerances.normTolerance) of 1")
            }
        } catch {
            failures.append("candidate-norm: \(error)")
        }
        return ParityGateResult(
            cosineDistance: distance,
            rankCorrelation: correlation,
            passed: failures.isEmpty,
            failures: failures
        )
    }
}
