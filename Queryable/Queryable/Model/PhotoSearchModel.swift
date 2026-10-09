//
//  PhotoSearcherModel.swift
//  TestEncoder
//
//  Created by Ke Fang on 2022/12/08.
//
import UIKit
import CoreML
import Foundation
import Accelerate

struct PhotoSearcherModel {
    private var texEncoder: TextEncoder?
    private(set) var spec: EmbeddingModelSpec = .mobileCLIP2S4

    mutating func load_text_encoder(resourcesAt resourceURL: URL, spec: EmbeddingModelSpec) throws {
        // TODO: move the pipeline creation to background task because it's heavy

        let encoder = try TextEncoder(resourcesAt: resourceURL, spec: spec)
        texEncoder = encoder
        self.spec = spec
    }
    
    func text_embedding(prompt: String) -> MLShapedArray<Float32> {
        let emb = try! texEncoder?.computeTextEmbedding(prompt: prompt)
        return emb!
    }
    
    func cosine_similarity(A: MLShapedArray<Float32>, B: MLShapedArray<Float32>) async -> Float {
        let magnitude = vDSP.sumOfSquares(A.scalars).squareRoot() * vDSP.sumOfSquares(B.scalars).squareRoot()
        let dotarray = vDSP.dot(A.scalars, B.scalars)
        return  dotarray / magnitude
    }
    
    func spherical_dist_loss(A: MLShapedArray<Float32>, B: MLShapedArray<Float32>) async -> Float {
        let a = vDSP.divide(A.scalars, sqrt(vDSP.sumOfSquares(A.scalars)))
        let b = vDSP.divide(B.scalars, sqrt(vDSP.sumOfSquares(B.scalars)))

        let magnitude = sqrt(vDSP.sumOfSquares(vDSP.subtract(a, b)))
        return pow(asin(magnitude / 2.0), 2) * 2.0
    }

}
