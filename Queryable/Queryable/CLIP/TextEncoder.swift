// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import Foundation
import CoreML

#if os(iOS)
import UIKit
#endif

///  A model for encoding text
public struct TextEncoder {

    let spec: EmbeddingModelSpec

    /// Text tokenizer
    var tokenizer: BPETokenizer

    /// Embedding model
    var model: MLModel
    
    init(resourcesAt baseURL: URL,
         spec: EmbeddingModelSpec = .mobileCLIP2S4,
         configuration config: MLModelConfiguration = .init()
    ) throws {
        guard spec.tokenizerKind == .clipBPE,
              let vocabularyName = spec.vocabularyName,
              let mergesName = spec.mergesName else {
            throw TextEncodingError.unsupportedTokenizer
        }
        guard spec.textInputType == .multiArrayFloat32 || spec.textInputType == .multiArrayInt32,
              spec.textOutputType == .multiArrayFloat32,
              spec.imageOutputType == .multiArrayFloat32 else {
            throw TextEncodingError.unsupportedFeatureType
        }
        let missing = spec.missingArtifacts(resourcesAt: baseURL)
        if let firstMissing = missing.first {
            throw ModelArtifactError.missingArtifact(firstMissing)
        }
        let textEncoderURL = baseURL.appending(path: spec.textModelName)
        let vocabURL = baseURL.appending(path: vocabularyName)
        let mergesURL = baseURL.appending(path: mergesName)
        
#if os(iOS)
        // Fallback to CPU only to avoid NN compute error on iPhone < 11 and iPad < 9th gen
        if !UIDevice.chipIsA13OrLater() {
            config.computeUnits = .cpuOnly
        }
#endif

        // Text tokenizer and encoder
        let tokenizer = try BPETokenizer(mergesAt: mergesURL, vocabularyAt: vocabURL)
        let textEncoderModel = try MLModel(contentsOf: textEncoderURL, configuration: config)

        try spec.validate(textModel: textEncoderModel)
        guard tokenizer.vocabulary.count == spec.vocabularySize else {
            throw TextEncodingError.modelContractMismatch
        }

        self.spec = spec
        self.tokenizer = tokenizer
        self.model = textEncoderModel
    }
    
    public func computeTextEmbedding(prompt: String) throws -> MLShapedArray<Float32> {
        let promptEmbedding = try self.encode(prompt)
        return promptEmbedding
    }
    
    /**
    /// Creates text encoder which embeds a tokenized string
    ///
    /// - Parameters:
    ///   - tokenizer: Tokenizer for input text
    ///   - model: Model for encoding tokenized text
    public init(tokenizer: BPETokenizer, model: MLModel) {
        self.tokenizer = tokenizer
        self.model = model
    }
     */

    /// Encode input text/string
    ///
    ///  - Parameters:
    ///     - text: Input text to be tokenized and then embedded
    ///  - Returns: Embedding representing the input text
    private func encode(_ text: String) throws -> MLShapedArray<Float32> {

        // Get models expected input length
        let inputLength = inputShape.last!

        // Tokenize, padding to the expected length
        var (tokens, ids) = tokenizer.tokenize(input: text, minCount: inputLength)

        // Truncate if necessary
        if ids.count > inputLength {
            tokens = tokens.dropLast(tokens.count - inputLength)
            ids = ids.dropLast(ids.count - inputLength)
            let truncated = tokenizer.decode(tokens: tokens)
            print("Needed to truncate input '\(text)' to '\(truncated)'")
        }

        // Use the model to generate the embedding
        return try encode(ids: ids)
    }

    /// Prediction queue
    let queue = DispatchQueue(label: "textencoder.predict")

    func encode(ids: [Int]) throws -> MLShapedArray<Float32> {
        let inputName = inputDescription.name
        let inputShape = inputShape

        let floatIds = ids.map { Float32($0) }
        let inputArray = MLShapedArray<Float32>(scalars: floatIds, shape: inputShape)
        let inputFeatures = try MLDictionaryFeatureProvider(
            dictionary: [inputName: MLMultiArray(inputArray)])

        let result = try queue.sync { try model.prediction(from: inputFeatures) }
        guard let embeddingFeature = result.featureValue(for: spec.textOutputName),
              let multiArray = embeddingFeature.multiArrayValue,
              multiArray.dataType == .float32,
              multiArray.count == spec.embeddingDimension else {
            throw TextEncodingError.invalidOutput
        }
        return MLShapedArray<Float32>(converting: multiArray)
    }

    enum TextEncodingError: Error {
        case modelContractMismatch
        case invalidOutput
        case unsupportedTokenizer
        case unsupportedFeatureType
    }

    var inputDescription: MLFeatureDescription {
        model.modelDescription.inputDescriptionsByName[spec.textInputName]!
    }

    var inputShape: [Int] {
        inputDescription.multiArrayConstraint!.shape.map { $0.intValue }
    }

}
