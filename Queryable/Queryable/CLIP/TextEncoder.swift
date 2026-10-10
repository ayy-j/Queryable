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
         spec: EmbeddingModelSpec = .appDefault,
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
        
#if os(iOS) && !targetEnvironment(macCatalyst)
        // UIDevice.model is a generic label ("iPhone"), not a hardware ID.
        // Keep the existing compatibility fallback only for known older devices.
        if Self.requiresCPUOnlyTextEncoding(hardwareIdentifier: Self.hardwareIdentifier) {
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

    static func requiresCPUOnlyTextEncoding(hardwareIdentifier: String) -> Bool {
        let parts = hardwareIdentifier.split(separator: ",")
        guard parts.count == 2 else { return false }
        for family in ["iPhone", "iPad"] where parts[0].hasPrefix(family) {
            guard let generation = Int(parts[0].dropFirst(family.count)) else { return false }
            // iPhone12,* is A13; iPad12,* is the A13 ninth-generation iPad.
            return generation < 12
        }
        return false
    }

#if os(iOS)
    private static var hardwareIdentifier: String {
        if let simulated = ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"] {
            return simulated
        }
        var systemInfo = utsname()
        uname(&systemInfo)
        return withUnsafeBytes(of: &systemInfo.machine) { bytes in
            String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
        }
    }
#endif
    
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

        let (_, ids) = try tokenizer.tokenize(input: text, contextLength: spec.contextLength)

        // Use the model to generate the embedding
        return try encode(ids: ids)
    }

    /// Prediction queue
    let queue = DispatchQueue(label: "textencoder.predict")

    func encode(ids: [Int]) throws -> MLShapedArray<Float32> {
        guard ids.count == spec.contextLength,
              let vocabularySize = spec.vocabularySize,
              ids.allSatisfy({ $0 >= 0 && $0 < vocabularySize }) else {
            throw TextEncodingError.invalidTokens
        }
        let inputName = inputDescription.name
        let inputShape = inputShape

        let inputArray = try Self.tokenArray(ids: ids, shape: inputShape, inputType: spec.textInputType)
        let inputFeatures = try MLDictionaryFeatureProvider(
            dictionary: [inputName: inputArray])

        let result = try queue.sync { try model.prediction(from: inputFeatures) }
        guard let embeddingFeature = result.featureValue(for: spec.textOutputName),
              let multiArray = embeddingFeature.multiArrayValue,
              multiArray.dataType == .float32,
              multiArray.count == spec.embeddingDimension else {
            throw TextEncodingError.invalidOutput
        }
        return MLShapedArray<Float32>(converting: multiArray)
    }

    static func tokenArray(ids: [Int], shape: [Int], inputType: ModelFeatureType) throws -> MLMultiArray {
        // Validate before MLShapedArray's shape precondition or a narrowing Int32 cast.
        guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }) else {
            throw TextEncodingError.invalidTokenShape
        }
        var count = 1
        for size in shape {
            let (product, overflow) = count.multipliedReportingOverflow(by: size)
            guard !overflow else { throw TextEncodingError.invalidTokenShape }
            count = product
        }
        guard count == ids.count else { throw TextEncodingError.invalidTokenShape }
        guard ids.allSatisfy({ $0 >= 0 && Int32(exactly: $0) != nil }) else {
            throw TextEncodingError.invalidTokens
        }
        switch inputType {
        case .multiArrayInt32:
            return MLMultiArray(MLShapedArray<Int32>(scalars: ids.map { Int32($0) }, shape: shape))
        case .multiArrayFloat32:
            return MLMultiArray(MLShapedArray<Float32>(scalars: ids.map { Float32($0) }, shape: shape))
        default:
            throw TextEncodingError.unsupportedFeatureType
        }
    }

    enum TextEncodingError: Error, Equatable {
        case invalidTokens
        case invalidTokenShape
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
