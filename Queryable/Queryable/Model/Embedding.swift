//
//  Embedding.swift
//  Queryable
//
//  Created by Ke Fang on 2022/12/20.
//

import Foundation
import CoreML
import CryptoKit

enum ModelFeatureType: String, Sendable {
    case image
    case multiArrayFloat32
    case multiArrayFloat16
    case multiArrayInt32

    var multiArrayDataType: MLMultiArrayDataType? {
        switch self {
        case .image:
            return nil
        case .multiArrayFloat32:
            return .float32
        case .multiArrayFloat16:
            return .float16
        case .multiArrayInt32:
            return .int32
        }
    }
}

enum TokenizerKind: String, Sendable {
    case clipBPE
    case gemma
}

enum EmbeddingNormalization: String, Sendable {
    case none
    case l2
}

enum EmbeddingStorageScalarType: String, Sendable {
    case float32
}

struct ImagePreprocessing: Equatable, Sendable {
    let resizeFilter: String
    let pixelFormat: String
    let aspectRatioMode: String
    let fingerprint: String
}

struct EmbeddingModelSpec: Equatable, Sendable {
    let modelID: String
    let revision: String
    let imageModelName: String
    let textModelName: String
    let imageInputName: String
    let imageInputType: ModelFeatureType
    let imageOutputName: String
    let imageOutputType: ModelFeatureType
    let textInputName: String
    let textInputType: ModelFeatureType
    let textOutputName: String
    let textOutputType: ModelFeatureType
    let imageSize: Int
    let imagePreprocessing: ImagePreprocessing
    let embeddingDimension: Int
    let tokenizerKind: TokenizerKind
    let tokenizerAssets: [String]
    let contextLength: Int
    let vocabularySize: Int?
    let normalization: EmbeddingNormalization
    let storageScalarType: EmbeddingStorageScalarType

    init(
        modelID: String,
        revision: String,
        imageModelName: String,
        textModelName: String,
        imageInputName: String,
        imageInputType: ModelFeatureType,
        imageOutputName: String,
        imageOutputType: ModelFeatureType,
        textInputName: String,
        textInputType: ModelFeatureType,
        textOutputName: String,
        textOutputType: ModelFeatureType,
        imageSize: Int,
        imagePreprocessing: ImagePreprocessing,
        embeddingDimension: Int,
        tokenizerKind: TokenizerKind,
        tokenizerAssets: [String],
        contextLength: Int,
        vocabularySize: Int?,
        normalization: EmbeddingNormalization,
        storageScalarType: String
    ) throws {
        let names = [
            modelID, revision, imageModelName, textModelName, imageInputName,
            imageOutputName, textInputName, textOutputName,
            imagePreprocessing.resizeFilter, imagePreprocessing.pixelFormat,
            imagePreprocessing.aspectRatioMode, imagePreprocessing.fingerprint
        ]
        guard names.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }),
              ([imageModelName, textModelName] + tokenizerAssets)
                .allSatisfy({ URL(fileURLWithPath: $0).lastPathComponent == $0 && $0 != "." && $0 != ".." }),
              modelID.range(of: #"^[A-Za-z0-9._-]+$"#, options: .regularExpression) != nil,
              imageSize > 0,
              embeddingDimension > 0,
              contextLength > 0,
              tokenizerAssets.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }),
              !tokenizerAssets.isEmpty,
              vocabularySize.map({ $0 > 0 }) ?? true,
              imageInputType == .image,
              imageOutputType.multiArrayDataType != nil,
              textInputType.multiArrayDataType != nil,
              textOutputType.multiArrayDataType != nil,
              EmbeddingStorageScalarType(rawValue: storageScalarType) != nil,
              tokenizerKind != .clipBPE || (tokenizerAssets.count == 2 && vocabularySize != nil) else {
            throw EmbeddingModelSpecError.invalidContract
        }

        self.modelID = modelID
        self.revision = revision
        self.imageModelName = imageModelName
        self.textModelName = textModelName
        self.imageInputName = imageInputName
        self.imageInputType = imageInputType
        self.imageOutputName = imageOutputName
        self.imageOutputType = imageOutputType
        self.textInputName = textInputName
        self.textInputType = textInputType
        self.textOutputName = textOutputName
        self.textOutputType = textOutputType
        self.imageSize = imageSize
        self.imagePreprocessing = imagePreprocessing
        self.embeddingDimension = embeddingDimension
        self.tokenizerKind = tokenizerKind
        self.tokenizerAssets = tokenizerAssets
        self.contextLength = contextLength
        self.vocabularySize = vocabularySize
        self.normalization = normalization
        self.storageScalarType = EmbeddingStorageScalarType(rawValue: storageScalarType)!
    }

    var normalizeEmbeddings: Bool { normalization == .l2 }
    var preprocessingFingerprint: String { imagePreprocessing.fingerprint }
    var vocabularyName: String? { tokenizerAssets.first }
    var mergesName: String? { tokenizerKind == .clipBPE ? tokenizerAssets.last : nil }

    /// All on-disk artifacts this spec needs, in a stable order.
    var requiredArtifactNames: [String] {
        [imageModelName, textModelName] + tokenizerAssets
    }

    /// Names of required artifacts absent at `baseURL` (the bundled `CoreMLModels` dir).
    /// Compiled `.mlmodelc` bundles are git-ignored (see `.gitignore`), so a fresh
    /// checkout/bundle contains only the tokenizer assets until the models are
    /// downloaded manually. Callers should check this before `MLModel(contentsOf:)`,
    /// whose "model is not found at URL" error does not explain the missing step.
    func missingArtifacts(resourcesAt baseURL: URL) -> [String] {
        let fileManager = FileManager.default
        return requiredArtifactNames.filter { name in
            !fileManager.fileExists(atPath: baseURL.appendingPathComponent(name).path)
        }
    }

    var compatibilityIdentity: String {
        var components = [
            modelID, revision, imageModelName, textModelName,
            imageInputName, imageInputType.rawValue, imageOutputName, imageOutputType.rawValue,
            textInputName, textInputType.rawValue, textOutputName, textOutputType.rawValue,
            String(imageSize), imagePreprocessing.resizeFilter, imagePreprocessing.pixelFormat,
            imagePreprocessing.aspectRatioMode, imagePreprocessing.fingerprint,
            String(embeddingDimension), tokenizerKind.rawValue,
            String(contextLength), vocabularySize.map(String.init) ?? "", normalization.rawValue,
            storageScalarType.rawValue
        ]
        components.append(contentsOf: tokenizerAssets)
        let data = components.map { "\($0.utf8.count):\($0)" }.joined()
        let digest = SHA256.hash(data: Data(data.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    static let mobileCLIPS2 = try! EmbeddingModelSpec(
        modelID: "mobileclip-s2",
        revision: "mobileclip-s2-v1",
        imageModelName: "ImageEncoder_mobileCLIP_s2.mlmodelc",
        textModelName: "TextEncoder_mobileCLIP_s2.mlmodelc",
        imageInputName: "colorImage",
        imageInputType: .image,
        imageOutputName: "embOutput",
        imageOutputType: .multiArrayFloat32,
        textInputName: "input_ids",
        textInputType: .multiArrayFloat32,
        textOutputName: "text_embeddings",
        textOutputType: .multiArrayFloat32,
        imageSize: 256,
        imagePreprocessing: ImagePreprocessing(
            resizeFilter: "CILanczosScaleTransform",
            pixelFormat: "32ARGB",
            aspectRatioMode: "stretch",
            fingerprint: "ci-lanczos-argb-256-v1"
        ),
        embeddingDimension: 512,
        tokenizerKind: .clipBPE,
        tokenizerAssets: ["vocab.json", "merges.txt"],
        contextLength: 77,
        vocabularySize: 49_408,
        normalization: .l2,
        storageScalarType: "float32"
    )

    /// MobileCLIP2-S4 (issue #16): FP16 towers exported from the pinned
    /// `apple/MobileCLIP2-S4` checkpoint. Int32 `input_tokens` text input,
    /// raw unnormalized 768-d features, ImageNet mean/std
    /// baked into the image tower. L2 normalization happens at storage/search time.
    static let mobileCLIP2S4 = try! EmbeddingModelSpec(
        modelID: "mobileclip2-s4",
        revision: "mobileclip2-s4-v1",
        imageModelName: "ImageEncoder_mobileCLIP2_s4.mlmodelc",
        textModelName: "TextEncoder_mobileCLIP2_s4.mlmodelc",
        imageInputName: "colorImage",
        imageInputType: .image,
        imageOutputName: "embOutput",
        imageOutputType: .multiArrayFloat32,
        textInputName: "input_tokens",
        textInputType: .multiArrayInt32,
        textOutputName: "text_embeddings",
        textOutputType: .multiArrayFloat32,
        imageSize: 256,
        imagePreprocessing: ImagePreprocessing(
            resizeFilter: "CILanczosScaleTransform",
            pixelFormat: "32ARGB",
            aspectRatioMode: "stretch",
            fingerprint: "ci-lanczos-argb-256-v1"
        ),
        embeddingDimension: 768,
        tokenizerKind: .clipBPE,
        tokenizerAssets: ["vocab.json", "merges.txt"],
        contextLength: 77,
        vocabularySize: 49_408,
        normalization: .l2,
        storageScalarType: "float32"
    )

    static func sigLIPSo400m(
        revision: String,
        imageModelName: String,
        textModelName: String,
        tokenizerAssets: [String],
        imagePreprocessing: ImagePreprocessing,
        imageInputName: String,
        imageOutputName: String,
        textInputName: String,
        textInputType: ModelFeatureType,
        textOutputName: String,
        embeddingDimension: Int = 1_152
    ) throws -> EmbeddingModelSpec {
        guard embeddingDimension == 1_152 || embeddingDimension == 1_536 else {
            throw EmbeddingModelSpecError.invalidContract
        }
        return try EmbeddingModelSpec(
            modelID: embeddingDimension == 1_536 ? "siglip-so400m-g" : "siglip-so400m",
            revision: revision,
            imageModelName: imageModelName,
            textModelName: textModelName,
            imageInputName: imageInputName,
            imageInputType: .image,
            imageOutputName: imageOutputName,
            imageOutputType: .multiArrayFloat32,
            textInputName: textInputName,
            textInputType: textInputType,
            textOutputName: textOutputName,
            textOutputType: .multiArrayFloat32,
            imageSize: 384,
            imagePreprocessing: imagePreprocessing,
            embeddingDimension: embeddingDimension,
            tokenizerKind: .gemma,
            tokenizerAssets: tokenizerAssets,
            contextLength: 64,
            vocabularySize: nil,
            normalization: .l2,
            storageScalarType: "float32"
        )
    }

    func checkpointHash(resourcesAt baseURL: URL) throws -> String {
        let fileManager = FileManager.default
        let artifacts = [imageModelName, textModelName] + tokenizerAssets
        let artifactURLs = artifacts.map { baseURL.appendingPathComponent($0) }
        var files = [URL]()

        for artifact in artifactURLs {
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: artifact.path, isDirectory: &isDirectory) else {
                throw ModelArtifactError.missingArtifact(artifact.lastPathComponent)
            }
            if isDirectory.boolValue {
                let fileCountBeforeEnumeration = files.count
                guard let enumerator = fileManager.enumerator(
                    at: artifact,
                    includingPropertiesForKeys: [.isRegularFileKey],
                    options: [.skipsHiddenFiles]
                ) else {
                    throw ModelArtifactError.unreadableArtifact(artifact.lastPathComponent)
                }
                for case let file as URL in enumerator {
                    if (try? file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true {
                        files.append(file)
                    }
                }
                guard files.count > fileCountBeforeEnumeration else {
                    throw ModelArtifactError.unreadableArtifact(artifact.lastPathComponent)
                }
            } else {
                files.append(artifact)
            }
        }

        var hasher = SHA256()
        hasher.update(data: Data(compatibilityIdentity.utf8))
        for file in files.sorted(by: { $0.path < $1.path }) {
            let relativePath = String(file.path.dropFirst(baseURL.path.count))
            hasher.update(data: Data(relativePath.utf8))
            let handle = try FileHandle(forReadingFrom: file)
            defer { try? handle.close() }
            while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty {
                hasher.update(data: data)
            }
        }

        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    func validate(imageModel: MLModel) throws {
        let inputs = imageModel.modelDescription.inputDescriptionsByName
        let outputs = imageModel.modelDescription.outputDescriptionsByName
        guard let imageInput = inputs[imageInputName],
              imageInput.type == .image,
              let imageConstraint = imageInput.imageConstraint,
              imageConstraint.pixelsWide == imageSize,
              imageConstraint.pixelsHigh == imageSize,
              let imageOutput = outputs[imageOutputName],
              imageOutput.type == .multiArray,
              imageOutput.multiArrayConstraint?.dataType == imageOutputType.multiArrayDataType,
              imageOutput.multiArrayConstraint?.shape.reduce(1, { $0 * $1.intValue }) == embeddingDimension else {
            throw EmbeddingModelSpecError.modelFeatureMismatch(imageModelName)
        }
    }

    func validateImageRuntimePreprocessing() throws {
        guard imagePreprocessing.pixelFormat == "32ARGB",
              imagePreprocessing.aspectRatioMode == "stretch" else {
            throw EmbeddingModelSpecError.unsupportedImagePreprocessing
        }
    }

    func validate(textModel: MLModel) throws {
        let inputs = textModel.modelDescription.inputDescriptionsByName
        let outputs = textModel.modelDescription.outputDescriptionsByName
        guard let textInput = inputs[textInputName],
              textInput.type == .multiArray,
              textInput.multiArrayConstraint?.dataType == textInputType.multiArrayDataType,
              textInput.multiArrayConstraint?.shape.last?.intValue == contextLength,
              let textOutput = outputs[textOutputName],
              textOutput.type == .multiArray,
              textOutput.multiArrayConstraint?.dataType == textOutputType.multiArrayDataType,
              textOutput.multiArrayConstraint?.shape.reduce(1, { $0 * $1.intValue }) == embeddingDimension else {
            throw EmbeddingModelSpecError.modelFeatureMismatch(textModelName)
        }
    }
}

enum EmbeddingModelSpecError: Error, LocalizedError {
    case invalidContract
    case modelFeatureMismatch(String)
    case unsupportedImagePreprocessing

    var errorDescription: String? {
        switch self {
        case .invalidContract:
            return "Model spec contract is invalid: one or more required fields are empty, out of range, or mutually inconsistent."
        case .modelFeatureMismatch(let modelName):
            return "Model '\(modelName)' feature shape or type does not match the embedding spec (wrong input/output name, dimension, or data type)."
        case .unsupportedImagePreprocessing:
            return "Unsupported image preprocessing configuration: only 32ARGB pixel format with stretch aspect ratio is supported."
        }
    }
}

struct EmbeddingModelRegistry {
    private(set) var models: [String: EmbeddingModelSpec] = [
        EmbeddingModelSpec.mobileCLIPS2.compatibilityIdentity: .mobileCLIPS2,
        EmbeddingModelSpec.mobileCLIP2S4.compatibilityIdentity: .mobileCLIP2S4
    ]

    func spec(for modelID: String, revision: String? = nil) -> EmbeddingModelSpec? {
        models.values.first {
            $0.modelID == modelID && (revision == nil || $0.revision == revision)
        }
    }

    mutating func register(_ spec: EmbeddingModelSpec, resourcesAt baseURL: URL) throws {
        guard !models.values.contains(where: {
            $0.modelID == spec.modelID && $0.revision == spec.revision
        }) else {
            throw EmbeddingModelRegistryError.duplicateModelRevision
        }
        guard spec.tokenizerKind == .clipBPE,
              spec.textInputType == .multiArrayFloat32 || spec.textInputType == .multiArrayInt32,
              spec.textOutputType == .multiArrayFloat32,
              spec.imageOutputType == .multiArrayFloat32 else {
            throw EmbeddingModelRegistryError.unsupportedRuntime
        }
        guard let vocabularyName = spec.vocabularyName,
              let mergesName = spec.mergesName,
              let vocabularySize = spec.vocabularySize else {
            throw EmbeddingModelRegistryError.invalidTokenizer
        }
        let tokenizer = try BPETokenizer(
            mergesAt: baseURL.appendingPathComponent(mergesName),
            vocabularyAt: baseURL.appendingPathComponent(vocabularyName)
        )
        guard tokenizer.vocabulary.count == vocabularySize else {
            throw EmbeddingModelRegistryError.invalidTokenizer
        }
        let imageModel = try MLModel(contentsOf: baseURL.appendingPathComponent(spec.imageModelName))
        let textModel = try MLModel(contentsOf: baseURL.appendingPathComponent(spec.textModelName))
        try spec.validateImageRuntimePreprocessing()
        try spec.validate(imageModel: imageModel)
        try spec.validate(textModel: textModel)
        _ = try spec.checkpointHash(resourcesAt: baseURL)
        models[spec.compatibilityIdentity] = spec
    }
}

enum EmbeddingModelRegistryError: Error {
    case duplicateModelRevision
    case unsupportedRuntime
    case invalidTokenizer
}

enum ModelArtifactError: Error {
    case missingArtifact(String)
    case unreadableArtifact(String)
}

extension ModelArtifactError: LocalizedError {
    static var modelDownloadURL: String {
        "https://drive.google.com/drive/folders/12ze3UcqrXt9qeySGh_j_zWE-PWRDTzJv?usp=drive_link"
    }

    var errorDescription: String? {
        switch self {
        case .missingArtifact(let name):
            return "Missing model file “\(name)”. Download it from \(Self.modelDownloadURL) and place it in the app’s CoreMLModels folder, then rebuild."
        case .unreadableArtifact(let name):
            return "Model file “\(name)” exists but could not be read. Re-download it from \(Self.modelDownloadURL) and rebuild."
        }
    }
}

class Embedding: NSObject, NSSecureCoding {
    static var supportsSecureCoding: Bool = true
    
    var id: String?
    var embedding: MLMultiArray?
    
    init(id: String, embedding: MLMultiArray) {
        self.id = id
        self.embedding = embedding
    }
    
    func encode(with aCoder: NSCoder) {
        aCoder.encode(self.id, forKey: "id")
        aCoder.encode(self.embedding, forKey: "embedding")
    }
    
    required init?(coder aDecoder: NSCoder) {
        self.id = aDecoder.decodeObject(forKey: "id") as? String
        self.embedding = aDecoder.decodeObject(forKey: "embedding") as? MLMultiArray
    }
}
