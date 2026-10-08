import XCTest
@testable import Queryable

final class EmbeddingModelSpecTests: XCTestCase {
    func testS2SpecPreservesResourceAndInferenceContract() {
        let spec = EmbeddingModelSpec.mobileCLIPS2

        XCTAssertEqual(spec.imageModelName, "ImageEncoder_mobileCLIP_s2.mlmodelc")
        XCTAssertEqual(spec.textModelName, "TextEncoder_mobileCLIP_s2.mlmodelc")
        XCTAssertEqual(spec.tokenizerAssets, ["vocab.json", "merges.txt"])
        XCTAssertEqual(spec.imageInputName, "colorImage")
        XCTAssertEqual(spec.imageOutputName, "embOutput")
        XCTAssertEqual(spec.textOutputName, "text_embeddings")
        XCTAssertEqual(spec.imageSize, 256)
        XCTAssertEqual(spec.embeddingDimension, 512)
        XCTAssertEqual(spec.contextLength, 77)
        XCTAssertEqual(spec.tokenizerKind, .clipBPE)
    }

    func testRejectsMissingMetadataInvalidDimensionsAndUnsupportedContracts() {
        XCTAssertThrowsError(try makeSpec(modelID: " "))
        XCTAssertThrowsError(try makeSpec(imageSize: 0))
        XCTAssertThrowsError(try makeSpec(embeddingDimension: 0))
        XCTAssertThrowsError(try makeSpec(contextLength: 0))
        XCTAssertThrowsError(try makeSpec(textInputType: .image))
        XCTAssertThrowsError(try makeSpec(storageScalarType: "float16"))
        XCTAssertThrowsError(try makeSpec(tokenizerAssets: ["vocab.json"]))
    }

    func testCompatibilityIdentityDistinguishesEqualDimensionModels() throws {
        let first = try makeSpec(modelID: "first", embeddingDimension: 512)
        let differentRevision = try makeSpec(modelID: "first", revision: "v2", embeddingDimension: 512)
        let differentTokenizer = try makeSpec(modelID: "first", embeddingDimension: 512, tokenizerAssets: ["other-vocab.json", "other-merges.txt"])

        XCTAssertEqual(first.embeddingDimension, differentRevision.embeddingDimension)
        XCTAssertEqual(first.embeddingDimension, differentTokenizer.embeddingDimension)
        XCTAssertNotEqual(first.compatibilityIdentity, differentRevision.compatibilityIdentity)
        XCTAssertNotEqual(first.compatibilityIdentity, differentTokenizer.compatibilityIdentity)
        XCTAssertNotEqual(first.compatibilityIdentity, EmbeddingModelSpec.mobileCLIPS2.compatibilityIdentity)
    }

    func testSigLIPContractsAreRepresentableWithoutRegisteringRuntime() throws {
        let preprocessing = ImagePreprocessing(
            resizeFilter: "CILanczosScaleTransform",
            pixelFormat: "32ARGB",
            aspectRatioMode: "stretch",
            fingerprint: "siglip-preprocess-v1"
        )
        let standard = try EmbeddingModelSpec.sigLIPSo400m(
            revision: "so400m-v1",
            imageModelName: "image.mlmodelc",
            textModelName: "text.mlmodelc",
            tokenizerAssets: ["gemma-tokenizer.model"],
            imagePreprocessing: preprocessing,
            imageInputName: "pixels",
            imageOutputName: "image_embeddings",
            textInputName: "tokens",
            textInputType: .multiArrayInt32,
            textOutputName: "text_embeddings"
        )
        let optionalG = try EmbeddingModelSpec.sigLIPSo400m(
            revision: "so400m-g-v1",
            imageModelName: "image-g.mlmodelc",
            textModelName: "text-g.mlmodelc",
            tokenizerAssets: ["gemma-tokenizer-g.model"],
            imagePreprocessing: preprocessing,
            imageInputName: "pixels",
            imageOutputName: "image_embeddings",
            textInputName: "tokens",
            textInputType: .multiArrayInt32,
            textOutputName: "text_embeddings",
            embeddingDimension: 1_536
        )

        XCTAssertEqual(standard.imageSize, 384)
        XCTAssertEqual(standard.embeddingDimension, 1_152)
        XCTAssertEqual(standard.tokenizerKind, .gemma)
        XCTAssertEqual(standard.textInputType, .multiArrayInt32)
        XCTAssertEqual(standard.contextLength, 64)
        XCTAssertEqual(optionalG.embeddingDimension, 1_536)
    }

    func testRegistryContainsOnlyVerifiedS2PresetByDefault() {
        let registry = EmbeddingModelRegistry()

        XCTAssertEqual(registry.spec(for: "mobileclip-s2"), .mobileCLIPS2)
        XCTAssertNil(registry.spec(for: "mobileclip2-s4"))
    }

    func testS4ContractCanBeDescribedForArtifactGatedRegistration() throws {
        let s4 = try makeSpec(modelID: "mobileclip2-s4", embeddingDimension: 768)

        XCTAssertEqual(s4.imageSize, 256)
        XCTAssertEqual(s4.embeddingDimension, 768)
        XCTAssertEqual(s4.contextLength, 77)
        XCTAssertEqual(s4.tokenizerKind, .clipBPE)
    }

    func testMissingArtifactsReportsAbsentModelBundles() throws {
        let fileManager = FileManager.default
        let dir = fileManager.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: dir) }

        let spec = try makeSpec()
        XCTAssertEqual(
            spec.missingArtifacts(resourcesAt: dir),
            ["image.mlmodelc", "text.mlmodelc", "vocab.json", "merges.txt"]
        )

        // Tokenizer assets present, compiled towers still missing (fresh-checkout state).
        try "v".write(to: dir.appendingPathComponent("vocab.json"), atomically: true, encoding: .utf8)
        try "m".write(to: dir.appendingPathComponent("merges.txt"), atomically: true, encoding: .utf8)
        XCTAssertEqual(
            spec.missingArtifacts(resourcesAt: dir),
            ["image.mlmodelc", "text.mlmodelc"]
        )

        try fileManager.createDirectory(at: dir.appendingPathComponent("image.mlmodelc"), withIntermediateDirectories: true)
        try fileManager.createDirectory(at: dir.appendingPathComponent("text.mlmodelc"), withIntermediateDirectories: true)
        XCTAssertTrue(spec.missingArtifacts(resourcesAt: dir).isEmpty)
    }

    func testMissingArtifactErrorDescribesDownloadRemedy() {
        let message = ModelArtifactError.missingArtifact("ImageEncoder_mobileCLIP_s2.mlmodelc").localizedDescription
        XCTAssertTrue(message.contains("ImageEncoder_mobileCLIP_s2.mlmodelc"))
        XCTAssertTrue(message.contains("drive.google.com"))
    }

    private func makeSpec(
        modelID: String = "test-model",
        revision: String = "v1",
        imageSize: Int = 256,
        embeddingDimension: Int = 512,
        contextLength: Int = 77,
        textInputType: ModelFeatureType = .multiArrayFloat32,
        tokenizerAssets: [String] = ["vocab.json", "merges.txt"],
        tokenizerKind: TokenizerKind = .clipBPE,
        storageScalarType: String = "float32"
    ) throws -> EmbeddingModelSpec {
        try EmbeddingModelSpec(
            modelID: modelID,
            revision: revision,
            imageModelName: "image.mlmodelc",
            textModelName: "text.mlmodelc",
            imageInputName: "colorImage",
            imageInputType: .image,
            imageOutputName: "embOutput",
            imageOutputType: .multiArrayFloat32,
            textInputName: "input_ids",
            textInputType: textInputType,
            textOutputName: "text_embeddings",
            textOutputType: .multiArrayFloat32,
            imageSize: imageSize,
            imagePreprocessing: ImagePreprocessing(
                resizeFilter: "CILanczosScaleTransform",
                pixelFormat: "32ARGB",
                aspectRatioMode: "stretch",
                fingerprint: "preprocess-v1"
            ),
            embeddingDimension: embeddingDimension,
            tokenizerKind: tokenizerKind,
            tokenizerAssets: tokenizerAssets,
            contextLength: contextLength,
            vocabularySize: tokenizerKind == .clipBPE ? 49_408 : nil,
            normalization: .l2,
            storageScalarType: storageScalarType
        )
    }
}
