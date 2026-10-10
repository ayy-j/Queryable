import XCTest
import CoreML
import CoreVideo
import CoreGraphics
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

    func testRegistryContainsVerifiedPresetsByDefault() {
        let registry = EmbeddingModelRegistry()

        XCTAssertEqual(registry.spec(for: "mobileclip-s2"), .mobileCLIPS2)
        XCTAssertEqual(registry.spec(for: "mobileclip2-s4"), .mobileCLIP2S4)
    }

    func testS4SpecMatchesConvertedArtifactContract() {
        let spec = EmbeddingModelSpec.mobileCLIP2S4

        XCTAssertEqual(spec.imageModelName, "ImageEncoder_mobileCLIP2_s4.mlmodelc")
        XCTAssertEqual(spec.textModelName, "TextEncoder_mobileCLIP2_s4.mlmodelc")
        XCTAssertEqual(spec.imageInputName, "colorImage")
        XCTAssertEqual(spec.imageOutputName, "embOutput")
        XCTAssertEqual(spec.textInputName, "input_tokens")
        XCTAssertEqual(spec.textInputType, .multiArrayInt32)
        XCTAssertEqual(spec.textOutputName, "text_embeddings")
        XCTAssertEqual(spec.imageSize, 256)
        XCTAssertEqual(spec.embeddingDimension, 768)
        XCTAssertEqual(spec.contextLength, 77)
        XCTAssertEqual(spec.vocabularySize, 49_408)
        XCTAssertEqual(spec.tokenizerKind, .clipBPE)
        XCTAssertEqual(spec.normalization, .l2)
        XCTAssertNotEqual(spec.compatibilityIdentity, EmbeddingModelSpec.mobileCLIPS2.compatibilityIdentity)
    }

    func testTokenArraysUseDeclaredInputTypeAndPreserveIDs() throws {
        let ids = [49_406, 320, 49_407, 0]
        for inputType in [ModelFeatureType.multiArrayInt32, .multiArrayFloat32] {
            let array = try TextEncoder.tokenArray(ids: ids, shape: [1, 4], inputType: inputType)

            XCTAssertEqual(array.dataType, inputType.multiArrayDataType)
            XCTAssertEqual(array.shape.map { $0.intValue }, [1, 4])
            XCTAssertEqual((0..<array.count).map { array[$0].intValue }, ids)
        }
        XCTAssertThrowsError(try TextEncoder.tokenArray(ids: ids, shape: [1, 4], inputType: .multiArrayFloat16))
    }

    func testS4TextEncoderLoadsAndPredictsWithBundledModel() throws {
        let resources = try XCTUnwrap(Bundle.main.url(forResource: "CoreMLModels", withExtension: nil))
        let spec = EmbeddingModelSpec.mobileCLIP2S4
        guard spec.missingArtifacts(resourcesAt: resources).isEmpty else {
            throw XCTSkip("S4 compiled model artifacts are not bundled in this checkout")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        let encoder = try TextEncoder(resourcesAt: resources, spec: spec, configuration: configuration)
        let embedding = try encoder.computeTextEmbedding(prompt: "a photo of a cat")

        XCTAssertEqual(embedding.shape, [1, spec.embeddingDimension])
        XCTAssertTrue(embedding.scalars.allSatisfy { $0.isFinite })
        XCTAssertTrue(embedding.scalars.contains { $0 != 0 })

        // Analytic IDs for repeated "a": 75 body tokens between BOS and EOS.
        let longPrompt = String(repeating: "a ", count: 200)
        let expectedIDs = [49_406] + Array(repeating: 320, count: 75) + [49_407]
        XCTAssertEqual(try encoder.tokenizer.tokenize(input: longPrompt, contextLength: 77).tokenIDs, expectedIDs)
        let actual = try encoder.computeTextEmbedding(prompt: longPrompt)
        let expected = try encoder.encode(ids: expectedIDs)
        for (value, reference) in zip(actual.scalars, expected.scalars) {
            XCTAssertEqual(value, reference, accuracy: 1e-6)
        }
        for invalidIDs in [Array(expectedIDs.dropLast()), Array(repeating: -1, count: 77), Array(repeating: 49_408, count: 77)] {
            XCTAssertThrowsError(try encoder.encode(ids: invalidIDs)) { error in
                XCTAssertEqual(error as? TextEncoder.TextEncodingError, .invalidTokens)
            }
        }
    }

    func testImageBufferPoolsMatchEachModelResolution() {
        for resolution in [256, 384] {
            let size = CGSize(width: resolution, height: resolution)
            guard let pool = ImgEncoder.pixelBufferPool(
                size: size,
                pixelFormat: kCVPixelFormatType_32ARGB
            ) else {
                XCTFail("Could not create a pixel buffer pool for \(resolution)x\(resolution)")
                continue
            }

            var pixelBuffer: CVPixelBuffer?
            guard CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pixelBuffer) == kCVReturnSuccess,
                  let pixelBuffer else {
                XCTFail("Could not allocate a \(resolution)x\(resolution) pixel buffer")
                continue
            }
            XCTAssertEqual(CVPixelBufferGetWidth(pixelBuffer), resolution)
            XCTAssertEqual(CVPixelBufferGetHeight(pixelBuffer), resolution)
        }
    }

    func testImageBufferPoolsAreReusedPerGeometry() {
        let size = CGSize(width: 384, height: 384)
        let first = ImgEncoder.pixelBufferPool(size: size, pixelFormat: kCVPixelFormatType_32ARGB)
        let second = ImgEncoder.pixelBufferPool(size: size, pixelFormat: kCVPixelFormatType_32ARGB)

        XCTAssertNotNil(first)
        XCTAssertTrue(first === second, "Pools must be cached per size and pixel format")
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

    private var boundaryTokenizer: BPETokenizer {
        BPETokenizer(merges: [:], vocabulary: [
            "<|startoftext|>": 11, "<|endoftext|>": 12, "a</w>": 7, "[PAD]": 0
        ])
    }

    func testFixedContextTokenizationAtAndBeyondEveryBoundary() throws {
        for context in [2, 4, 77] {
            for words in [0, max(0, context - 3), context - 2, context - 1, context * 10] {
                let (tokens, ids) = try boundaryTokenizer.tokenize(
                    input: String(repeating: "a ", count: words), contextLength: context)
                let retained = min(words, context - 2)
                let expected = [11] + Array(repeating: 7, count: retained) + [12]
                    + Array(repeating: 0, count: context - retained - 2)
                XCTAssertEqual(ids, expected, "context \(context), words \(words)")
                XCTAssertEqual(tokens.count, context)
                XCTAssertEqual(tokens.first, "<|startoftext|>")
                XCTAssertEqual(tokens[retained + 1], "<|endoftext|>")
            }
        }
    }

    func testFixedContextRejectsInvalidLengthAndMissingSpecialTokens() {
        for length in [Int.min, -1, 0, 1] {
            XCTAssertThrowsError(try boundaryTokenizer.tokenize(input: "a", contextLength: length)) { error in
                XCTAssertEqual(error as? BPETokenizer.TokenizationError, .invalidContextLength)
            }
        }
        for missing in ["<|startoftext|>", "<|endoftext|>"] {
            var vocabulary = boundaryTokenizer.vocabulary
            vocabulary.removeValue(forKey: missing)
            let tokenizer = BPETokenizer(merges: [:], vocabulary: vocabulary)
            XCTAssertThrowsError(try tokenizer.tokenize(input: "a", contextLength: 77)) { error in
                XCTAssertEqual(error as? BPETokenizer.TokenizationError, .missingSpecialToken(missing))
            }
        }
    }

    func testRawTokenizerRemainsUnboundedWhileFixedContextPreservesEndToken() throws {
        let prompt = String(repeating: "a ", count: 100)
        let raw = boundaryTokenizer.tokenize(input: prompt, minCount: 77).tokenIDs
        let fixed = try boundaryTokenizer.tokenize(input: prompt, contextLength: 77).tokenIDs
        XCTAssertEqual(raw.count, 102)
        XCTAssertEqual(raw.last, 12)
        XCTAssertEqual(fixed.count, 77)
        XCTAssertEqual(Array(fixed.dropLast()), Array(raw.prefix(76)))
        XCTAssertEqual(fixed.last, 12)
    }

    func testFixedContextUsesExistingZeroPaddingFallback() throws {
        let tokenizer = BPETokenizer(merges: [:], vocabulary: ["<|startoftext|>": 11, "<|endoftext|>": 12])
        XCTAssertEqual(try tokenizer.tokenize(input: "", contextLength: 4).tokenIDs, [11, 12, 0, 0])
    }

    func testTokenArrayRejectsMismatchedOrOverflowingShapesBeforeAllocation() {
        for shape in [[], [0], [-1], [1, 3], [2, 2], [Int.max, 2]] {
            XCTAssertThrowsError(try TextEncoder.tokenArray(ids: [11, 12], shape: shape, inputType: .multiArrayInt32)) { error in
                XCTAssertEqual(error as? TextEncoder.TextEncodingError, .invalidTokenShape)
            }
        }
    }

    func testTokenArrayRejectsInvalidIDsWithoutNarrowingCastCrash() {
        for type in [ModelFeatureType.multiArrayInt32, .multiArrayFloat32] {
            for id in [-1, Int(Int32.max) + 1, Int.max] {
                XCTAssertThrowsError(try TextEncoder.tokenArray(ids: [id], shape: [1, 1], inputType: type)) { error in
                    XCTAssertEqual(error as? TextEncoder.TextEncodingError, .invalidTokens)
                }
            }
        }
    }

    func testImageOutputValidationAcceptsSupportedDimensionsAndVectorShapes() throws {
        for dimension in [512, 768, 1_152] {
            let spec = try makeSpec(embeddingDimension: dimension)
            for shape in [[dimension], [1, dimension]] {
                let provider = try imageOutputProvider(shape: shape, firstValue: 3)
                let single = try ImgEncoder.validatedEmbedding(from: provider, spec: spec)
                let batch = try ImgEncoder.validatedEmbeddings(
                    from: MLArrayBatchProvider(array: [provider, provider]), expectedCount: 2, spec: spec
                )

                XCTAssertEqual(single.shape, shape)
                XCTAssertEqual(single.scalars.first, 3, "Validation must preserve unnormalized output")
                XCTAssertEqual(batch.count, 2)
                XCTAssertEqual(batch[0].scalars, single.scalars)
                XCTAssertEqual(batch[1].shape, shape)
            }
        }
    }

    func testImageOutputValidationRejectsWrongFeatureNamesTypesShapesAndDimensions() throws {
        for dimension in [512, 768, 1_152] {
            let spec = try makeSpec(embeddingDimension: dimension)
            var invalid: [MLFeatureProvider] = [
                try imageOutputProvider(shape: [dimension], name: "wrong_output"),
                try MLDictionaryFeatureProvider(dictionary: ["embOutput": "not an array"]),
                try MLDictionaryFeatureProvider(dictionary: [:])
            ]
            for type in [MLMultiArrayDataType.double, .int32, .float16] {
                invalid.append(try imageOutputProvider(shape: [dimension], dataType: type))
            }
            for shape in [[dimension - 1], [dimension + 1], [dimension, 1], [1, 1, dimension], [2, dimension / 2]] {
                invalid.append(try imageOutputProvider(shape: shape))
            }
            for provider in invalid {
                XCTAssertThrowsError(try ImgEncoder.validatedEmbedding(from: provider, spec: spec))
                XCTAssertThrowsError(try ImgEncoder.validatedEmbeddings(
                    from: MLArrayBatchProvider(array: [provider]), expectedCount: 1, spec: spec
                ))
            }
        }
    }

    func testImageOutputValidationRejectsNonfiniteBlankAndNearZeroVectors() throws {
        let spec = try makeSpec()
        for value in [Float.nan, .infinity, -.infinity, 0, 1e-9, 1e-8] {
            let provider = try imageOutputProvider(shape: [512], firstValue: value)
            XCTAssertThrowsError(try ImgEncoder.validatedEmbedding(from: provider, spec: spec))
            XCTAssertThrowsError(try ImgEncoder.validatedEmbeddings(
                from: MLArrayBatchProvider(array: [provider]), expectedCount: 1, spec: spec
            ))
        }
        // A bad component after a valid component must still invalidate the vector.
        let array = try MLMultiArray(shape: [512], dataType: .float32)
        for index in 0..<array.count { array[index] = 0 }
        array[0] = 1
        array[511] = NSNumber(value: Float.nan)
        let provider = try MLDictionaryFeatureProvider(dictionary: ["embOutput": array])
        XCTAssertThrowsError(try ImgEncoder.validatedEmbedding(from: provider, spec: spec))
    }

    func testImageOutputValidationComputesNormWithoutFloat32Overflow() throws {
        let spec = try makeSpec()
        for value in [Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, 2e-8] {
            let provider = try imageOutputProvider(shape: [512], firstValue: value)
            let embedding = try ImgEncoder.validatedEmbedding(from: provider, spec: spec)
            XCTAssertEqual(embedding.scalars.first, value)
        }
    }

    func testImageBatchValidationRequiresExactCountAndNeverReturnsPartialBatch() throws {
        let spec = try makeSpec()
        let valid = try imageOutputProvider(shape: [512])
        let invalid = try imageOutputProvider(shape: [512], firstValue: .nan)
        for providers: [MLFeatureProvider] in [[], [valid], [valid, valid, valid]] {
            XCTAssertThrowsError(try ImgEncoder.validatedEmbeddings(
                from: MLArrayBatchProvider(array: providers), expectedCount: 2, spec: spec
            ))
        }
        for providers: [MLFeatureProvider] in [[valid, invalid], [invalid, valid]] {
            var returned: [MLShapedArray<Float32>]? = nil
            XCTAssertThrowsError(returned = try ImgEncoder.validatedEmbeddings(
                from: MLArrayBatchProvider(array: providers), expectedCount: 2, spec: spec
            ))
            XCTAssertNil(returned)
        }
        let empty = MLArrayBatchProvider(array: [])
        XCTAssertTrue(try ImgEncoder.validatedEmbeddings(from: empty, expectedCount: 0, spec: spec).isEmpty)
        XCTAssertThrowsError(try ImgEncoder.validatedEmbeddings(from: empty, expectedCount: -1, spec: spec))
    }

    private func imageOutputProvider(
        shape: [Int],
        name: String = "embOutput",
        dataType: MLMultiArrayDataType = .float32,
        firstValue: Float = 1
    ) throws -> MLFeatureProvider {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: dataType)
        for index in 0..<array.count { array[index] = 0 }
        array[0] = NSNumber(value: firstValue)
        return try MLDictionaryFeatureProvider(dictionary: [name: array])
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
