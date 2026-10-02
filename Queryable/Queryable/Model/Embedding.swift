//
//  Embedding.swift
//  Queryable
//
//  Created by Ke Fang on 2022/12/20.
//

import Foundation
import CoreML
import CryptoKit

struct EmbeddingModelSpec: Equatable, Sendable {
    let modelID: String
    let imageModelName: String
    let textModelName: String
    let vocabularyName: String
    let mergesName: String
    let imageInputName: String
    let imageOutputName: String
    let textOutputName: String
    let imageSize: Int
    let embeddingDimension: Int
    let contextLength: Int
    let vocabularySize: Int
    let normalizeEmbeddings: Bool
    let preprocessingFingerprint: String

    static let mobileCLIPS2 = EmbeddingModelSpec(
        modelID: "mobileclip-s2",
        imageModelName: "ImageEncoder_mobileCLIP_s2.mlmodelc",
        textModelName: "TextEncoder_mobileCLIP_s2.mlmodelc",
        vocabularyName: "vocab.json",
        mergesName: "merges.txt",
        imageInputName: "colorImage",
        imageOutputName: "embOutput",
        textOutputName: "text_embeddings",
        imageSize: 256,
        embeddingDimension: 512,
        contextLength: 77,
        vocabularySize: 49_408,
        normalizeEmbeddings: true,
        preprocessingFingerprint: "ci-lanczos-argb-256-v1"
    )

    static let mobileCLIP2S4 = EmbeddingModelSpec(
        modelID: "mobileclip2-s4",
        imageModelName: "ImageEncoder_mobileCLIP2_s4.mlmodelc",
        textModelName: "TextEncoder_mobileCLIP2_s4.mlmodelc",
        vocabularyName: "vocab_mobileclip2_s4.json",
        mergesName: "merges_mobileclip2_s4.txt",
        imageInputName: "colorImage",
        imageOutputName: "embOutput",
        textOutputName: "text_embeddings",
        imageSize: 256,
        embeddingDimension: 768,
        contextLength: 77,
        vocabularySize: 49_408,
        normalizeEmbeddings: true,
        preprocessingFingerprint: "ci-lanczos-argb-256-v1"
    )

    func checkpointHash(resourcesAt baseURL: URL) throws -> String {
        let fileManager = FileManager.default
        let artifacts = [imageModelName, textModelName, vocabularyName, mergesName]
            .map { baseURL.appendingPathComponent($0) }
        var files = [URL]()

        for artifact in artifacts {
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: artifact.path, isDirectory: &isDirectory) else {
                throw ModelArtifactError.missingArtifact(artifact.lastPathComponent)
            }
            if isDirectory.boolValue {
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
            } else {
                files.append(artifact)
            }
        }

        var hasher = SHA256()
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
}

enum ModelArtifactError: Error {
    case missingArtifact(String)
    case unreadableArtifact(String)
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
