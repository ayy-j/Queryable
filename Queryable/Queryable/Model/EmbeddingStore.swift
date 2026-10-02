//
//  EmbeddingStore.swift
//  Queryable
//
//  Efficient binary embedding storage with incremental saves.
//  Replaces NSKeyedArchiver full-file rewrites with append-only journal + tombstones.
//
//  File format (v2):
//    Header: "QEMB" + version UInt32 + record count UInt64 + metadata length + JSON metadata
//    Record: idLength UInt16 + id UTF-8 bytes + embedding Float32[model dimension]
//
//  Journal file: same record format, no header (append-only for new embeddings)
//  Tombstone file: newline-separated IDs of deleted embeddings
//

import Foundation
import CoreML
import Accelerate

/// @unchecked Sendable: all stored properties are immutable after init (let).
/// loadAll() is a pure reader that returns a fresh dictionary with no shared mutable state,
/// so it is safe to call from a detached Task. Write methods (appendNew, markDeleted, etc.)
/// are only called from the @MainActor-isolated PhotoSearcher, so no concurrent writes occur.
class EmbeddingStore: @unchecked Sendable {
    private struct HeaderMetadata: Codable, Equatable {
        let modelID: String
        let checkpointHash: String
        let dimension: Int
        let scalarType: String
        let preprocessingFingerprint: String
        let normalized: Bool
    }

    private let spec: EmbeddingModelSpec
    private let checkpointHash: String
    private let headerMagic: [UInt8] = [0x51, 0x45, 0x4D, 0x42] // "QEMB"
    private let formatVersion: UInt32 = 2
    private let recordEmbeddingSize: Int

    private let mainFileName: String
    private let journalFileName: String
    private let tombstoneFileName: String
    private let legacyFileName: String
    private let legacyBinaryFileName: String
    private let baseDir: URL

    init(spec: EmbeddingModelSpec, checkpointHash: String) {
        self.spec = spec
        self.checkpointHash = checkpointHash
        self.recordEmbeddingSize = spec.embeddingDimension * MemoryLayout<Float32>.size
        let baseName = "imageEmbedding.\(spec.modelID).v2"
        self.mainFileName = "\(baseName).qemb"
        self.journalFileName = "\(baseName)_journal.qemb"
        self.tombstoneFileName = "\(baseName)_tombstones.txt"
        self.legacyFileName = "imageEmbedding"
        self.legacyBinaryFileName = "imageEmbedding.qemb"
        self.baseDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    // MARK: - Load

    /// Load all embeddings. Tries new binary format first, falls back to legacy NSKeyedArchiver.
    /// Returns nil if no data exists.
    func loadAll() -> [String: MLMultiArray]? {
        let mainPath = baseDir.appendingPathComponent(mainFileName)
        let journalPath = baseDir.appendingPathComponent(journalFileName)

        if FileManager.default.fileExists(atPath: mainPath.path) ||
           FileManager.default.fileExists(atPath: journalPath.path) {
            return loadFromBinaryFormat()
        } else if spec.modelID == EmbeddingModelSpec.mobileCLIPS2.modelID,
                  let embeddings = loadLegacyS2() {
            if saveAll(embeddings) {
                try? FileManager.default.removeItem(at: baseDir.appendingPathComponent(legacyFileName))
                try? FileManager.default.removeItem(at: baseDir.appendingPathComponent(legacyBinaryFileName))
            }
            return embeddings
        }

        return nil
    }

    /// Load from the new binary format (main file + journal - tombstones).
    private func loadFromBinaryFormat() -> [String: MLMultiArray]? {
        let startTime = Date()
        var embeddings = [String: MLMultiArray]()

        let mainPath = baseDir.appendingPathComponent(mainFileName)
        guard let mainData = try? Data(contentsOf: mainPath),
              let header = readAndValidateHeader(mainData) else { return nil }
        guard readRecordsFromBinary(mainData, startingAt: header.recordsOffset, into: &embeddings) else {
            return nil
        }

        // Load journal (incremental additions)
        let journalPath = baseDir.appendingPathComponent(journalFileName)
        if let journalData = try? Data(contentsOf: journalPath) {
            guard readRecordsFromBinary(journalData, startingAt: 0, into: &embeddings) else { return nil }
        }

        // Apply tombstones (deletions)
        let tombstones = loadTombstones()
        for id in tombstones {
            embeddings.removeValue(forKey: id)
        }

        print("[EmbeddingStore] Loaded \(embeddings.count) embeddings in \(String(format: "%.3f", Date().timeIntervalSince(startTime)))s")
        return embeddings.isEmpty ? nil : embeddings
    }

    /// Import untagged indexes only for the model that was hard-coded by earlier releases.
    private func loadLegacyS2() -> [String: MLMultiArray]? {
        let binaryPath = baseDir.appendingPathComponent(legacyBinaryFileName)
        if let data = try? Data(contentsOf: binaryPath),
           data.count >= 12,
           Array(data[0..<4]) == headerMagic,
           readUInt32(data, at: 4) == 1 {
            var embeddings = [String: MLMultiArray]()
            guard readRecordsFromBinary(data, startingAt: 12, into: &embeddings) else { return nil }
            return embeddings.isEmpty ? nil : embeddings
        }

        let filePath = baseDir.appendingPathComponent(legacyFileName)
        do {
            let startTime = Date()
            let data = try Data(contentsOf: filePath)
            let decoded = try NSKeyedUnarchiver.unarchivedArrayOfObjects(
                ofClasses: [Embedding.self, MLMultiArray.self, NSString.self],
                from: data
            ) as? [Embedding]

            var embeddings = [String: MLMultiArray]()
            for emb in decoded ?? [] {
                if let id = emb.id, let embedding = emb.embedding {
                    guard embedding.dataType == .float32,
                          embedding.count == spec.embeddingDimension else {
                        print("[EmbeddingStore] Rejected legacy embedding with unexpected dimensions")
                        return nil
                    }
                    embeddings[id] = embedding
                }
            }

            print("[EmbeddingStore] Loaded \(embeddings.count) legacy embeddings in \(String(format: "%.3f", Date().timeIntervalSince(startTime)))s")
            return embeddings.isEmpty ? nil : embeddings
        } catch {
            print("[EmbeddingStore] Failed to load legacy format: \(error)")
            return nil
        }
    }

    // MARK: - Save

    /// Full save: write all embeddings to the main file, clear journal and tombstones.
    @discardableResult
    func saveAll(_ embeddings: [String: MLMultiArray]) -> Bool {
        guard isValid(embeddings) else { return false }
        let startTime = Date()
        let mainPath = baseDir.appendingPathComponent(mainFileName)

        guard var data = try? makeHeader(recordCount: UInt64(embeddings.count)) else { return false }

        // Records
        for (id, mlArray) in embeddings {
            appendRecord(id: id, mlArray: mlArray, to: &data)
        }

        do {
            try data.write(to: mainPath, options: .atomic)
            // Clear journal and tombstones after full save
            clearJournal()
            clearTombstones()
            print("[EmbeddingStore] Saved \(embeddings.count) embeddings in \(String(format: "%.3f", Date().timeIntervalSince(startTime)))s")
            return true
        } catch {
            print("[EmbeddingStore] Failed to save: \(error)")
            return false
        }
    }

    /// Incremental save: append new embeddings to the journal file.
    /// Also removes these IDs from tombstones so re-indexed photos survive restart.
    @discardableResult
    func appendNew(_ newEmbeddings: [String: MLMultiArray]) -> Bool {
        guard !newEmbeddings.isEmpty else { return true }
        guard isValid(newEmbeddings) else { return false }

        // Scrub re-indexed IDs from tombstones to prevent stale deletions on restart
        removeTombstones(for: Set(newEmbeddings.keys))

        let mainPath = baseDir.appendingPathComponent(mainFileName)
        if !FileManager.default.fileExists(atPath: mainPath.path) {
            guard let header = try? makeHeader(recordCount: 0) else { return false }
            do {
                try header.write(to: mainPath, options: .atomic)
            } catch {
                print("[EmbeddingStore] Failed to initialize index header: \(error)")
                return false
            }
        }

        let journalPath = baseDir.appendingPathComponent(journalFileName)

        var data = Data()
        for (id, mlArray) in newEmbeddings {
            appendRecord(id: id, mlArray: mlArray, to: &data)
        }

        do {
            if FileManager.default.fileExists(atPath: journalPath.path) {
                let handle = try FileHandle(forWritingTo: journalPath)
                defer { handle.closeFile() }
                handle.seekToEndOfFile()
                handle.write(data)
            } else {
                try data.write(to: journalPath, options: .atomic)
            }
            guard incrementHeaderRecordCount(by: UInt64(newEmbeddings.count)) else { return false }
            print("[EmbeddingStore] Appended \(newEmbeddings.count) embeddings to journal")
            return true
        } catch {
            print("[EmbeddingStore] Failed to append: \(error)")
            return false
        }
    }

    /// Mark embeddings as deleted by adding to tombstone file.
    @discardableResult
    func markDeleted(_ deletedIds: [String]) -> Bool {
        guard !deletedIds.isEmpty else { return true }

        let tombstonePath = baseDir.appendingPathComponent(tombstoneFileName)
        let content = deletedIds.joined(separator: "\n") + "\n"

        do {
            guard let contentData = content.data(using: .utf8) else { return false }
            if FileManager.default.fileExists(atPath: tombstonePath.path) {
                let handle = try FileHandle(forWritingTo: tombstonePath)
                defer { handle.closeFile() }
                handle.seekToEndOfFile()
                handle.write(contentData)
            } else {
                try content.write(to: tombstonePath, atomically: true, encoding: .utf8)
            }
            return true
        } catch {
            print("[EmbeddingStore] Failed to write tombstones: \(error)")
            return false
        }
    }

    /// Compact: rewrite the main file from in-memory dict, clearing journal and tombstones.
    @discardableResult
    func compact(_ embeddings: [String: MLMultiArray]) -> Bool {
        return saveAll(embeddings)
    }

    /// Check if journal + tombstones warrant compaction.
    func needsCompaction() -> Bool {
        let journalPath = baseDir.appendingPathComponent(journalFileName)
        let tombstonePath = baseDir.appendingPathComponent(tombstoneFileName)

        let journalSize = (try? FileManager.default.attributesOfItem(atPath: journalPath.path)[.size] as? Int) ?? 0
        let hasTombstones = FileManager.default.fileExists(atPath: tombstonePath.path)

        return journalSize > 5_000_000 || hasTombstones
    }

    // MARK: - Binary Format Helpers

    private func makeHeader(recordCount: UInt64) throws -> Data {
        let metadata = HeaderMetadata(
            modelID: spec.modelID,
            checkpointHash: checkpointHash,
            dimension: spec.embeddingDimension,
            scalarType: "float32",
            preprocessingFingerprint: spec.preprocessingFingerprint,
            normalized: spec.normalizeEmbeddings
        )
        let metadataData = try JSONEncoder().encode(metadata)
        guard metadataData.count <= Int(UInt32.max) else { throw CocoaError(.fileWriteTooLarge) }

        var data = Data(headerMagic)
        var version = formatVersion
        var count = recordCount
        var metadataLength = UInt32(metadataData.count)
        data.append(Data(bytes: &version, count: MemoryLayout<UInt32>.size))
        data.append(Data(bytes: &count, count: MemoryLayout<UInt64>.size))
        data.append(Data(bytes: &metadataLength, count: MemoryLayout<UInt32>.size))
        data.append(metadataData)
        return data
    }

    private func readAndValidateHeader(_ data: Data) -> (recordsOffset: Int, recordCount: UInt64)? {
        guard data.count >= 24,
              Array(data[0..<4]) == headerMagic,
              readUInt32(data, at: 4) == formatVersion,
              let recordCount = readUInt64(data, at: 8),
              let metadataLength = readUInt32(data, at: 16) else { return nil }
        let recordsOffset = 20 + Int(metadataLength)
        guard recordsOffset <= data.count,
              let metadata = try? JSONDecoder().decode(
                HeaderMetadata.self,
                from: data[20..<recordsOffset]
              ),
              metadata == HeaderMetadata(
                modelID: spec.modelID,
                checkpointHash: checkpointHash,
                dimension: spec.embeddingDimension,
                scalarType: "float32",
                preprocessingFingerprint: spec.preprocessingFingerprint,
                normalized: spec.normalizeEmbeddings
              ) else { return nil }
        return (recordsOffset, recordCount)
    }

    private func incrementHeaderRecordCount(by increment: UInt64) -> Bool {
        let mainPath = baseDir.appendingPathComponent(mainFileName)
        guard let data = try? Data(contentsOf: mainPath),
              let header = readAndValidateHeader(data),
              header.recordCount <= UInt64.max - increment else { return false }
        var count = header.recordCount + increment
        do {
            let handle = try FileHandle(forWritingTo: mainPath)
            defer { try? handle.close() }
            try handle.seek(toOffset: 8)
            try handle.write(contentsOf: Data(bytes: &count, count: MemoryLayout<UInt64>.size))
            return true
        } catch {
            print("[EmbeddingStore] Failed to update record count: \(error)")
            return false
        }

        enum EmbeddingStoreError: Error {
            case notReady
            case writeFailed
        }
    }

    private func isValid(_ embeddings: [String: MLMultiArray]) -> Bool {
        embeddings.allSatisfy {
            $0.key.utf8.count <= Int(UInt16.max) &&
            $0.value.dataType == .float32 &&
            $0.value.count == spec.embeddingDimension
        }
    }

    private func appendRecord(id: String, mlArray: MLMultiArray, to data: inout Data) {
        let idBytes = Array(id.utf8)
        var idLen = UInt16(idBytes.count)
        data.append(Data(bytes: &idLen, count: 2))
        data.append(contentsOf: idBytes)

        // Write embedding as raw Float32 bytes
        let shaped = MLShapedArray<Float32>(converting: mlArray)
        let scalars = shaped.scalars
        let norm = sqrt(vDSP.sumOfSquares(scalars))
        let values = spec.normalizeEmbeddings && norm > 1e-8
            ? scalars.map { $0 / norm }
            : scalars
        values.withUnsafeBufferPointer { ptr in
            data.append(UnsafeBufferPointer(start: ptr.baseAddress, count: ptr.count))
        }
    }

    private func readRecordsFromBinary(
        _ data: Data,
        startingAt startOffset: Int,
        into embeddings: inout [String: MLMultiArray]
    ) -> Bool {
        var offset = startOffset
        while offset < data.count {
            guard offset + 2 <= data.count else { return false }
            var idLen: UInt16 = 0
            _ = withUnsafeMutableBytes(of: &idLen) { dest in
                data.copyBytes(to: dest, from: offset..<(offset + 2))
            }
            offset += 2

            guard offset + Int(idLen) + recordEmbeddingSize <= data.count else { return false }
            let idData = data[offset..<(offset + Int(idLen))]
            guard let id = String(data: idData, encoding: .utf8) else { return false }
            offset += Int(idLen)

            var floats = [Float32](repeating: 0, count: spec.embeddingDimension)
            floats.withUnsafeMutableBufferPointer { dest in
                _ = data.copyBytes(to: UnsafeMutableRawBufferPointer(dest), from: offset..<(offset + recordEmbeddingSize))
            }
            offset += recordEmbeddingSize

            let shaped = MLShapedArray<Float32>(scalars: floats, shape: [1, spec.embeddingDimension])
            embeddings[id] = MLMultiArray(shaped)
        }
        return true
    }

    private func readUInt32(_ data: Data, at offset: Int) -> UInt32? {
        guard offset + MemoryLayout<UInt32>.size <= data.count else { return nil }
        return data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self) }
    }

    private func readUInt64(_ data: Data, at offset: Int) -> UInt64? {
        guard offset + MemoryLayout<UInt64>.size <= data.count else { return nil }
        return data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt64.self) }
    }

    private func loadTombstones() -> Set<String> {
        let tombstonePath = baseDir.appendingPathComponent(tombstoneFileName)
        guard let content = try? String(contentsOf: tombstonePath, encoding: .utf8) else {
            return []
        }
        return Set(content.components(separatedBy: .newlines).filter { !$0.isEmpty })
    }

    private func clearJournal() {
        let journalPath = baseDir.appendingPathComponent(journalFileName)
        try? FileManager.default.removeItem(at: journalPath)
    }

    private func clearTombstones() {
        let tombstonePath = baseDir.appendingPathComponent(tombstoneFileName)
        try? FileManager.default.removeItem(at: tombstonePath)
    }

    /// Remove specific IDs from the tombstone file (e.g. when re-indexing a previously deleted photo).
    private func removeTombstones(for idsToRemove: Set<String>) {
        guard !idsToRemove.isEmpty else { return }
        let tombstonePath = baseDir.appendingPathComponent(tombstoneFileName)
        guard let content = try? String(contentsOf: tombstonePath, encoding: .utf8) else { return }

        let remaining = content.components(separatedBy: .newlines)
            .filter { !$0.isEmpty && !idsToRemove.contains($0) }

        if remaining.isEmpty {
            try? FileManager.default.removeItem(at: tombstonePath)
        } else {
            let updated = remaining.joined(separator: "\n") + "\n"
            try? updated.write(to: tombstonePath, atomically: true, encoding: .utf8)
        }
    }
}
