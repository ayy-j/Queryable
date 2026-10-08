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

enum EmbeddingStoreError: Error {
    /// The store has not been created for the active model spec yet.
    case notReady
    /// The append-only journal could not accept the new embeddings.
    case writeFailed
    /// The header metadata does not fit the on-disk length field.
    case metadataTooLarge(Int)
}

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
    private let baseDir: URL

    init(spec: EmbeddingModelSpec, checkpointHash: String) {
        self.spec = spec
        self.checkpointHash = checkpointHash
        self.recordEmbeddingSize = spec.embeddingDimension * MemoryLayout<Float32>.size
        let baseName = "imageEmbedding.\(spec.modelID).\(checkpointHash).v2"
        self.mainFileName = "\(baseName).qemb"
        self.journalFileName = "\(baseName)_journal.qemb"
        self.tombstoneFileName = "\(baseName)_tombstones.txt"
        self.baseDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    // MARK: - Load

    /// Load embeddings only when their model and checkpoint metadata match this store.
    /// Returns nil if no data exists.
    func loadAll() -> [String: MLMultiArray]? {
        let mainPath = baseDir.appendingPathComponent(mainFileName)
        let journalPath = baseDir.appendingPathComponent(journalFileName)

        if FileManager.default.fileExists(atPath: mainPath.path) ||
           FileManager.default.fileExists(atPath: journalPath.path) {
            return loadFromBinaryFormat()
        }

        return nil
    }

    /// Load from the new binary format (main file + journal - tombstones).
    private func loadFromBinaryFormat() -> [String: MLMultiArray]? {
        let startTime = Date()
        var embeddings = [String: MLMultiArray]()

        let mainPath = baseDir.appendingPathComponent(mainFileName)
        guard let mainHandle = try? FileHandle(forReadingFrom: mainPath) else { return nil }
        defer { try? mainHandle.close() }
        guard let header = try? readHeader(from: mainHandle),
              readRecordsFromBinary(mainHandle, startingAt: header.recordsOffset, into: &embeddings) else {
            return nil
        }

        // Load journal (incremental additions)
        let journalPath = baseDir.appendingPathComponent(journalFileName)
        if FileManager.default.fileExists(atPath: journalPath.path) {
            guard let journalHandle = try? FileHandle(forReadingFrom: journalPath) else { return nil }
            defer { try? journalHandle.close() }
            guard readRecordsFromBinary(journalHandle, startingAt: 0, into: &embeddings) else { return nil }
        }

        // Apply tombstones (deletions)
        let tombstones = loadTombstones()
        for id in tombstones {
            embeddings.removeValue(forKey: id)
        }

        print("[EmbeddingStore] Loaded \(embeddings.count) embeddings in \(String(format: "%.3f", Date().timeIntervalSince(startTime)))s")
        return embeddings.isEmpty ? nil : embeddings
    }

    // MARK: - Save

    /// Full save: write all embeddings to the main file, clear journal and tombstones.
    @discardableResult
    func saveAll(_ embeddings: [String: MLMultiArray]) -> Bool {
        guard isValid(embeddings) else { return false }
        let startTime = Date()
        let mainPath = baseDir.appendingPathComponent(mainFileName)

        guard let header = try? makeHeader(recordCount: UInt64(embeddings.count)) else { return false }
        let temporaryPath = baseDir.appendingPathComponent("\(mainFileName).\(UUID().uuidString).tmp")
        guard FileManager.default.createFile(atPath: temporaryPath.path, contents: nil),
              let handle = try? FileHandle(forWritingTo: temporaryPath) else { return false }
        do {
            try handle.write(contentsOf: header)
            for (id, mlArray) in embeddings {
                try handle.write(contentsOf: recordData(id: id, mlArray: mlArray))
            }
            try handle.synchronize()
            try handle.close()

            if FileManager.default.fileExists(atPath: mainPath.path) {
                _ = try FileManager.default.replaceItemAt(mainPath, withItemAt: temporaryPath)
            } else {
                try FileManager.default.moveItem(at: temporaryPath, to: mainPath)
            }

            // Clear journal and tombstones after full save
            clearJournal()
            clearTombstones()
            print("[EmbeddingStore] Saved \(embeddings.count) embeddings in \(String(format: "%.3f", Date().timeIntervalSince(startTime)))s")
            return true
        } catch {
            try? handle.close()
            try? FileManager.default.removeItem(at: temporaryPath)
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
            data.append(recordData(id: id, mlArray: mlArray))
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
            scalarType: spec.storageScalarType.rawValue,
            preprocessingFingerprint: spec.preprocessingFingerprint,
            normalized: spec.normalizeEmbeddings
        )
        let metadataData = try JSONEncoder().encode(metadata)
        guard metadataData.count <= Int(UInt32.max) else { throw EmbeddingStoreError.metadataTooLarge(metadataData.count) }

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
        guard data.count >= 20,
              Array(data[0..<4]) == headerMagic,
              readUInt32(data, at: 4) == formatVersion,
              let recordCount = readUInt64(data, at: 8),
              let metadataLength = readUInt32(data, at: 16) else { return nil }
        guard metadataLength <= 65_536 else { return nil }
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
                scalarType: spec.storageScalarType.rawValue,
                preprocessingFingerprint: spec.preprocessingFingerprint,
                normalized: spec.normalizeEmbeddings
              ) else { return nil }
        return (recordsOffset, recordCount)
    }

    private func incrementHeaderRecordCount(by increment: UInt64) -> Bool {
        let mainPath = baseDir.appendingPathComponent(mainFileName)
        guard let handle = try? FileHandle(forUpdating: mainPath) else { return false }
        defer { try? handle.close() }
        guard let header = try? readHeader(from: handle),
              header.recordCount <= UInt64.max - increment else { return false }
        var count = header.recordCount + increment
        do {
            try handle.seek(toOffset: 8)
            try handle.write(contentsOf: Data(bytes: &count, count: MemoryLayout<UInt64>.size))
            return true
        } catch {
            print("[EmbeddingStore] Failed to update record count: \(error)")
            return false
        }
    }

    private func isValid(_ embeddings: [String: MLMultiArray]) -> Bool {
        embeddings.allSatisfy {
            $0.key.utf8.count <= Int(UInt16.max) &&
            $0.value.dataType == .float32 &&
            $0.value.count == spec.embeddingDimension
        }
    }

    private func recordData(id: String, mlArray: MLMultiArray) -> Data {
        var data = Data()
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
        return data
    }

    private func readRecordsFromBinary(
        _ handle: FileHandle,
        startingAt startOffset: Int,
        into embeddings: inout [String: MLMultiArray]
    ) -> Bool {
        do {
            try handle.seek(toOffset: UInt64(startOffset))
            while true {
                let idLengthData = try readExactly(2, from: handle)
                if idLengthData.isEmpty { break }
                guard idLengthData.count == 2,
                      let idLen = readUInt16(idLengthData, at: 0) else { return false }
                let idData = try readExactly(Int(idLen), from: handle)
                guard idData.count == Int(idLen),
                      let id = String(data: idData, encoding: .utf8) else { return false }
                let embeddingData = try readExactly(recordEmbeddingSize, from: handle)
                guard embeddingData.count == recordEmbeddingSize else { return false }
                var floats = [Float32](repeating: 0, count: spec.embeddingDimension)
                floats.withUnsafeMutableBufferPointer { dest in
                    _ = embeddingData.copyBytes(to: UnsafeMutableRawBufferPointer(dest))
                }

                let shaped = MLShapedArray<Float32>(scalars: floats, shape: [1, spec.embeddingDimension])
                embeddings[id] = MLMultiArray(shaped)
            }
            return true
        } catch {
            return false
        }
    }

    private func readHeader(from handle: FileHandle) throws -> (recordsOffset: Int, recordCount: UInt64) {
        try handle.seek(toOffset: 0)
        let fixedHeader = try readExactly(20, from: handle)
        guard fixedHeader.count == 20,
              let metadataLength = readUInt32(fixedHeader, at: 16),
              metadataLength <= 65_536 else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let metadata = try readExactly(Int(metadataLength), from: handle)
        guard metadata.count == Int(metadataLength) else {
            throw CocoaError(.fileReadCorruptFile)
        }

        var headerData = fixedHeader
        headerData.append(metadata)
        guard let header = readAndValidateHeader(headerData) else {
            throw CocoaError(.fileReadCorruptFile)
        }
        return header
    }

    private func readExactly(_ count: Int, from handle: FileHandle) throws -> Data {
        var result = Data()
        while result.count < count {
            guard let chunk = try handle.read(upToCount: count - result.count), !chunk.isEmpty else {
                break
            }
            result.append(chunk)
        }
        return result
    }

    private func readUInt16(_ data: Data, at offset: Int) -> UInt16? {
        guard offset + MemoryLayout<UInt16>.size <= data.count else { return nil }
        return data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt16.self) }
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
