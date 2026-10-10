# QEMB storage and recovery

## Record encoding

New immutable segments retain the QEMB v2 Float32 encoding. All integers and
IEEE 754 Float32 bit patterns are little-endian:

| Offset | Field |
| --- | --- |
| 0 | Four bytes `QEMB` |
| 4 | UInt32 version, exactly 2 |
| 8 | UInt64 physical record count in this segment |
| 16 | UInt32 UTF-8 JSON metadata length, at most 65,536 |
| 20 | Metadata followed by records |

Metadata binds model ID, checkpoint hash, dimension, scalar type,
preprocessing fingerprint, and normalization. The transaction manifest also
binds the complete model compatibility identity (including tokenizer and paired
tower contracts); its directory is derived from both identities. Float16 remains
unsupported and issue #8 stays open for that acceptance criterion.

A record contains a UInt16 UTF-8 identifier length, identifier bytes, and exactly
`dimension` Float32 values. IDs must be nonempty, at most 65,535 bytes, and contain
no newline characters. Values must be finite. Zero vectors remain readable for
legacy blank-entry repair. Inputs are validated before any write. Normalization
accumulates the squared norm in Double. Counts include physical records in each
segment, including replacements; replay produces one live value per identifier.

## Transaction protocol

New writes use immutable QEMB v2 record segments and one checksummed JSON
manifest. A manifest binds full compatibility metadata, generation UUID,
revision, ordered segment references (SHA-256, physical count, deletions), and
opaque checkpoint bytes. Upserts and deletions in one segment form one operation:
deletions apply first, then upserts. Checkpoint progress is authoritative only
when committed in the same manifest as its corresponding vectors/deletions.

A writer validates inputs, writes and synchronizes a new segment, synchronizes
its directory, writes and synchronizes a temporary manifest, atomically renames
it over the manifest, then synchronizes the directory. The rename is the single
visibility/commit boundary. Interruptions before it retain the old commit;
after it recovery sees the entire new commit. Count fields never change in
place. Unreferenced files are ignored. Compaction replaces segment references
in that same transaction, so old tombstones cannot replay over the compacted
snapshot. After directory synchronization, unreferenced segments and temporary files are
best-effort deleted under the same reader/writer lock. Interruption during cleanup
can only retain extra files; it cannot remove referenced data. Legacy files and
other generation directories are never included in this cleanup.

Recovery verifies manifest checksum/version/identity, every referenced segment
checksum/header/count, then replays complete segments. Missing storage and valid
empty storage are distinct; incompatible metadata and corruption are errors.
No valid prefix or silent empty result is returned by the typed API. A failed
write before manifest rename can be retried. A failure after rename has an
uncertain acknowledgement: reload before continuing; the durable checkpoint
and revision settle whether it committed. Repeated recovery is read-only.

Build generations have separate directories under a full-identity digest.
Operations specify their generation; a mismatch is rejected. A validated old
v2 main/journal/tombstone set is imported by the first successful transaction,
without modifying legacy files. Invalid/interrupted legacy sets cannot prove a
committed boundary and remain preserved with an explicit corrupt result.
Legacy v1/archive data still requires an explicit recoverable rebuild.

## API and maintenance

`load()` returns `nil` only for missing storage. A snapshot with an empty embedding
dictionary is a valid committed empty store. It returns generation, revision,
and checkpoint bytes along with vectors. `incompatible`, `corrupt`, and
`staleGeneration` are explicit failures; ordinary I/O errors also propagate.
`commit` requires a generation and returns the committed revision. Existing
Boolean write wrappers and optional `loadAll` remain only for compatibility;
production recovery uses the typed API. All store loads and commits serialize
through a process-wide lock. The app has one process writing the store.

Old v2 main-file counts cover main plus legacy journal records. Their strict
loader requires exact counts and valid complete records and tombstones. A missing
or damaged journal cannot be safely inferred and is reported corrupt, with all
source files preserved. There is no automatic destructive salvage. A migrated
manifest takes precedence over old sidecars permanently. New generation stores
do not fall back to another generation or to legacy storage.

Compaction writes a new immutable full snapshot and swaps only its manifest
references; checkpoint and generation are preserved. After a successful commit, unreachable files in that generation are reclaimed
under the process-wide lock. Failed cleanup may leave harmless extra files until
the next successful commit. Retained legacy sources and separate build generations
are not reclaimed automatically.
Immutable segment checksums are SHA-256; manifests contain a checksummed payload.
This detects accidental damage, not malicious filesystem modification.

## Verification and limits

`EmbeddingStoreTests` uses generated vectors in temporary directories. It injects
failure after segment write/sync/directory sync, manifest write/sync/rename, and
final directory sync, and orphan cleanup. It exercises append plus checkpoint/deletion, tombstone
replacement, and compaction at every boundary and repeatedly reopens each state.
Other cases cover generations, stale writers, invalid writes, corrupt/truncated
segments, migration without source deletion, counts, byte order, and large finite
normalization. The process continues during these deterministic tests; actual
forced process termination and device filesystem/power-loss testing remain
separate manual checks. Directory fsync and FileHandle synchronization establish
the intended durability ordering; simulated exceptions are not physical power
failure evidence.
