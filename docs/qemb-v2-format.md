# QEMB v2 storage validation

The current implementation stores Float32 vectors. Float16 persistence remains
open in issue #8. These rules describe the existing v2 format and strict loader;
they do not provide a transaction/recovery protocol.

## Bytes and identity

All integer fields and IEEE 754 Float32 bit patterns are little-endian, matching
existing files produced on supported Apple devices. The header is:

| Offset | Field |
| --- | --- |
| 0 | Four bytes `QEMB` |
| 4 | UInt32 version, exactly 2 |
| 8 | UInt64 physical record count across main file and journal |
| 16 | UInt32 UTF-8 JSON metadata byte length, at most 65,536 |
| 20 | Metadata, followed by the main records |

Metadata must match the requested model ID, checkpoint hash, dimension, scalar
type, preprocessing fingerprint, and normalization flag. Unsupported versions,
unknown scalar types, and incompatible metadata fail loading.

Each record consists of a UInt16 ID byte length, that many UTF-8 ID bytes, and
exactly `dimension` Float32 values. IDs must be nonempty and contain no newline
characters, since deletions use a newline-separated tombstone file. Values must
be finite. Zero vectors remain readable for compatibility with the app's
existing blank-entry repair path. Writes reject invalid IDs, types, dimensions,
and nonfinite values before changing any files. Normalization accumulates the
squared norm in Double to avoid overflow for large finite Float32 values.

## Count and load behavior

The count includes replaced and deleted records until compaction. For example,
two main records followed by a replacement and an addition in the journal have a
count of four, regardless of tombstones. Journal entries replace earlier values
with the same ID; tombstones are applied afterward. A successful full save or
compaction resets the count to the number of records in the new main file.

The loader reads complete records and requires their total to equal the header
count. Truncated headers/records, invalid UTF-8, nonfinite vectors, unexpected
extra records, and missing whole records reject the entire load. A valid prefix
is never returned. Rejection returns `nil` through the existing API and does not
modify or delete any source files; that API currently also uses `nil` for absent
or empty indexes.

## Remaining recovery work

Journal writes and header-count updates are separate operations. Termination
between them can leave a count mismatch, which now fails closed. Main-file
replacement and journal/tombstone cleanup are also separate operations; stale
sidecars after interrupted compaction can fail validation or replay deletions.
Generation-bound sidecars, atomic commit/recovery, distinguishable load errors,
and fault-injection/forced-termination tests remain required by issues #8/#13.
Do not treat these validation tests as evidence of crash-safe recovery.

Legacy v1 and archived indexes lack verified model provenance. This loader leaves
them untouched and requires re-indexing rather than inventing an identity.

## Verification

`EmbeddingStoreTests` uses generated vectors and isolated temporary directories;
it never touches the user's saved index. It covers byte order, restart,
replacement, deletion, compaction, counts, malformed/truncated data, invalid
writes, large finite normalization, and readable legacy blank values. Run it
using the simulator command in [the test guide](test-infrastructure-plan.md),
optionally adding `-only-testing:QueryableTests/EmbeddingStoreTests`.
