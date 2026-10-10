# Indexing and activation recovery protocol

Design recorded before implementation, October 9, 2026. See the storage format
and model coordinator documents for their exact durable encodings.

## Invariants and commit boundaries

* An index belongs to the complete model compatibility identity, artifact hash,
  and one build generation. Asynchronous work captures an operation epoch and
  may mutate or publish only while that epoch is current.
* Storage commits vectors, deletions, and a checkpoint in one manifest replacement.
  Successful IDs are the committed records, never a next-photo offset. Failed or
  unavailable photos have no successful record and remain eligible on resume.
* A checkpoint records the required library IDs and observed progress. Resume
  takes a fresh Photos snapshot and reconciles it with committed records. Missing
  IDs are deleted only with full library authorization; limited access hides
  inaccessible results without declaring the underlying photos deleted.
* A replacement generation has independent storage and progress. The active
  generation stays searchable until replacement work is committed, the library
  has been reconciled again, and model validation succeeds. Only then does one
  coordinator state replacement change active/requested/previous references.
* A crash after index completion but before activation leaves a resumable target
  and the old active index. A crash after activation loads the new active pointer.
* Pause/cancel invalidate outstanding work and release phase resources. Committed
  batches survive; unfinished requests may be repeated. Cancel retains the target
  for explicit resume; restart creates a new generation; rollback validates the
  retained previous generation before switching the durable pointer.
* Storage errors never mean an empty library. Corruption/incompatibility is shown
  explicitly; original files remain available and rebuild is an explicit action.
* Active-index mutations (edited-photo invalidation, library reconciliation,
  incremental saves, compaction) commit through the typed store API against the
  tracked active generation and reload the committed revision before publishing
  in memory. The legacy Boolean wrappers remain only for compatibility tests.
* Same-model resume carries the committed checkpoint bytes (required IDs,
  versions, completion) into the new generation, not just version dates.
* A damaged coordinator file is never auto-replaced: startup surfaces it and
  waits for the explicit Repair action, which quarantines only the corrupt
  manifest beside the retained indexes.

## Lifecycle and Photos policy

Image requests and indexing are cancellable. Check epoch after every suspension
before accepting results. Text and similar-photo queries capture their model spec,
embedding snapshot, and epoch before encoding; after every suspension they
re-check the epoch and spec identity before publishing, so a model switch cannot
mix one generation's query with another generation's index. Ranking validates the
GPU index dimension against the captured spec and falls back to the captured CPU
snapshot on mismatch. Image tower ownership ends on every build exit; text tower
is acquired for text queries and released afterward. Similar-photo search uses
stored vectors.
Backgrounding, low-power mode, or serious/critical thermal state pauses work at a
safe boundary. Batch size remains 32 pending device measurement; this policy is
conservative and does not claim a tuned throughput or energy budget.

Library changes during a build invalidate its snapshot. Reconciliation runs
before final activation; new/unavailable photos leave it incomplete, removed
photos cannot be committed by late work. Permission revocation pauses instead of
interpreting an empty result as mass deletion. Actual process-kill and device
Photos/iCloud/resource measurements remain separate from injected restart tests.

## Migration

Validated model-aware v2 indexes are imported non-destructively. Legacy v1 and
archives lack provenance and are preserved for explicit rebuild. Interrupted
legacy sidecars with ambiguous ordering cannot establish a safe commit boundary:
report corruption and retain their bytes rather than guess which writes committed.
S4 stays the app default. S2 is selectable only if actual installed artifacts pass
its declared contract; there is no automatic substitution or model download.
Private quality benchmarks and SigLIP experiments remain postponed.
