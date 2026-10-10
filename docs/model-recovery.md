# Model generation and recovery protocol

The default remains MobileCLIP2-S4. A selection is a durable request, separate
from the active searchable index. Each request receives a fresh UUID and binds
its store, checkpoint, operation tokens and caches to the complete model spec
compatibility identity and artifact checkpoint hash. Restart makes a new UUID;
resume reuses the existing UUID. No model is silently substituted.

## Invariants and commit boundaries

`model-index-state.json` is a checksummed, versioned envelope containing active,
requested and previous references and the requested phase. Its commit is: write
an adjacent temporary file, synchronize the file, atomically rename it over the
state, then synchronize the parent directory. In-memory state changes only after
commit. A failed post-rename operation reloads the visible committed state before
reporting the error. A relaunch sees either complete old or complete new state.
Invalid JSON, checksums, schema or state invariants are explicit corruption errors;
the original file is preserved, never replaced with an empty default.

Persist the request before starting work. Persist embedding/checkpoint work before
activation. Activation accepts evidence matching the exact requested reference,
a nonzero committed store revision and a complete checkpoint. The integration
must obtain this evidence from a successfully loaded committed store and must
reconcile the current accessible Photos snapshot before declaring completion.
Changing the single coordinator file then atomically promotes requested to active
and retains the previous active reference for rollback. Stores are not removed
at activation, cancellation, restart or rollback. Unreferenced builds may be
retained; garbage collection is deliberately outside this protocol.

The coordinator is owned on the main actor by the application; operations capture
references and additionally use per-run cancellation tokens. Every asynchronous
completion must check its token and matching reference before publishing results,
writing a batch or activating. A requested rebuild does not invalidate queries
against the still-active reference. Activation/rollback invalidate active caches.
Search and similar-photo queries additionally snapshot their spec and embeddings
and re-validate epoch plus spec identity after encoding; the GPU path validates
its dimension against the captured spec before executing.

## Recovery, cancellation and migration

A persisted `building` request is exposed as `paused` on restart; work does not
resume automatically. Pause, failure and cancellation retain committed progress.
Resume explicitly returns the same request to building; restart creates a new
request and namespace. A stale reference cannot change the request phase or
activate. Rollback swaps active and previous atomically and cancels outstanding
requests. Failed state writes are surfaced to the user; work stops until retry.

Legacy compatible stores must be loaded and copied into a generation namespace
before `adoptActive` is called. Adoption is allowed only for a coordinator with no
active or requested state. Missing coordinator state is distinct from corrupt
state. An empty committed index is valid; absent index files or incomplete work
are not activation evidence. A corrupt/incompatible store requires an explicit
new rebuild, preserving original files. Changed permissions do not prove deletion;
only a sufficiently authorized reconciliation may remove absent Photos IDs.
Unavailable iCloud assets remain pending and prevent completion/activation.

S2 is not an automatic fallback: selection requires available, validated artifacts
matching its spec. Private quality benchmarks remain postponed. Model download and
SigLIP experimentation are outside this change.

## Local S2 artifact verification (2026-10-09)

Both S2 compiled towers exist in the checkout's ignored `models/` directory.
Loading both with macOS Core ML and `.cpuOnly` confirms:

| Tower | Actual input | Actual output |
| --- | --- | --- |
| ImageEncoder_mobileCLIP_s2 | `colorImage`, color image 256 × 256 | `embOutput`, Float32 [1, 512] |
| TextEncoder_mobileCLIP_s2 | `input_tokens`, Int32 [1, 77] | `text_embeddings`, Float32 [1, 512] |

The existing `mobileCLIPS2` spec instead declares `input_ids`, Float32 for the
text input. Thus the actual local S2 tower fails that spec's feature validation.
Its presence does not establish a usable fallback, and this change does not
reinterpret old vectors or silently alter the old spec identity to fit it. S2
selection must report this contract mismatch. S4 remains the working default.
This inspection establishes load and feature contracts, not retrieval quality or
physical-device residency/latency. No downloads or private quality runs occurred.

## Validation scope

Coordinator tests inject errors before file creation, after the write, after file
synchronization, after rename, and after directory synchronization. Each activation
boundary is reopened repeatedly and must recover exactly the old or new references;
request-boundary tests require the old active index to remain selected. These are
application I/O fault tests, not simulated power loss. A standalone macOS harness
also executed the production coordinator's five activation boundaries and repeated
reopens with a minimal model-spec fixture. Actual forced process termination,
device power-loss behavior and physical-device model residency remain separate
validation requirements.
