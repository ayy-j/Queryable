# Running the app tests

Status: simulator execution verified on October 9, 2026, with Xcode 27.0 and the iOS 27.0 simulator runtime.

The latest combined simulator run executed **93 tests: 90 passed, 3 GPU graph tests skipped, 0 failures**. The three graph tests skip deliberately on the simulator and ran successfully in the earlier native Mac suite: **11 tests passed, no skips or failures**. The separate opt-in performance test was deliberately excluded. Simulator result bundle: `build/image-preprocessing-final.xcresult`; log: `build/image-preprocessing-final.log`. Native log: `build/native-gpu-mutations.log` (all ignored by Git).

From the repository folder, choose a simulator returned by `xcrun simctl list devices available` and run:

```sh
TEST_RUNNER_QUERYABLE_BENCHMARK=1 xcodebuild test \
  -project Queryable/Queryable.xcodeproj \
  -scheme QueryablePerformance \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -derivedDataPath build/app-tests \
  -skip-testing:QueryableTests/PerformanceMeasurementTests \
  CODE_SIGNING_ALLOWED=NO ENABLE_TESTABILITY=YES ONLY_ACTIVE_ARCH=YES
```

The existing performance scheme supplies a test host without debugger instrumentation. The environment flag opens an empty host instead of the normal app interface, preventing competing model loading or Photos access. It does not enable the opt-in performance test because that test is excluded explicitly. `ENABLE_TESTABILITY=YES` is necessary for the Release build's `@testable` imports. No simulator signing identity is needed.

The model packaging script needs a complete local S4 model and tokenizer set. Missing resources cause a build failure; the model prediction test explicitly skips if its artifacts are absent. Check the executed and skipped counts when recording results.

## Coverage and remaining work

The new failure tests use generated images, fake photo callbacks/encoders, and temporary storage. They cover delayed/final/duplicate callbacks, offline-only missing images, cancellation before and during requests, individual encoding failures, invalid outputs, save failure/retry, and preservation of a successful 5,000-photo checkpoint. They do not read or change personal photos or the real saved index.

Nine isolated `EmbeddingStoreTests` now cover explicit byte order, incorrect physical record counts, malformed/truncated headers and records, invalid IDs/nonfinite values, rejected writes preserving existing files, restart/replacement/deletion/compaction, large finite normalization, and readable blank entries. See [the storage format](qemb-v2-format.md).

Eleven `SimilarityValidationTests` cover input validation and GPU index mutations at all four required dimensions. Eight run on the simulator without graph execution; three graph tests run natively on Mac or on a physical device. Dense CPU/GPU scores must agree within `2e-3` and rankings must match after upserts, rejected batches, removals, rebuilds, and empty transitions. Reproduce with `./tools/test-gpu-index-mutations.sh`; see [mutation evidence and limitations](gpu-index-mutations.md).

Twelve `PhotoSearchModelTests`/`SearchReliabilityTests` now cover analytic CPU cosine/ranking references at 512/768/1152/1536 dimensions (absolute tolerance `1e-6`), invalid CPU inputs, prediction failures and retry, stale-result/error cleanup, S2 fallback dimensions, and missing/successful similar-photo searches. They use generated vectors and injected saves; physical-device UI and GPU-failure recovery checks remain open.

Six further model-contract tests cover fixed-context boundaries, start/end markers, missing special tokens, zero-padding fallback, malformed token shapes, and invalid integer IDs. The existing bundled S4 prediction test now also compares a long prompt with explicit EOS-preserving token input; see [the parity guide](parity-gates.md#fixed-context-text-input). Independent tokenizer/vector parity remains open.

Five additional image-output tests exercise the shared single/batch runtime validator using synthetic feature providers at 512/768/1152 dimensions. They cover expected feature names, scalar types, vector shapes/dimensions, finite/nonzero norms, extreme finite values, exact batch counts, and rejection of invalid members without partial results. The encoder returns empty batches before predicting. These checks do not supply independent image preprocessing/model parity evidence.

Five image-input/ownership tests now cover unsupported filter/pixel/aspect contracts before loading or rendering, actual 256/384 px Core Image rendering through the shared single/batch input helper, invalid pool geometry, retained buffers through flushing, and strided output copying with independent ownership and source release. All 29 model-contract tests passed. See [the preprocessing and buffer guide](image-preprocessing-validation.md) for reproduced failures and coverage limits.

The suite still needs real Photos permission/iCloud walkthroughs, atomic interrupted storage recovery coverage, independent model parity fixtures, and physical-iOS CPU/GPU ranking comparisons. Native Mac component ranking checks now pass; Float32 graph accumulation latency and transient memory still require measurement. The installed simulator crashed inside Apple's GPU graph code in an earlier performance run; use a physical device for GPU validation. The physical iPhone benchmark is described in [the performance guide](performance-measurement.md).

Automated GitHub checks remain to be added. A future CI job must provide a supported Xcode/runtime and model resources, report missing/skipped model coverage, and fail when any executed test fails. The earlier October 8 claims that this machine had no simulator runtime or usable signing identity are obsolete. The Designed-for-iPad Mac test host stalled in later validation; the native component runner is the working Mac measurement path.
