# Running the app tests

Status: simulator execution verified on October 9, 2026, with Xcode 27.0 and the iOS 27.0 simulator runtime.

The full ordinary suite now executes. The latest run passed **61 tests with no failures or skips**, including the bundled S4 text prediction, photo-request callback/cancellation tests, indexing failure/retry tests, an isolated journal round trip, and the existing model/tokenizer/parity-metric checks. The separate opt-in performance test was deliberately excluded. Result bundle: `build/search-validation.xcresult`; log: `build/search-validation.log` (both ignored by Git).

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

Seven `SimilarityValidationTests` cover input validation at all four required dimensions, including incompatible strides/types/shapes, nonfinite and zero/near-zero vectors, large finite normalization, and rejection through the empty GPU index API. See [search validation scope](search-vector-validation.md). These tests do not execute GPU ranking graphs.

The suite still needs real Photos permission/iCloud walkthroughs, atomic interrupted storage recovery coverage, independent model parity fixtures, and CPU/GPU ranking comparisons. The installed simulator crashed inside Apple's GPU graph code in an earlier performance run; use a physical device for GPU validation. The physical iPhone benchmark is described in [the performance guide](performance-measurement.md).

Automated GitHub checks remain to be added. A future CI job must provide a supported Xcode/runtime and model resources, report missing/skipped model coverage, and fail when any executed test fails. The earlier October 8 claims that this machine had no simulator runtime or usable signing identity are obsolete. The Designed-for-iPad Mac test host stalled in later validation; the native component runner is the working Mac measurement path.
