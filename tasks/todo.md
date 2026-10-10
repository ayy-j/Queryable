# Queryable execution checklist

Companion to the [plain-language plan](/Users/abackman/Queryable/tasks/plan.md). Unchecked items are remaining work. Existing GitHub issues remain the source of requirements; this checklist groups them into manageable changes without creating duplicate issues.

## First: trustworthy indexing

### 1. Establish a reliable test run

- [x] Run the full app test suite on a configured simulator: 45 ordinary tests passed on October 9, including bundled S4 text prediction; no skips. The opt-in performance test was excluded explicitly.
- [ ] Add automatic checks to future proposed changes and update the outdated test-setup document.

**Verify:** full tests execute, rather than merely compile; a deliberately failing test makes the automatic check fail. The iOS 27 simulator now executes the ordinary suite; the setup document records the working command and coverage limits. Code Review reported no checks for merged PRs #26 and #28.

**Depends on:** none. **Scope:** medium. **Likely files:** new `.github/workflows/ios-tests.yml`, project settings, `docs/test-infrastructure-plan.md`.

### 2. Wait for usable photos and preserve retry opportunities

- [x] Wait for the completed photo request, handling preview images, cancellation, missing local images, and multiple callbacks correctly; regression tests pass.
- [x] Save valid successful image results only and keep failed/unavailable photos retryable. Existing blank/nonfinite entries are included in the next indexing attempt, with a visible repair count; existing disk entries remain until replaced. Automated failure/retry tests pass.
- [ ] Complete real-device Photos/iCloud/permission and legacy-index repair walkthroughs.

**Verify:** delayed and repeated callbacks finish exactly once; a failed request adds no successful index entry; retry succeeds. Test offline iCloud-only photos and changed Photos permissions.

**Depends on:** 1 for automated verification. **Issues:** #7, #13. **Scope:** medium. **Likely files:** `CachedImageManager.swift`, `PhotoSearcher.swift`, new photo-loading tests.

### 3. Show truthful progress and failure states

- [x] Distinguish photos checked, saved entries, entries waiting to save, and photos needing retry in the indexing screen.
- [x] Keep failed saves in a recoverable error state with unsaved entries retained in memory; show “finished” only after the final successful save and no remaining failed photos. Tests include failure after a successful 5,000-photo checkpoint.
- [ ] Verify the new counts and recovery controls with a small real photo set on phone and Mac.

**Verify:** simulated photo and disk failures never report full success; totals remain accurate after a retry. Verify the same behavior in the app interface.

**Depends on:** 2. **Issues:** #13, #20. **Scope:** medium. **Likely files:** `PhotoSearcher.swift`, `BuildIndexView.swift`, new indexing-state tests.

**Checkpoint:** a small known photo set produces nonblank search data, useful text/similar-photo results, and honest completion counts. The next user-provided device log records 21,793 entries appended and compacted successfully, followed by completed GPU searches. This supplies a real-library save/search observation; relaunch recovery, result quality, permissions/iCloud behavior, and the recovery interface still need explicit checks.

## Next: saved work and safe transitions

### 4. Protect saved work through interruption

**October 9 storage-validation slice:** added strict physical record-count validation, explicit little-endian fields, invalid ID/nonfinite-value rejection, and nine isolated storage tests covering restart, replacement, deletion, compaction, and corruption. All 54 ordinary simulator tests passed (`build/storage-validation.xcresult`); the opt-in performance test was excluded. See [format and recovery limits](../docs/qemb-v2-format.md). Atomic crash recovery and generation-bound sidecars remain open; this does not complete item 4 or issue #8.

**October 9 recovery-hardening follow-up (uncommitted):** active-index mutations (edited-photo invalidation, library reconciliation, incremental saves, compaction) now commit through the typed store API against the tracked active generation and reload the committed revision before publishing in memory; same-model resume carries committed checkpoint bytes into the new generation; search/similar-photo capture spec plus embedding snapshots and re-validate epoch and spec identity after encoding with a GPU-dimension check; a damaged coordinator file surfaces explicitly and waits for the Repair action (quarantines only the corrupt manifest). Three new integration tests cover typed active commits, stale-search suppression, and corrupt-coordinator repair. Full suite: 113 tests pass, 3 GPU-only skips (`build/full-recovery-final.xcresult`).

- [ ] Test and fix save/load, additions, deletions, damaged files, record counts, and interrupted saves; preserve the last valid data.
- [ ] Resume only successfully committed work, including interruption while combining saved batches and deleting photos.

**Verify:** terminate/relaunch at multiple save boundaries; run damaged-file, low-storage, and write-failure tests. No lost or duplicated successful work, and no partially accepted corrupt index. Define recovery and file-format rules explicitly.

**Depends on:** 1, 3. **Issues:** #8, #13. **Scope:** medium, delivered as separate save/recovery changes. **Likely files:** `EmbeddingStore.swift`, `PhotoSearcher.swift`, new storage/recovery tests, storage-format documentation.

### 5. Validate search and handle model errors safely

**October 9 image preprocessing and buffer follow-up (#7):** unsupported filters now fail before model loading or Core Image KVC, and single/batch input construction uses one validated helper. Pool geometry rejects invalid numeric sizes, flushing requests all unused buffers, and detachment copies logical scalars instead of padding from strided arrays. Five new tests cover actual 256/384 px rendering, invalid configurations/sizes, pool isolation/flush safety, and independent output ownership at 512/768/1152 dimensions. Both filter-validation and strided-copy regressions were reproduced before their fixes. All 90 ordinary simulator tests passed with 3 GPU-only skips and no failures (`build/image-preprocessing-final.xcresult`). See [coverage and remaining parity/device work](../docs/image-preprocessing-validation.md).

- [x] Validate supported image preprocessing before loading/rendering and cover model-sized buffers, pool flushing, and detached output ownership with generated inputs (#7).

**Concurrent agent follow-up (#7/#10):** shared image-output checks now reject malformed single/batch outputs, require exact batch counts, and return empty batches immediately; five synthetic-provider tests cover 512/768/1152 dimensions. GPU updates now replace existing IDs rather than duplicating rows, with mutation/state and native dense cosine/ranking tests across all four dimensions. Those tests exposed Float16 accumulation drift; Float32 graph arithmetic fixes it while retaining Float16 storage. Final verification: 85 simulator tests passed, 3 native/device graph tests skipped there, and all 11 native GPU tests passed without skips; no failures (`build/concurrent-encoder-gpu-final.xcresult`, `build/native-gpu-mutations.log`). See [GPU evidence and remaining memory/latency work](../docs/gpu-index-mutations.md). Physical iOS ranking, model identity, independent parity, and app-level acceptance remain open.

**October 9 GPU-validation slice:** build/add/query paths now reject wrong types/shapes, noncontiguous Float32 layouts, nonfinite values, and zero/near-zero norms before pointer access or graph execution. Double normalization avoids overflow on large finite vectors. Seven new tests cover 512/768/1152/1536 dimensions; all 61 ordinary simulator tests passed with no skips (`build/search-validation.xcresult`), excluding the opt-in performance test. See [scope and remaining work](../docs/search-vector-validation.md). The CPU fallback and text-error follow-up is recorded below; model identity and device ranking comparisons remain open, so item 5 and issue #10 are incomplete.

- [x] Reject invalid numbers, wrong shapes, unsupported memory layouts, and unusable blank search data before ranking; handle text-model errors without crashing.

**October 9 CPU/error follow-up:** validated CPU scoring and throwing text prediction now feed shared text/similar-photo ranking. Both phone and Mac views display failures, stale results are cleared, retries clear errors, and similar-photo failures stop the spinner. Twelve new tests cover analytic CPU cosine/ranking references at all four dimensions, malformed data, prediction failures, and state recovery. All 73 ordinary simulator tests passed with no skips (`build/search-reliability-final.xcresult`), excluding the opt-in performance test. Device GPU comparisons, backend-failure recovery, independent tokenizer parity and UI walkthroughs remain outstanding.
- [ ] Compare GPU and CPU rankings at every required model size, including additions, deletions, empty indexes, and similar-photo searches.

**October 9 fixed-context follow-up (#7):** long prompts now retain BOS/EOS and fit the declared context; short prompts retain EOS before padding. Malformed token shapes/IDs fail before unsafe array construction. Six new boundary/error tests and an expanded real S4 prediction test passed in the 79-test ordinary simulator run (`build/text-context.xcresult`), with no skips; the opt-in performance test was excluded. Raw-token fixtures remain unchanged. See [source rule and coverage limits](../docs/parity-gates.md#fixed-context-text-input); independent tokenizer/vector parity remains open.

**Verify:** known expected rankings match within declared tolerance; broken inputs produce an error rather than arbitrary results or a crash. Inspect long-search truncation and preserve the required end marker.

**Depends on:** 1, 2. **Issues:** #7, #10. **Scope:** medium. **Likely files:** `GPUSimilaritySearch.swift`, `PhotoSearchModel.swift`, `PhotoSearcher.swift`, `TextEncoder.swift`, new search tests.

### 6. Keep a working model available during a switch

**October 9 recovery-hardening follow-up (uncommitted):** search/similar-photo now snapshot spec plus embeddings and re-validate epoch and spec identity after encoding, with a GPU-dimension check before executing; corrupt coordinator state surfaces explicitly with a Repair action that quarantines only the manifest. Covered by the new stale-search and corrupt-repair integration tests in the 113-test full-suite pass (`build/full-recovery-final.xcresult`).

- [ ] Verify the actual S2 files and input contract before promising S2 fallback; retain its valid model and index.
- [ ] Remember the active and requested models, prevent old in-flight tasks from contaminating a new index, and activate a rebuilt index only when ready.

**Verify:** switch/relaunch during loading, searching, and indexing; cancel or fail a rebuild; return to the last valid model with correct results. Never reuse another model’s index.

**Depends on:** 4, 5. **Issues:** #11. **Scope:** medium, with switching and rollback verified separately. **Likely files:** `Embedding.swift`, `PhotoSearcher.swift`, a new model-state component, transition tests.

### 7. Add pause/resume and predictable resource cleanup

- [ ] Pause, resume, cancel, and retry indexing from saved progress; define sensible behavior when the device gets hot or enters low-power mode.
- [ ] Release unused model resources on completion, failure, backgrounding, cancellation, and switching; preserve existing image-buffer protections.

**Verify:** repeated indexing/search cycles release unused resources; interruption resumes safely and preserves saved counts. Device-specific tuning waits for measurements in item 12.

**Depends on:** 3, 4, 6. **Issues:** #12, #13. **Scope:** medium, split into lifecycle and controls changes. **Likely files:** `PhotoSearcher.swift`, model-state component, `ImgEncoder.swift`, lifecycle/resume tests.

**Checkpoint:** close/reopen, interrupted rebuild, failed save, and model rollback all preserve usable saved search data.

## Then: correct models and easy installation

### 8. Use independent reference checks

- [ ] Generate expected word-processing and model results from the original model, rather than from the app’s own implementation.
- [ ] Connect the existing comparison checks to real image/text results, including empty, Unicode, punctuation, and long searches.

**Verify:** app results agree with the independently produced reference; intentional word-processing or image-preparation mistakes fail the check. Version the reference inputs, model files, tool versions, and pass/fail tolerances.

**Depends on:** 1; integrate encoder fixes from 5. **Issues:** #5, #7. **Scope:** medium. **Likely files:** `generate_parity_reference.py`, `gen_token_fixtures.swift`, fixture files, `ParityMetricsTests.swift`, comparison documentation.

### 9. Make the S4 model files reproducible

- [ ] Record the original checkpoint, exact conversion tools, file fingerprints, sizes, supported systems, and required notices.
- [ ] Reproduce and check both S4 model halves against item 8; document image preparation and where output normalization happens.

**Verify:** single-image, image-batch, and text predictions pass the reference comparisons on supported hardware. Document any difference from the issue’s planned normalization method instead of concealing it.

**Depends on:** 8 and the recorded platform/model-use decisions from #4. **Issues:** #16. **Scope:** medium. **Likely files:** new S4 conversion tool, model manifest, `Embedding.swift`, conversion documentation, model tests.

### 10. Implement the chosen download strategy

- [ ] Record the remaining #4 terms/notices and apply iOS 18 support. Identify permitted model sources and trusted file fingerprints.
- [ ] Implement user-requested downloads, space checks, cancellation/retry, integrity checks, and installation of a complete compatible model before activation.

**Verify:** fresh install, interrupted download, damaged files, insufficient space, relaunch, and offline search/indexing all behave correctly. Keep the current local packaging helper clearly marked as a development path; do not mistake it for this feature.

**Depends on:** 6, 9 and confirmed acquisition inputs. **Issues:** #4, #15. **Scope:** medium, delivered as separate installation and runtime-discovery changes. **Likely files:** new model-installation component, `PhotoSearcher.swift`, project settings, installation tests, setup documentation.

### 11. Connect model and indexing controls

- [ ] Show active/requested models, actual space requirements, rebuild expectations, and install/progress/error/retry states.
- [ ] Connect controls to the shared switching and resume rules; check accessibility, existing translations, and privacy wording.

**Verify:** fresh-user and interrupted-work walkthroughs on phone and Mac; announced counts and actions match saved state. Explain that model downloads use the network while private photos/searches remain local.

**Depends on:** 3, 6, 7, 10. **Issues:** #20. **Scope:** medium; handle copy/translations in a separate small change. **Likely files:** `ConfigView.swift`, `BuildIndexView.swift`, interface tests, localization resources.

**Checkpoint:** a fresh installation can acquire a model, build a useful index, recover from interruption, and search offline with clear controls.

## Later: measurements and the formal S4 decision

### 12. Resume the postponed quality and device comparison

- [x] Add an automated component performance runner with generated inputs, repeated timing samples, memory/thermal observations, reproducible reports, and baseline comparisons. See the [measurement guide](/Users/abackman/Queryable/docs/performance-measurement.md).
- [x] Record an initial M5 Mac baseline: three native runs, batches of 32, and search libraries up to 25,000 entries. Validate the model-only simulator reporting path.
- [x] Record the physical iPhone 17 Pro baseline with the same three-run workload. All tests passed; measured text encoding used CPU-only execution. Broader app-level and quality checks remain open.
- [ ] Build the private 2,000–10,000-photo / 300–500-search evaluation set and define numerical targets before looking at S4 results.
- [ ] Compare S2/S4 search and similar-photo results; measure speed, memory, storage, loading, and heat on the agreed devices and a large synthetic library.

**Verify:** repeatable reports identify model/app versions and device conditions. Personal data remains private; public reports contain aggregate results only. Identify remaining memory duplication before deciding whether another storage representation is needed.

**Depends on:** 4, 5, 8, 9; **postponed until benchmark work resumes.** **Issues:** #6, #9. **Scope:** medium per evaluator/measurement change. **Likely files:** new evaluation tools, private dataset description, public-safe report templates, measurement notes.

### 13. Decide speed, precision, and storage tradeoffs

- [x] Correct the device capability check to read the hardware identifier, retaining CPU fallback for known older iPhone/iPad generations; policy regression tests pass. The original iPhone 17 Pro baseline selected `cpuOnly` through the old generic model-label check.
- [x] Verify effective text compute settings on the physical iPhone: all three runs in `performance-results/iphone-indexing-fixes-retry` passed with `all` for both models. Text prediction improved, but load time, memory, and the search P95 increased; the performance guide records this tradeoff. The first attempt ended in a process-exit watchdog during Core ML preparation and supplied no benchmark report.
- [ ] Compare supported compute options and smaller model variants; test lower-precision saved search data separately.
- [ ] Record quality and resource tradeoffs; keep current precision if alternatives fail. Complete the required storage-format support/tests even if lower precision is not selected.

**Verify:** each accepted variant passes the existing accuracy checks and predeclared budgets; unsupported/failed variants remain unavailable. All model sizes and saved-data variants have restart/ranking tests.

**Depends on:** 9, 12. **Issues:** #8, #9, #14. **Scope:** medium per independent variant. **Likely files:** conversion tool, `EmbeddingStore.swift`, `Embedding.swift`, variant/storage tests, comparison report.

### 14. Record the S4 approval or hold decision

- [ ] Perform the integrated offline, recovery, rollback, large-library, and device checks against the recorded criteria.
- [ ] Record the evidence-based decision in #18. Keep documentation consistent with S4 as the selected app default while clearly stating that the release gate remains open until its acceptance evidence is complete.

**Verify:** all required evidence is linked; absent benchmark evidence keeps the gate open. The user confirmed on October 9 that the current model loads and works on their phone and Mac; keep the broader quality, resource, recovery, and fresh-install checks outstanding. Do not equate a successful build or current default with release approval.

**Depends on:** 1–13. **Issues:** #18. **Scope:** small. **Likely files:** release report, README, model-selection/default settings where necessary.

**Checkpoint:** #18 has a passed acceptance report before treating S4 as an approved upgrade.

## Optional Mac quality tier: after S4 acceptance

### 15. Trial SigLIP, then record a decision

- [ ] First build/check its exact word-processing component (#17); next reproduce and validate the Mac trial model (#21).
- [ ] Compare English search benefits and device costs under #19; record promote/defer/reject. If promoted, integrate it through the existing installation, switching, resource, and UI paths and repeat their checks.

**Verify:** separate changes for tokenizer, model export, comparison, and any promotion. Each has reference results and an independent index; existing S4 continues to work. Word-processing reference tests still cover multiple languages as required by #17, without changing the English-only product scope.

**Depends on:** 8–14. **Issues:** #17, #21, #19. **Scope:** medium per separate change. **Likely files:** new tokenizer and tests, SigLIP conversion tool, experimental model spec, comparison/decision reports; promotion gets its own bounded plan if approved.

Under the current criteria, #19 needs evidence even for a no-go. A scope-only postponement today does not satisfy that gate. #22 is excluded from required work; pursue it only after its stated prerequisites and explicit authorization, or record an appropriate not-planned decision.

## Review evidence and limits

Reviewed GitHub issue bodies/comments and all six merged PRs against upstream/local commit `2a43a7b`, plus the uncommitted fixes from this chat. This was a static review and planning pass, not a new full app test run or inspection of private on-device indexes.

- **Photo-loading race:** [indexer](/Users/abackman/Queryable/Queryable/Queryable/ViewModel/PhotoSearcher.swift:248) reads the image immediately after [a callback-based request](/Users/abackman/Queryable/Queryable/Queryable/PhotoHelper/CachedImageManager.swift:66). Apple confirms [photo requests return immediately by default](https://developer.apple.com/documentation/photos/phimagerequestoptions/issynchronous), so the callback can arrive later.
- **Blank entries counted as work:** [photo-fetch failure](/Users/abackman/Queryable/Queryable/Queryable/ViewModel/PhotoSearcher.swift:259) and [encoding failure](/Users/abackman/Queryable/Queryable/Queryable/ViewModel/PhotoSearcher.swift:287) save zero-filled data. The indexer counts attempted photos rather than committed successes.
- **False completion:** [save-error handling](/Users/abackman/Queryable/Queryable/Queryable/ViewModel/PhotoSearcher.swift:411) still reaches [the finished state](/Users/abackman/Queryable/Queryable/Queryable/ViewModel/PhotoSearcher.swift:428).
- **Search validation gaps:** [GPU validation](/Users/abackman/Queryable/Queryable/Queryable/Model/GPUSimilaritySearch.swift:314) checks type/count without validating all values/layouts; [text search](/Users/abackman/Queryable/Queryable/Queryable/Model/PhotoSearchModel.swift:25) still forces errors into crashes. CPU cosine scoring also needs explicit blank/invalid-input handling.
- **Reference checks unfinished:** [the Python generator](/Users/abackman/Queryable/tools/generate_parity_reference.py:133) raises an unfinished-export error. Existing [word fixtures](/Users/abackman/Queryable/tools/gen_token_fixtures.swift:57) were generated by the app’s own tokenizer, so agreement does not independently prove agreement with the original model.
- **S2 fallback needs verification:** the current preset expects `input_ids`/Float32, but the local S2 model metadata reports `input_tokens`/Int32. Do not promise working rollback before checking the actual paired artifacts.
- **Device smoke check confirmed:** on October 9, the user verified the current model loads and works on their phone and Mac. This confirms basic runnability on both devices, but not the full #5/#6/#18 evidence requirements. The local S4 input, build-packaging, and wording fixes remain uncommitted.
- **Recorded priorities:** [#4](https://github.com/ayy-j/Queryable/issues/4#issuecomment-6031023515), [#6](https://github.com/ayy-j/Queryable/issues/6#issuecomment-6031023369), and [#15](https://github.com/ayy-j/Queryable/issues/15#issuecomment-6031023051) establish platform, English-only, personal-use, download, and benchmark-deferral decisions. [#1’s latest update](https://github.com/ayy-j/Queryable/issues/1#issuecomment-6072112117) distinguishes partial implementation from release completion.

## Complete issue coverage

| Open issue | Checklist items |
|---|---|
| [#1](https://github.com/ayy-j/Queryable/issues/1) Overall upgrade | All required S4 items, #18 approval, then #19 decision |
| [#4](https://github.com/ayy-j/Queryable/issues/4) Remaining decisions | 9–10, 14 |
| [#5](https://github.com/ayy-j/Queryable/issues/5) Original-model comparisons | 1, 8–9 |
| [#6](https://github.com/ayy-j/Queryable/issues/6) Private benchmark | 12, postponed |
| [#7](https://github.com/ayy-j/Queryable/issues/7) Image/text encoding | 2, 5, 8–9 |
| [#8](https://github.com/ayy-j/Queryable/issues/8) Saved-data correctness | 4, 13 |
| [#9](https://github.com/ayy-j/Queryable/issues/9) Memory/storage tradeoffs | 7, 12–13 |
| [#10](https://github.com/ayy-j/Queryable/issues/10) Search correctness | 5 |
| [#11](https://github.com/ayy-j/Queryable/issues/11) Model switching | 6 |
| [#12](https://github.com/ayy-j/Queryable/issues/12) Resource cleanup | 7 |
| [#13](https://github.com/ayy-j/Queryable/issues/13) Resume/recovery | 2–4, 7, 12 |
| [#14](https://github.com/ayy-j/Queryable/issues/14) Speed/size options | 13 |
| [#15](https://github.com/ayy-j/Queryable/issues/15) Model installation | 10 |
| [#16](https://github.com/ayy-j/Queryable/issues/16) Reproducible S4 files | 9 |
| [#17](https://github.com/ayy-j/Queryable/issues/17) SigLIP word processing | 15, later |
| [#18](https://github.com/ayy-j/Queryable/issues/18) S4 approval | 14 |
| [#19](https://github.com/ayy-j/Queryable/issues/19) SigLIP decision | 15, later |
| [#20](https://github.com/ayy-j/Queryable/issues/20) Clear controls | 3, 11 |
| [#21](https://github.com/ayy-j/Queryable/issues/21) SigLIP trial model | 15, later |
| [#22](https://github.com/ayy-j/Queryable/issues/22) Larger experiment | Optional; excluded from completion |

The already-closed #3 needs no duplicate implementation issue. Keep every remaining issue open until its own required checks pass; merged groundwork alone is insufficient.
