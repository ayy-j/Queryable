# Automatic performance measurements

Run one command to build optimized measurement code, measure the model and search components, and save a readable report. The default run repeats the workload in three separate processes. It uses generated images and search data, so you do not need to prepare photos or grant photo-library access.

## Run on your Mac

From the repository folder:

```sh
python3 tools/measure-performance.py run --native-mac
```

This compiles the app's actual model descriptions, word processor, image encoder, and GPU search code into a native Mac test runner. It needs Xcode's Swift tools and the existing local models, with no signed app host or package downloads. A tiny image adapter passes generated images into the shared encoder. The results are labeled **macOS-native-components**: they measure the shared components and exclude the iPad app's Mac compatibility layer.

Model files and word-processing files default to `Queryable/Queryable/CoreMLModels`. Use `--resources /path/to/CoreMLModels` if your complete set lives elsewhere.

The alternative Xcode destination `--destination 'platform=macOS,arch=arm64,variant=Designed for iPad'` is available, but its test host stalled before test execution on this machine. A working ordinary app launch does not guarantee a working Xcode test host. Use the native command above for automated Mac measurements.

## Run on your phone

Connect and unlock the phone, enable Developer Mode, and find its identifier:

```sh
xcodebuild -showdestinations \
  -project Queryable/Queryable.xcodeproj -scheme Queryable
```

Use the identifier from the **iOS** entry:

```sh
python3 tools/measure-performance.py run \
  --destination 'platform=iOS,id=YOUR_PHONE_IDENTIFIER'
```

The runner uses the app's signing team for both the app and its tests. If needed, add `--build-setting DEVELOPMENT_TEAM=YOUR_TEAM_ID`. Complete signing setup in Xcode first; the command does not change your Apple account or provisioning settings. Its dedicated `QueryablePerformance` scheme disables the debugger, coverage, and sanitizer instrumentation; normal app launch is replaced with an empty test host during the run.

## What you get

The command prints the path to `performance-results/<date>/summary.md`. Each folder also contains machine-readable `aggregate.json`, individual run reports, and build/test logs. Native Mac runs preserve the staged source package; Xcode runs preserve the original result bundles. The folder is ignored by Git.

| Measurement | Meaning |
|---|---|
| Image/text model load | First construction in each app process; text includes loading the word-processing files. |
| First image batch / text query | The first prediction, kept separate from repeated predictions. |
| Image batch / batch of one | Resizing, image encoding, copying results into ordinary memory, and checking the outputs. The default batch is 32, matching the indexer. |
| Text query | Turning one of four fixed English searches into its search vector. |
| Index build | Preparing/uploading a generated library for GPU search. Defaults: 1,000, 10,000, and 25,000 entries. Generating the input vectors is setup outside this timing. |
| First / repeated search | GPU scoring, reading scores back, checking them, and the app's full sort to choose the best 120 results. |
| Text search | Word processing, text prediction, and the same scoring/sorting work together. |
| Memory | Process memory sampled every 20 milliseconds, plus snapshots around each measurement. |
| Thermal state | Whether the system reports normal temperature or increasing heat pressure. |

**Median** is the middle result; it helps reduce the effect of a one-off delay. The summary takes the median of the individual run medians. **P95** is the time within which 95% of all recorded samples finished. Raw samples and the spread between runs are preserved. For image batches, “work items/second” means images/second; for index construction, entries/second; for search, searches/second.

Reports identify the hardware, OS, Xcode version, model contract, requested and effective compute settings, workload, Git commit, and a fingerprint of the source files, including uncommitted changes. Model and word-processing files are fingerprinted after timing to avoid warming their disk caches before measurement. The compute setting records permitted processors; it does not prove which processor Core ML used for each operation.

## Compare after a code change

An [initial M5 Mac baseline](performance-baselines/mac-m5-2026-10-09.json) is saved with the project. It used three native runs, 20 samples per repeated operation, batches of 32 images, and generated libraries up to 25,000 entries on October 9, 2026:

| Result | Measured value |
|---|---:|
| Image encoding throughput | 153 images/second |
| Text prediction + search over 25,000 entries | 6.06 ms median; 6.40 ms P95 |
| Sampled peak process memory | 326 MiB, median across runs |
| Thermal state | Normal throughout |

These values describe the native shared components using generated data. They exclude photo loading, index disk I/O, and the app interface. Keep the JSON file for automated comparisons on the same hardware and conditions; use a new baseline for other devices.

An **iPhone 17 Pro baseline** was collected on October 9, 2026, on iOS 27.0.1 using the same three-run workload:

| Result | Measured value |
|---|---:|
| Image encoding throughput | 87 images/second |
| Text prediction + search over 25,000 entries | 11.97 ms median; 12.65 ms P95 |
| GPU scoring and sorting over 25,000 entries | 3.92 ms median |
| Image model loading | 3.18 seconds median |
| Text model and word-file loading | 0.31 seconds median |
| Sampled peak process memory | 873 MiB, median across runs |
| Thermal state | Normal throughout |

All three physical-device tests passed. Full reports and raw samples are saved locally in `performance-results/iphone-baseline`; use that folder with `--baseline` for future runs on this phone. These are component measurements using generated inputs, rather than timings for preparing the real photo library or drawing the interface.

The original baseline requested `all`, but text encoding used `cpuOnly` because the old device check examined the generic `UIDevice.current.model` label. The October 9 fix reads the hardware identifier and retains the compatibility fallback for older iPhone/iPad generations. All three follow-up runs passed on the physical iPhone and recorded `all` for both models. The [post-fix device report](performance-baselines/iphone17-pro-after-fixes-2026-10-09.json) preserves the measurements; complete local results are in `performance-results/iphone-indexing-fixes-retry`.

| Follow-up result | Before fix | After fix |
|---|---:|---:|
| Effective text compute setting | CPU only | All permitted processors |
| Repeated text prediction | 10.57 ms | 6.23 ms |
| Text + search over 25,000 entries, median | 11.97 ms | 10.95 ms |
| Text + search over 25,000 entries, P95 | 12.65 ms | 22.86 ms |
| Text model and word-file loading | 0.31 s | 1.51 s |
| Sampled peak process memory | 873 MiB | 1,041 MiB |
| Image throughput | 87 images/s | 87 images/s |

The fix removes an unintended CPU restriction; it does not improve every measurement. Text prediction was faster, while model loading, sampled memory, and the search tail increased. Both sets used nominal thermal conditions. Processor settings differ, so these results describe a configuration change; the automatic same-configuration regression comparison deliberately rejects them. They do not establish the best compute option or search accuracy.

The first follow-up attempt produced no valid report: the app was terminated by a process-exit watchdog while Core ML was preparing the text model. The three-run retry completed successfully; the earlier termination remains recorded in `performance-results/iphone-indexing-fixes`, without treating it as a measured performance result.

Use the same device and arguments each time. Keep other heavy work quiet and let the device cool before starting.

```sh
python3 tools/measure-performance.py run --native-mac \
  --output performance-results/baseline

# After making your code change:
python3 tools/measure-performance.py run --native-mac \
  --output performance-results/candidate \
  --baseline performance-results/baseline \
  --max-regression-percent 15
```

The second command writes a comparison and exits with code 1 if any median time is more than 15% slower. This is an adjustable starting threshold, not an agreed product acceptance target. Account for run-to-run noise before drawing conclusions, particularly for model loads and first-use measurements with only one sample per process.

Compare existing results without rebuilding:

```sh
python3 tools/measure-performance.py compare \
  performance-results/baseline performance-results/candidate
```

Comparison rejects mismatched devices, OS/Xcode versions, workloads, compute settings, or model fingerprints. It also rejects runs with serious/critical thermal pressure. A code change can differ between runs. A different model or compute option can be measured in a separate report; it does not qualify as the same-workload regression comparison.

## Quick runner check

This smaller workload checks the model measurements and reporting path on a simulator:

```sh
python3 tools/measure-performance.py run \
  --destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  --runs 1 --samples 3 --warmups 1 --batch-size 2 \
  --model-only --compute-units cpuOnly
```

Simulator timings describe the host Mac and its simulator overhead. Use the physical phone for phone performance decisions. The installed iOS 27 simulator crashed inside Apple's GPU graph code during validation, so simulator runs require `--model-only`. Reports explicitly show that GPU search is excluded. The command allows Xcode to start the selected simulator as part of testing.

Use `run --help` for other workload sizes, repetitions, and compute settings. The default model is S4. S2 is selectable, but requires its exact compatible model files; missing files or a model contract mismatch fail the run rather than falling back silently.

## Limits

These are repeatable **component speed and process-memory measurements**. They do not measure search accuracy, loading photos from Photos/iCloud, saving/loading the real index, UI responsiveness, battery use, or a prolonged heat test. They do not replace the postponed private-photo quality benchmark or the broader device acceptance checks.

The model-load measurement starts in a fresh app process, but does not clear OS disk caches or Core ML compiler caches. It is not a cold-boot measurement. The sampled memory peak can miss spikes shorter than 20 milliseconds and includes XCTest and the app host; it is not a complete accounting of GPU/Neural Engine memory. Thermal state is an OS category, not a temperature reading.

The runner fails on missing models, invalid/nonfinite model outputs, missing timing or memory data, incomplete search results, failed tests, or missing report attachments. Partial reports/logs remain available for diagnosis. Ordinary test runs skip this opt-in benchmark.

Native Mac runs write JSON directly. Xcode runs use XCTest's [persistent attachments](https://developer.apple.com/documentation/xctest/xctattachment/lifetime-swift.property), documented `TEST_RUNNER_` environment forwarding (`man xcodebuild`), and `xcresulttool export attachments`.
