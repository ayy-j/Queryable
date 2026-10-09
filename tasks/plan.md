# Queryable: plan to finish the remaining work

Reviewed October 9, 2026. Covers all **20 open issues** in [ayy-j/Queryable](https://github.com/ayy-j/Queryable/issues), their comments, merged changes, and the current local code.

**Recommended order: make indexing trustworthy, finish model installation and controls, then prove that the new model improves search. Evaluate the larger Mac model afterward.**

“Indexing” means preparing the saved photo-search data. S2 is the older search model; S4 is the newer one. SigLIP is a possible additional model for higher-quality Mac searches.

## What is already done

The app has the groundwork to support different models and keep their saved search data separate. The model-description issue, [#3](https://github.com/ayy-j/Queryable/issues/3), is closed. Memory improvements, reusable image buffers, and some automated checks are merged. All six existing pull requests are merged; none is waiting for review.

Our recent local fixes make S4 accept the correct text input and package its files during clean builds. Those fixes remain uncommitted. They help local testing, but do not complete the planned in-app downloads or prove search quality. The model-comparison tools still need real reference results.

**Update, October 9:** You’ve confirmed the current model loads and works on your phone and Mac. That settles basic runnability on both; measured search quality, speed, memory use, recovery, and fresh-install downloads remain to be checked.

## Keep the decisions already recorded

The [October 7 decisions](https://github.com/ayy-j/Queryable/issues/4#issuecomment-6031023515) specify English-only search, personal self-signed use, an iOS 18 minimum, and the existing iPad app running on Apple silicon Macs. Target hardware is the iPhone 17 Pro, Mac Studio M2 Max, and MacBook Air M5.

Models should eventually download when requested, then work offline. Photos and searches stay on the device. The current bundled-model setup is a development bridge to that feature. A [programmatic performance runner](/Users/abackman/Queryable/docs/performance-measurement.md) now measures model loading, encoding, search, memory, and thermal state using generated inputs. Private-photo quality benchmarks remain postponed.

## Complete the work in this order

| Step | Work to finish | What “done” means | Related issues |
|---|---|---|---|
| **1. Make progress trustworthy** | Wait for photos to finish loading before indexing them. Separate successful, unavailable, and failed photos. Report save failures honestly. Set up full automated test execution. | Delayed photos produce real search data; failures stay retryable; “finished” means the work was saved. | [#7](https://github.com/ayy-j/Queryable/issues/7), [#13](https://github.com/ayy-j/Queryable/issues/13), [#20](https://github.com/ayy-j/Queryable/issues/20) |
| **2. Protect saved work** | Make interruptions, damaged files, and low storage recoverable. Verify search rankings and the older S2 model. Add safe model switching, pause/resume/cancel, and reliable cleanup. | Relaunch keeps completed work. Switching never mixes models. A failed rebuild preserves the previous working index. | [#8](https://github.com/ayy-j/Queryable/issues/8), [#9](https://github.com/ayy-j/Queryable/issues/9), [#10](https://github.com/ayy-j/Queryable/issues/10), [#11](https://github.com/ayy-j/Queryable/issues/11), [#12](https://github.com/ayy-j/Queryable/issues/12), [#13](https://github.com/ayy-j/Queryable/issues/13) |
| **3. Verify the model’s calculations** | Compare the app’s word handling and image/text calculations with the original model’s results. Record how to reproduce the S4 files. | Both image and text checks pass, including empty and long searches. Results identify the exact model files and app version tested. | [#5](https://github.com/ayy-j/Queryable/issues/5), [#7](https://github.com/ayy-j/Queryable/issues/7), [#16](https://github.com/ayy-j/Queryable/issues/16) |
| **4. Finish installation and controls** | Confirm model-use terms and download sources; apply the agreed OS support. Add checked, recoverable downloads and clear model, space, progress, and retry controls. | A fresh install can acquire a complete model, reject damaged downloads, recover from interruption, and search offline. | [#4](https://github.com/ayy-j/Queryable/issues/4), [#15](https://github.com/ayy-j/Queryable/issues/15), [#20](https://github.com/ayy-j/Queryable/issues/20) |
| **5. Measure and approve S4** | When benchmark work resumes, compare S2 and S4 on the same private photos and searches. Measure accuracy, speed, memory, storage, and heat. Test faster/smaller options only after the basic model passes. | Agreed targets pass on the selected devices, with a written decision to approve or hold back S4. Keeping the current precision is acceptable if smaller versions perform worse. | [#6](https://github.com/ayy-j/Queryable/issues/6), [#9](https://github.com/ayy-j/Queryable/issues/9), [#14](https://github.com/ayy-j/Queryable/issues/14), [#18](https://github.com/ayy-j/Queryable/issues/18) |
| **6. Decide the optional Mac upgrade** | After S4 is accepted, build SigLIP’s word-processing component and trial model, then compare the benefit with its extra cost and complexity. | Record an evidence-backed decision to offer it, defer it, or reject it. If offered, it passes the same installation, recovery, and search checks. | [#17](https://github.com/ayy-j/Queryable/issues/17), [#21](https://github.com/ayy-j/Queryable/issues/21), [#19](https://github.com/ayy-j/Queryable/issues/19) |

Step 1 comes first. After saved-work protections exist, model correctness checks and download work can progress separately. Step 5 waits for benchmark work to resume; Step 6 waits for S4 acceptance. The still-larger experiment, [#22](https://github.com/ayy-j/Queryable/issues/22), is explicitly optional and does not block completion.

## Why reliability comes first

The review found paths where photo loading returns before the image is ready, failed photos receive blank search entries, and indexing displays “finished” after a save failure. These are code-level risks, not proof that your current index is damaged. The October 9 fixes now await final photo callbacks, retain failed work for retry, require a successful save before completion, and include existing blank entries in the next indexing attempt with a visible repair count. All 45 ordinary simulator tests pass. Real-photo and interface walkthroughs remain to be done. Evidence and individual work items are in the [execution checklist](/Users/abackman/Queryable/tasks/todo.md).

## What remains postponed

Respect the recorded [benchmark deferral](https://github.com/ayy-j/Queryable/issues/6#issuecomment-6031023369). Reliability, installation, and correctness work can continue now. The formal S4 release decision remains open until measured comparisons are available; S4 being the current default is not evidence that those checks passed.

SigLIP remains later work. Under the current issue criteria, postponing it today does not complete its evidence-based decision. Any reduction in that scope needs to be recorded explicitly.

## When the project is finished

The app installs models reliably, builds and resumes a useful photo index, handles failures clearly, searches offline, and passes the agreed device and accuracy checks. Then close the S4 approval issue [#18](https://github.com/ayy-j/Queryable/issues/18), record the SigLIP decision in [#19](https://github.com/ayy-j/Queryable/issues/19), and close the umbrella issue [#1](https://github.com/ayy-j/Queryable/issues/1). A decision against SigLIP can satisfy its gate; the larger experiment need not be built.

Each issue should link to its implementing change and test results before closing. The initial review created this plan. The later performance and indexing fixes are recorded in the checklist; GitHub issues have not been changed.
