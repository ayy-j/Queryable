# GPU index mutation regression checks

`GPUSimilaritySearch.addEmbeddings` treats incoming IDs as upserts. Existing IDs retain their row positions and receive normalized replacement vectors. New IDs append once. Untouched Float16 rows copy unchanged. The whole incoming dictionary is validated before a staged buffer replaces the current index; a rejected batch leaves the prior IDs, matrix, and graph usable. Every successful mutation rebuilds the graph for the resulting row count. Persistent matrix and query inputs remain Float16; the graph casts both operands to Float32 for matrix multiplication and returns Float32 scores.

The registered `SimilarityValidationTests.swift` exercises 512, 768, 1152, and 1536 dimensions. A state-only test covers replacement-only, mixed replacement/addition, repeated updates, invalid batches, removal, and empty rebuilds without executing a graph. Three GPU tests check scores and rankings after replacements, mixed upserts, repeated updates, rejected batches, removals, full rebuilds, and empty transitions. Dense synthetic vectors have separated cosine scores; an independent CPU implementation accumulates dot products and norms in Double. The predeclared absolute Float16 tolerance is `2e-3`; GPU and CPU rankings must also match exactly.

Run all validation and mutation tests natively on a Metal-capable Mac from the repository root:

```sh
./tools/test-gpu-index-mutations.sh
```

Pass standard Swift test arguments to select a test, for example:

```sh
./tools/test-gpu-index-mutations.sh --filter SimilarityValidationTests.testGPUUpsertsMatchIndependentCPUCosineAtEveryRequiredDimension
```

The runner creates a temporary Swift package under `/private/tmp/queryable-gpu-mutations.*`, copies only the production GPU source and the registered test file, changes only the test module import, and removes the package on exit. It has no UIKit dependency and uses a private build scratch directory. It requires Swift/Metal cache access; a restricted sandbox may require an approved native execution. No photos or downloaded models are needed.

The three tests that execute MPSGraph explicitly skip on iOS Simulator because Apple's graph execution has crashed there. They run on native macOS through this runner or on a physical iOS device through the app test target. State and malformed-input tests remain safe to run in the simulator. Native component coverage does not establish physical iOS performance, end-to-end model parity, photo lifecycle behavior, or the complete issue #10 acceptance gates.

A diagnostic native run using dense, alternating synthetic vectors exposed a separate precision gap in the existing Float16 matrix multiplication: examples included CPU cosine `0.6000000` versus GPU `0.6064453` and CPU `0.8000000` versus GPU `0.8085938`, above `2e-3`. The mutation regression fixtures retain these dense probes and the original `2e-3` tolerance. Casting both graph operands to Float32 fixes the demonstrated accumulation drift while retaining Float16 buffer storage. The installed SDK declares `castTensor:toType:name:` in `MPSGraphTensorShapeOps.h` available from macOS 12 / iOS 15. Float32 graph operands may increase transient GPU memory and arithmetic cost compared with pure Float16 multiplication; app search latency and peak device memory were not measured by these component tests.

Verified on this native Apple Silicon Mac on 2026-10-09: all 11 `SimilarityValidationTests` passed with 0 failures and 0 skips, including the dense GPU score/ranking assertions at all four dimensions. Reproduce with the runner above. This confirms the component numerical and mutation checks; the simulator suite and complete app acceptance remain separate validations.
