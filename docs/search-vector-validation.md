# GPU search vector validation (issue #10)

`GPUSimilaritySearch` checks every build/add input before allocating a replacement
index or changing its IDs. It checks every query before returning empty results
or submitting a graph. Rejected inputs throw `SimilaritySearchError`.

The supported index input is a Float32 `MLMultiArray` shaped `[D]` or `[1,D]`,
with stride 1 along the vector axis. The singleton axis does not contribute an
offset. Other shapes, scalar types, and strided vector layouts are rejected
before reading the raw pointer. Queries use `MLShapedArray<Float32>` with the same
supported shapes and exact dimension. Neither path truncates or pads.

All values must be finite and the L2 norm must exceed `1e-8`. Blank/near-zero
vectors are rejected rather than becoming arbitrary tied rankings. The norm and
normalization division use Double arithmetic to avoid overflow for finite
Float32 inputs; normalized vectors are then converted to Float16 for the GPU.

A valid query on an empty index returns an empty dictionary. An invalid query on
an empty index still throws. Missing state for a nonempty index, missing graph
results, and nonfinite returned scores throw rather than appearing as successful
empty results. Build/add validation failures leave the existing index intact.

`SimilarityValidationTests` covers all four required dimensions (512, 768, 1152,
1536), supported shapes, incorrect lengths, matrix shapes with the same element
count, incompatible scalar types, noncontiguous storage, nonfinite/blank values,
and large finite values. The empty-index integration test exercises the actual
GPU API without graph execution. The validation helper tests require no Metal
device; the integration test explicitly skips if Metal is unavailable.

This is a bounded input-validation change. GPU-vs-CPU ranking tolerance tests,
add/remove/rebuild ranking coverage, model/index identity at activation, CPU
fallback validation, safe text-prediction error handling, and user-visible search
errors remain open under #7/#10/#11. Existing UI code still maps caught GPU query
errors to its current result state. A GPU index containing a legacy blank vector
now fails build validation; storage preserves those records for the separate
re-index repair path. Device graph execution remains required for ranking
acceptance; simulator validation does not establish GPU numerical parity.

Use the simulator command in [the test guide](test-infrastructure-plan.md),
optionally adding `-only-testing:QueryableTests/SimilarityValidationTests`.
