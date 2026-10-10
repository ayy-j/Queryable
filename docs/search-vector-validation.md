# Search vector validation and error handling (issues #7/#10)

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

## CPU fallback and search errors

`PhotoSearcherModel.similarityScores` applies the same shape/type/stride/value/norm
rules to the query and every indexed vector. It accumulates normalized cosine
products in Double and clamps rounding drift to `[-1,1]`. An invalid entry throws;
no partial score dictionary is returned. The cosine and spherical-distance
helpers also validate their inputs instead of dividing by zero or accepting
incompatible vectors.

Text prediction throws when the encoder is unloaded, prediction fails, or its
output is invalid. No force-try or force-unwrap remains in the search-model
wrapper. Invalid text output is reported as a model failure, distinct from an
invalid saved index. The optional predictor closure exists for failure tests;
ordinary app execution uses `TextEncoder`.

Text and similar-photo search share backend selection and ranking. GPU missing
state/results errors disable that GPU instance and use the validated CPU scorer.
Invalid vectors are surfaced as errors. Successful rankings use photo IDs to
break score ties consistently. Results are published only after complete scoring;
failed attempts clear old results, and retries clear previous error messages.
An empty successful text search reports `NO_RESULT`; a failure reports
`SEARCH_ERROR`. Missing similar-photo references report an error and always stop
the spinner. Phone and Mac result views display the error messages.

`PhotoSearchModelTests` checks analytic CPU cosine values/rankings at all four
dimensions with absolute tolerance `1e-6`, invalid query/index inputs, large finite
vectors, missing encoders, prediction failures, and invalid model outputs.
`SearchReliabilityTests` exercises unloaded/failed/invalid text prediction,
successful retry through the real CPU fallback, S2 dimension selection,
empty-index behavior, and missing/successful similar-photo requests. Generated
inputs and injected saves avoid changing the real saved index.

## Remaining acceptance work

GPU-vs-CPU ranking tolerance tests, add/remove/rebuild ranking coverage,
model/index identity at activation, tokenizer truncation/end-marker checks, and
physical-device UI/error walkthroughs remain open under #7/#10/#11. A GPU index
containing a legacy blank vector fails build validation; storage preserves those
records for the separate re-index repair path. Device graph execution remains
required for ranking acceptance; these simulator tests do not establish GPU
numerical parity or verify GPU backend-failure recovery.

Use the simulator command in [the test guide](test-infrastructure-plan.md),
optionally adding `-only-testing:QueryableTests/SimilarityValidationTests`.
