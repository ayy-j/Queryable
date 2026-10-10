# Image preprocessing and buffer validation

This follow-up to issue #7 makes the supported image input contract explicit and
tests the shared input path used by single and batch encoding. The runtime
supports `CILanczosScaleTransform`, `32ARGB` pixels, and stretched aspect ratio.
Other filter, pixel-format, or aspect-ratio configurations throw
`unsupportedImagePreprocessing` before model loading or rendering.

Previously, any nonempty filter name passed preprocessing validation. Rendering
then assigned `inputScale` and `inputAspectRatio` through Core Image KVC. A filter
such as `CIGaussianBlur` does not implement that resize contract. The regression
test failed before the fix because both an incompatible filter and an unknown
filter were admitted. Apple's [Lanczos filter documentation](https://developer.apple.com/documentation/coreimage/cilanczosscaletransform)
defines the scale and aspect-ratio parameters used here.

## Shared inputs and pools

`ImgEncoder.imageFeatureProvider` validates the spec, derives input geometry and
feature name from it, and constructs the rendered image feature. Both
`encode(image:)` and `encodeBatch(images:)` use this helper. Existing supported
resize behavior and compatibility fingerprints stay the same.

Pool dimensions must be positive, finite, integral values representable as
`Int`; validation precedes integer conversion. The standalone-buffer fallback
uses the same geometry guard. Pools remain keyed by width, height, and pixel
format. Cleanup now requests `.excessBuffers`, which Apple documents as freeing
[all unused buffers regardless of age](https://developer.apple.com/documentation/corevideo/cvpixelbufferpoolflushflags/excessbuffers).
The previous empty flags freed only aged-out buffers. Live feature providers
retain their buffers through flushing.

## Detached output ownership

The detachment helper still copies embeddings into independently allocated
`MLMultiArray` storage. It now copies `MLShapedArray.scalars`, which respects
logical strides. The old pointer copy ignored strides and included padding in
strided arrays. A synthetic strided source reproduced incorrect values at
512/768/1152 dimensions before the fix. Its custom deallocator also verifies
that retaining the detached embedding does not retain source storage.

## Verification and limits

Five added tests cover unsupported preprocessing before artifact loading and
input creation, model-sized single/batch input features, invalid pool geometry,
live buffers through flushing and reuse, and detached strided output ownership.
Existing pool tests also verify that geometry and pixel-format keys remain
separate. Generated rectangular RGB images exercise actual Core Image rendering
at 256 and 384 px, with the spec's input feature name and ARGB center pixels
checked. Output-copy tests use generated values and isolated allocations.

Run the ordinary simulator suite using [the test guide](test-infrastructure-plan.md),
or add `-only-testing:QueryableTests/EmbeddingModelSpecTests` for focused coverage.
The final combined run executed **93 tests: 90 passed, 3 GPU graph tests skipped,
0 failures**. All 29 model-contract tests passed, including the bundled S4 text
prediction test. Result bundle: `build/image-preprocessing-final.xcresult`;
log: `build/image-preprocessing-final.log`.

The regression logs are `build/image-preprocessing-red.log` (unsupported filters)
and `build/image-preprocessing-focused2.log` (strided copy). The initial red run
recorded assertion failures but stalled during Xcode diagnostic collection and
was terminated; later runs use `-collect-test-diagnostics never`.

These tests do not establish independent preprocessing/model parity, UIImage
orientation behavior, target-device memory reclamation, or the Core ML
IOSurface limit during large-library indexing. Allocation overhead after flushing
all unused buffers also needs device measurement. Those checks and the remaining
#5/#7/#12 release and lifecycle criteria remain open.
