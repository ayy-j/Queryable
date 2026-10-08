# Resolving the XCTest execution block

Status: blocked · Owner: maintainer · Last verified: 2026-10-08 against Xcode 27.0 (build 27A266a)

## The block

`QueryableTests` compiles (`xcodebuild build-for-testing` succeeds), but the suite
**cannot be executed in place** on this machine. Two independent causes, either of
which is sufficient to block a run:

1. **No simulator runtimes are installed.** `xcrun simctl list runtimes` returns
   nothing, so no `platform=iOS Simulator` destination exists.
2. **No signing identity for the fallback destination.** The only other viable
   destination, `platform=macOS,arch=arm64,variant=Designed for iPad`, fails with
   `No signing certificate "iOS Development" found` for team `8ZKRT2YUM5`, and
   ad-hoc signing is refused for SDK iOS 27.0.

Current substitute: a throwaway SwiftPM harness compiles the real
`Embedding.swift`, `BPETokenizer*.swift`, and `ImgEncoder.swift` on macOS and runs
the spec/pool tests (8/8 passing). It **cannot** cover `ImgEncoder.encode`,
`EmbeddingStore` file I/O, or `GPUSimilaritySearch` — those need the real app
target on a simulator or device.

## Resolution plan

### Step 1 — Install an iOS Simulator runtime *(removes cause 1)*

```sh
sudo xcodebuild -downloadPlatform iOS \
  -buildVersion 27A266a
```

or interactively: Xcode → Settings → Components → iOS Simulator.

Verify:

```sh
xcrun simctl list runtimes   # expect at least one iOS 27.x runtime
```

### Step 2 — Run the suite on the simulator *(no signing required)*

Simulator destinations do not need a development certificate:

```sh
DEVELOPER_DIR=/Applications/Xcode-27.0.0.app/Contents/Developer \
xcodebuild test \
  -project Queryable/Queryable.xcodeproj \
  -scheme Queryable \
  -destination 'platform=iOS Simulator,name=iPhone 17' \
  CODE_SIGNING_ALLOWED=NO
```

Acceptance: all tests pass, including the two pixel-buffer-pool tests added with
the model-driven pool change.

### Step 3 — Exercise the paths the harness cannot reach

Once Step 2 is green, add execution coverage for:

- `ImgEncoder.encode` / `encodeBatch` — requires the MobileCLIP-S2 `.mlmodelc`
  towers in the test bundle (or a fixture model); assert output shape and
  normalization against `spec.embeddingDimension`.
- `EmbeddingStore` round-trip — write to a temp directory, `loadAll()`, assert
  header fields, journal replay, and the `metadataTooLarge` guard.
- `GPUSimilaritySearch` — dimension-parameterized top-k against a small known
  corpus on the simulator GPU.

### Step 4 — (Optional) Mac / Designed-for-iPad runs

Only needed if Mac-native test runs are desired. Requires a valid **iOS
Development** certificate for team `8ZKRT2YUM5` in the keychain (Xcode →
Settings → Accounts → Manage Certificates). Ad-hoc signing is not accepted by
SDK iOS 27.0 for this variant, so there is no cert-free path here.

### Step 5 — Wire CI so the block cannot regress

Add a GitHub Actions job on a `macos-15`-or-newer runner (Xcode 27 selected via
`DEVELOPER_DIR`) that runs the Step 2 command on every PR. Runner images ship
with simulator runtimes preinstalled, which removes the local-machine dependency
entirely and makes "tests actually executed" a merge gate rather than a
manual claim.

## Out of scope

The epic-level blockers (S4/SigLIP `.mlmodelc` towers, Gemma tokenizer, parity
fixtures, benchmark corpora) are tracked in issue #1 and are not resolved by this
plan.
