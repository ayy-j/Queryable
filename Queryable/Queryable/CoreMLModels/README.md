# CoreMLModels

This folder holds tokenizer assets and optional app-local compiled Core ML models.
The default model is MobileCLIP2-S4.

## Why the `.mlmodelc` bundles are missing from git

Compiled `*.mlmodelc/` bundles are intentionally git-ignored (see `.gitignore`:
they are large build artifacts). A fresh checkout therefore contains only the
small tokenizer assets (`vocab.json`, `merges.txt`).

## Required files

The Xcode target's **Bundle Core ML Models** build phase packages the two S4
towers and tokenizer assets on every build. It uses complete model bundles from
this folder first, then falls back to the repo's top-level `models/` directory.
If the models already exist in `models/`, no manual copy is needed.

Otherwise download them from
[Google Drive](https://drive.google.com/drive/folders/12ze3UcqrXt9qeySGh_j_zWE-PWRDTzJv?usp=drive_link)
(legacy MobileCLIP-S2 bundles are also available there) and place the S4 bundles
in this folder or the repo's top-level `models/` directory before building:

- `ImageEncoder_mobileCLIP2_s4.mlmodelc/` (directory) — current default model
- `TextEncoder_mobileCLIP2_s4.mlmodelc/` (directory) — current default model

Final layout:

```text
Queryable/Queryable/CoreMLModels/
├── ImageEncoder_mobileCLIP2_s4.mlmodelc/
├── TextEncoder_mobileCLIP2_s4.mlmodelc/
├── merges.txt
├── vocab.json
└── README.md
```

## Clean builds and missing artifacts

Keep the downloaded bundles in the checkout, not in Derived Data or a built
`.app`. Cleaning the build folder or deleting Derived Data only removes generated
outputs; the next build packages the persistent model files again.

The build fails with the exact missing model name if neither location contains a
complete bundle (`coremldata.bin`, `model.mil`, and `weights/weight.bin`). Missing
tokenizer assets also fail the build. This prevents deploying an app that cannot
find its default model. Startup errors use a "Search model unavailable" heading
and display the actual error, including contract mismatches or loading failures.
