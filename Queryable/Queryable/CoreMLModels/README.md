# CoreMLModels

This folder holds the compiled Core ML models and tokenizer assets used by Queryable.

## Why the `.mlmodelc` bundles are missing from git

Compiled `*.mlmodelc/` bundles are intentionally git-ignored (see `.gitignore`:
they are large build artifacts). A fresh checkout therefore contains only the
small tokenizer assets (`vocab.json`, `merges.txt`).

## Required files

The compiled `.mlmodelc` bundles are git-ignored. If the top-level `models/`
directory of this repo already contains them, copy them here:

```sh
cp -R ../../models/ImageEncoder_mobileCLIP2_s4.mlmodelc \
      ../../models/TextEncoder_mobileCLIP2_s4.mlmodelc .
```

Otherwise download them from
[Google Drive](https://drive.google.com/drive/folders/12ze3UcqrXt9qeySGh_j_zWE-PWRDTzJv?usp=drive_link)
(legacy MobileCLIP-S2 bundles are also available there) and place them in this
folder before building/running:

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

## What happens if you skip this

The app detects the missing files at startup and shows a "Model files missing"
message (with the download link) instead of crashing with Core ML's cryptic
`The model is not found at URL: ... .mlmodelc` error.
