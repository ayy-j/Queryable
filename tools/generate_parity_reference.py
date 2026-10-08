#!/usr/bin/env python3
"""Generate pinned PyTorch reference vectors for issue #5 parity gates.

Produces a versioned VectorFixtureFile JSON that the Swift parity harness
compares Core ML outputs against. Run on an Apple host with the pinned
checkpoint and toolchain; the output JSON carries full provenance.

Usage:
    python3 tools/generate_parity_reference.py \
        --model mobileclip-s2 \
        --checkpoint /path/to/checkpoint.pt \
        --out Queryable/ParityFixtures/mobileclip-s2-vectors-v1.json

Pinned toolchain (record actual versions in the output manifest):
    torch==2.4.0, ml-mobileclip @ <commit>, transformers==4.44.0

Without --checkpoint the script emits the fixture schema with placeholder
vectors and exits non-zero, so CI fails loudly instead of passing silently.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys

FIXTURE_VERSION = "vectors-v1"

MODEL_CONTRACTS = {
    "mobileclip-s2": {
        "model_id": "mobileclip-s2",
        "model_revision": "mobileclip-s2-v1",
        "tokenizer_assets": ["vocab.json", "merges.txt"],
        "preprocessing_fingerprint": "ci-lanczos-argb-256-v1",
        "embedding_dimension": 512,
        "context_length": 77,
    },
    "mobileclip2-s4": {
        "model_id": "mobileclip2-s4",
        "model_revision": "mobileclip2-s4-v1",
        "tokenizer_assets": ["vocab.json", "merges.txt"],
        "preprocessing_fingerprint": "ci-lanczos-argb-256-v1",
        "embedding_dimension": 768,
        "context_length": 77,
    },
    "siglip-so400m": {
        "model_id": "siglip-so400m",
        "model_revision": "so400m-v1",
        "tokenizer_assets": ["gemma-tokenizer.model"],
        "preprocessing_fingerprint": "siglip-preprocess-v1",
        "embedding_dimension": 1152,
        "context_length": 64,
    },
}

# Prompts shared with the Swift token fixtures so text vectors and token
# fixtures cover the same edge cases.
PROBE_PROMPTS = [
    ("empty", ""),
    ("single-word", "cat"),
    ("punctuation", "Hello, world!"),
    ("unicode-diacritics", "café naïve résumé"),
    ("emoji", "cat 🐱 dog 🐶"),
    ("long-description", "a small orange tabby cat sitting on a windowsill"),
    ("ocr-like", "SALE 50% OFF ends 12/31/2024!!!"),
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_placeholder_vectors(contract: dict) -> list[dict]:
    dim = contract["embedding_dimension"]
    return [
        {
            "id": f"text-{pid}",
            "kind": "text",
            "source": prompt,
            "vector": [0.0] * dim,
        }
        for pid, prompt in PROBE_PROMPTS
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_CONTRACTS), required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--torch-version", default="2.4.0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    contract = MODEL_CONTRACTS[args.model]
    manifest = {
        "fixtureVersion": FIXTURE_VERSION,
        "modelID": contract["model_id"],
        "modelRevision": contract["model_revision"],
        "checkpointHash": sha256_file(args.checkpoint) if args.checkpoint else None,
        "tokenizerAssets": contract["tokenizer_assets"],
        "preprocessingFingerprint": contract["preprocessing_fingerprint"],
        "embeddingDimension": contract["embedding_dimension"],
        "contextLength": contract["context_length"],
        "createdAt": datetime.date.today().isoformat(),
    }

    if args.checkpoint is None:
        print(
            "No --checkpoint provided: emitting schema placeholder (all-zero vectors).",
            file=sys.stderr,
        )
        print(
            "Replace with real PyTorch outputs before gating any release.",
            file=sys.stderr,
        )
        vectors = build_placeholder_vectors(contract)
        ok = False
    else:
        try:
            import torch  # noqa: F401
        except ImportError:
            print("torch is required to generate real reference vectors.", file=sys.stderr)
            return 2
        # Model-specific export goes here, pinned to ml-mobileclip @ <commit>.
        # This stub records provenance; the actual encode_image/encode_text
        # calls land with issue #16 (S4) / #21 (So400m).
        raise NotImplementedError(
            "Real export not yet wired: add the pinned encode_image/encode_text "
            "calls for %s, then re-run." % args.model
        )

    payload = {"manifest": manifest, "precision": "fp32", "vectors": vectors}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
