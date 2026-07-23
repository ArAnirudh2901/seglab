#!/usr/bin/env python3
"""Precompute CLIP ViT-B/32 text embeddings for the open-vocab lane (js/clip-text.js).

Runtime CLIP in the browser doesn't fit an 8 GB device (q8 collapses alignment,
fp16 won't build, fp32 ~250 MB pressures memory). Instead we run CLIP ONCE here,
offline at fp32, over a fixed vocabulary and ship the resulting embedding table;
the browser just looks vectors up (no text model, fp32 quality, ~tiny RAM).

Vocabulary = the YOLOE-26 prompt-free vocab (4585 curated visual-object words —
already contains every taxonomy kind: flower/rose/tulip/…) plus the basic colour
words. YOLO-World then covers the same broad open vocabulary as the fast baked
lane, but via CLIP embeddings + its stronger localisation head.

Output (models/clip-vocab/):
  clip-vocab.json  — { dim, count, words: [...] }  (index i ↔ words[i])
  clip-vocab.f32   — little-endian float32, count×dim, L2-normalised, row i ↔ words[i]

Usage: python scripts/build-clip-vocab.py
"""
import json
import struct
from pathlib import Path

import torch
from transformers import CLIPModel, CLIPTokenizerFast

ROOT = Path(__file__).resolve().parent.parent
YOLOE_VOCAB = ROOT / "models" / "yoloe" / "yoloe-26s-pf.vocab.json"
OUT = ROOT / "models" / "clip-vocab"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "openai/clip-vit-base-patch32"

COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "violet",
          "pink", "brown", "white", "gray", "black"]

# Curated vocabulary: YOLOE's object words + colours, unique + lowercased.
words = list(json.loads(YOLOE_VOCAB.read_text()).values())
seen = {}
for w in [*words, *COLORS]:
    k = str(w).strip().lower()
    if k and k not in seen:
        seen[k] = True
vocab = list(seen.keys())
print(f"[clip-vocab] {len(vocab)} words (from YOLOE vocab + {len(COLORS)} colours)")

clip = CLIPModel.from_pretrained(MODEL).eval()
tok = CLIPTokenizerFast.from_pretrained(MODEL)

dim = clip.text_projection.out_features
buf = bytearray()
BATCH = 256
with torch.no_grad():
    for i in range(0, len(vocab), BATCH):
        chunk = vocab[i:i + BATCH]
        t = tok(chunk, padding=True, truncation=True, return_tensors="pt")
        pooled = clip.text_model(input_ids=t.input_ids, attention_mask=t.attention_mask).pooler_output
        feats = clip.text_projection(pooled)
        feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 (matches the export smoke test)
        for row in feats.tolist():
            buf += struct.pack(f"<{dim}f", *row)
        print(f"[clip-vocab] embedded {min(i + BATCH, len(vocab))}/{len(vocab)}", end="\r")

(OUT / "clip-vocab.f32").write_bytes(buf)
(OUT / "clip-vocab.json").write_text(json.dumps({"dim": dim, "count": len(vocab), "words": vocab}, ensure_ascii=False))
print(f"\n[clip-vocab] wrote {len(vocab)}×{dim} → clip-vocab.f32 ({len(buf)/1e6:.1f}MB) + clip-vocab.json")
