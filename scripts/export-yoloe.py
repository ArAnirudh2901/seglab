#!/usr/bin/env python3
"""Export YOLOE-26 prompt-free vision models for the browser lane (js/yoloe-detect.js).
Prunes the ONNX to output0 (boxes; SlimSAM owns masks) and writes models/yoloe/.
fp32 for now — onnxconverter-common fp16 mistypes the LRPC head (TODO: native half / int8).

Usage: python scripts/export-yoloe.py [s m l ...]   (default: s)
"""
import json
import sys
from pathlib import Path

import onnx
from ultralytics import YOLOE

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models" / "yoloe"
SPIKE = ROOT / "spikes" / "yoloe"
OUT.mkdir(parents=True, exist_ok=True)
scales = [a for a in sys.argv[1:] if a in {"n", "s", "m", "l", "x"}] or ["s"]

for scale in scales:
    weight = f"yoloe-26{scale}-seg-pf.pt"
    cached = SPIKE / weight
    print(f"[export] {scale}: loading {weight}")
    model = YOLOE(str(cached) if cached.exists() else weight)
    names = model.names
    raw = model.export(format="onnx", imgsz=640, opset=19, simplify=True, half=False)
    dst = OUT / f"yoloe-26{scale}-seg-pf.onnx"
    onnx.utils.extract_model(str(raw), str(dst), ["images"], ["output0"])
    (OUT / f"yoloe-26{scale}-pf.vocab.json").write_text(json.dumps(names, ensure_ascii=False))
    print(f"[export] {scale}: {dst.name} {dst.stat().st_size / 1e6:.1f}MB, {len(names)} labels")

print("[export] done ->", OUT)
