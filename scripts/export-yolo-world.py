#!/usr/bin/env python3
"""Export YOLO-World-v2 as an OPEN-VOCABULARY browser lane (js/yolo-world-detect.js).

Unlike YOLOE (prompt-free — text is re-parameterized/baked at export, fixed vocab),
YOLO-World-v2's contrastive head takes the class text features as a live tensor. We
wrap the model so `forward(images, txt_feats)` exposes `txt_feats` as a RUNTIME ONNX
input, so the browser can detect ANY phrase: encode it with CLIP ViT-B/32 (transformers.js
`Xenova/clip-vit-base-patch32`, the exact encoder Ultralytics uses) and feed the 512-d
vectors here. Runs on onnxruntime-web WASM (no GPU/f16) or WebGPU.

Contract (verified): inputs images[1,3,640,640] f32 + txt_feats[1,NC,512] f32 →
output0[1,4+NC,8400] = (cx,cy,w,h box + NC sigmoid class scores) per anchor. NOT
NMS-free (YOLOv8 head) — the caller thresholds + runs NMS (text-core.nms).

NC is the class-slot capacity, FIXED at export (the head width `no = nc + 4*reg_max`
is baked). The browser fills the slots with the phrase + its taxonomy synonyms
(search-taxonomy.expandQuery) and pads the rest by repetition.

Usage: python scripts/export-yolo-world.py [s m l x ...]   (default: s)
Produces models/yolo-world/yolo-world-{scale}.onnx (fp32) and .int8.onnx (WASM).
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from ultralytics import YOLOWorld

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models" / "yolo-world"
OUT.mkdir(parents=True, exist_ok=True)

NC = 32           # class-slot capacity (phrase + synonyms; pad by repeat)
IMGSZ = 640
scales = [a for a in sys.argv[1:] if a in {"s", "m", "l", "x"}] or ["s"]


class RuntimeText(nn.Module):
    """Expose the WorldModel so text features are a forward argument (ONNX input)."""

    def __init__(self, world_model):
        super().__init__()
        self.wm = world_model

    def forward(self, images, txt_feats):
        return self.wm.predict(images, txt_feats=txt_feats)


for scale in scales:
    weight = f"yolov8{scale}-worldv2.pt"
    print(f"[export] {scale}: loading {weight}")
    wm = YOLOWorld(weight).model.eval()
    head = wm.model[-1]
    head.export = True
    head.format = "onnx"
    head.nc = NC
    head.no = head.nc + head.reg_max * 4  # rebuild the baked class width

    dummy_img = torch.zeros(1, 3, IMGSZ, IMGSZ)
    dummy_txt = torch.randn(1, NC, 512)
    dummy_txt = dummy_txt / dummy_txt.norm(dim=-1, keepdim=True)

    fp32 = OUT / f"yolo-world-{scale}.onnx"
    torch.onnx.export(
        RuntimeText(wm), (dummy_img, dummy_txt), str(fp32),
        input_names=["images", "txt_feats"], output_names=["output0"],
        opset_version=19, do_constant_folding=True, dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(fp32)))

    # Dynamic INT8: weights → int8 (≈4× smaller), activations stay fp32. The
    # universal WASM lane; fp32 stays for WebGPU where the memory is cheap.
    int8 = OUT / f"yolo-world-{scale}.int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    print(f"[export] {scale}: {fp32.name} {fp32.stat().st_size/1e6:.1f}MB · "
          f"{int8.name} {int8.stat().st_size/1e6:.1f}MB · NC={NC}")

print("[export] done ->", OUT)
