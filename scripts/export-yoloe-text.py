#!/usr/bin/env python3
"""Export YOLOE-26 TEXT-PROMPT to ONNX: txt_feats is a live graph input.

The -pf (prompt-free) export bakes a 4585-class vocabulary into an LRPC head, so
arbitrary phrases are impossible. This variant takes the class embeddings as an
input instead, which is what makes the lane open-vocabulary.

RepRTA stays INSIDE the graph (head.get_tpe = normalize(reprta(t))): the runtime
feeds raw MobileCLIP2-B text vectors and the graph adapts them. Keeping the 1.58M
-param adapter here rather than in JS means the browser only ever needs a plain
CLIP text encoder, and the two halves can never drift out of sync.

The class axis is DYNAMIC. A static 32 with pad-by-repeat looks harmless — the
duplicate slots score identically, so a max-over-classes is unaffected — but the
head emits each anchor once PER SLOT, and the output is a fixed top-300. Measured
on the canonical NEF: a one-phrase query got 10 unique boxes out of 300 rows,
because the other 290 were the same ten anchors repeated across 32 identical
classes. That capped recall at ~10 instances no matter what was in the frame.
Feeding exactly as many slots as there are phrases returns the full 300.

fp16 is applied by the shared mixed-precision pass. The -pf export's documented
fp16 blocker was "onnxconverter-common mistypes the LRPC head" — this variant has
no LRPC head, so blanket conversion is tried first and only falls back to a
derived block list. A block list containing weight-heavy Conv is REJECTED: it
would forfeit the entire size win (spikes/sam21/FINDINGS.md §1.2, where blocking
MatMul took the SAM encoder 77.7 → 142.3 MB).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "spikes" / "sam21"))

SLOTS = 32          # tracing width and the runtime's upper bound (js/yoloe-detect.js)
IMGSZ = 640         # YOLOE-26 native square
DIM = 512           # MobileCLIP2-B text width
OPSET = 19


class TextPromptYOLOE(nn.Module):
    """images + txt_feats -> detections, with RepRTA applied in-graph."""

    def __init__(self, model: nn.Module, slots: int = SLOTS):
        super().__init__()
        self.model = model
        head = model.model[-1]
        head.nc = slots
        head.export = True
        head.format = "onnx"
        # Prompt-free models carry an LRPC head and refuse live prompts.
        assert not hasattr(head, "lrpc"), "checkpoint is prompt-free; use the non -pf weights"

    def forward(self, images: torch.Tensor, txt_feats: torch.Tensor):
        return self.model.predict(images, tpe=txt_feats)


def load_model(weights: Path):
    from ultralytics import YOLOE

    yoloe = YOLOE(str(weights))
    core = yoloe.model.float().eval()
    for p in core.parameters():
        p.requires_grad_(False)
    return core


def export_fp32(core: nn.Module, dst: Path) -> Path:
    wrapper = TextPromptYOLOE(core).eval()
    images = torch.zeros(1, 3, IMGSZ, IMGSZ)
    # Random unit vectors, not zeros: a zero text embedding makes normalize()
    # produce NaN and the tracer bakes the degenerate branch.
    txt = torch.randn(1, SLOTS, DIM)
    txt = txt / txt.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        ref = wrapper(images, txt)
    out0 = ref[0] if isinstance(ref, (tuple, list)) else ref
    print(f"  traced output: {tuple(out0.shape)}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (images, txt),
        str(dst),
        opset_version=OPSET,
        input_names=["images", "txt_feats"],
        output_names=["output0"],
        # Only the class axis is dynamic; the 640² image and the top-300 output
        # stay static so WebGPU still specialises everything that matters.
        dynamic_axes={"txt_feats": {1: "nc"}},
        do_constant_folding=True,
        dynamo=False,
    )
    # Trim to the detection output only. The segment head's 3.3 MB proto tensor
    # would otherwise cross the worker boundary on every query; masks come from
    # the SAM 2.1 lane.
    model = onnx.load(str(dst))
    keep = [o.name for o in model.graph.output][:1]
    onnx.utils.extract_model(str(dst), str(dst), ["images", "txt_feats"], keep)
    print(f"  fp32 -> {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


def onnx_reference(path: Path, images: np.ndarray, txt: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(None, {"images": images, "txt_feats": txt})[0]


def weight_heavy_ops(model: onnx.ModelProto, blocked: list[str], min_mb: float = 1.0) -> dict[str, float]:
    """MB of initializer weight held by each blocked op type."""
    init = {i.name: i for i in model.graph.initializer}
    tally: dict[str, float] = {}
    for node in model.graph.node:
        if node.op_type not in blocked:
            continue
        mb = sum(
            int(np.prod(init[i].dims)) * 4 / 1e6
            for i in node.input
            if i in init and init[i].dims
        )
        tally[node.op_type] = tally.get(node.op_type, 0.0) + mb
    return {k: v for k, v in tally.items() if v >= min_mb}


def to_fp16(src: Path, dst: Path, images: np.ndarray, txt: np.ndarray, tol: float) -> Path:
    """Blanket fp16 first; a derived block list only if that fails validation."""
    from onnxconverter_common import float16

    from mixed_precision import repair_boundaries, repair_casts

    ref = onnx_reference(src, images, txt)

    def build(block: list[str] | None) -> onnx.ModelProto:
        m = onnx.load(str(src))
        conv = float16.convert_float_to_float16(
            m, keep_io_types=True, op_block_list=block or None, disable_shape_infer=False,
        )
        # Both repairs mutate in place and return a count, not the model.
        n_cast = repair_casts(conv)
        n_bound = repair_boundaries(conv)
        print(f"    repairs: {n_cast} casts, {n_bound} boundaries")
        return conv

    def agreement(model: onnx.ModelProto) -> float:
        onnx.save(model, str(dst))
        got = onnx_reference(dst, images, txt)
        # Score column drives ranking; compare it directly rather than a whole-
        # tensor cosine, which stays high while the ordering has already moved.
        a = ref[..., 4].ravel()
        b = got[..., 4].ravel()
        return float(np.abs(a - b).max())

    err = agreement(build(None))
    print(f"  blanket fp16: max score delta {err:.5f}")
    if err <= tol:
        print(f"  fp16 -> {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)  [blanket]")
        return dst

    # Fall back to a mixed build. Softmax/normalize are the fragile ops and hold
    # no weights; Conv holds essentially all of them and must never be blocked.
    for block in (["Softmax"], ["Softmax", "ReduceSum"], ["Softmax", "ReduceSum", "Div"]):
        model = build(block)
        heavy = weight_heavy_ops(onnx.load(str(src)), block)
        if heavy:
            print(f"  REJECT block {block}: holds weights {heavy} — size win forfeited")
            continue
        err = agreement(model)
        print(f"  mixed {block}: max score delta {err:.5f}")
        if err <= tol:
            print(f"  fp16 -> {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)  [mixed {block}]")
            return dst
    raise SystemExit(f"fp16 conversion never reached tolerance {tol}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yoloe-26l-seg.pt")
    ap.add_argument("--scale", default="l")
    ap.add_argument("--out", default="models/yoloe")
    ap.add_argument("--tol", type=float, default=0.02, help="max abs score delta vs fp32")
    args = ap.parse_args()

    out = Path(args.out)
    fp32 = out / f"yoloe-26{args.scale}-text.fp32.onnx"
    fp16 = out / f"yoloe-26{args.scale}-text.fp16.onnx"

    print(f"[1/3] loading {args.weights}")
    core = load_model(Path(args.weights))

    print("[2/3] tracing to ONNX (txt_feats live, RepRTA in-graph)")
    export_fp32(core, fp32)

    print("[3/3] fp16")
    rng = np.random.default_rng(0)
    images = rng.random((1, 3, IMGSZ, IMGSZ), dtype=np.float32)
    txt = rng.standard_normal((1, SLOTS, DIM)).astype(np.float32)
    txt /= np.linalg.norm(txt, axis=-1, keepdims=True)
    to_fp16(fp32, fp16, images, txt, args.tol)


if __name__ == "__main__":
    main()
