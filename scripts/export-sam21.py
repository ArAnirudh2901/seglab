#!/usr/bin/env python3
"""Export SAM 2.1 (darktable-ai) encoder/decoder for the browser mask lane.

Pipeline: fetch -> sha256 verify -> probe real shapes -> freeze -> emit fp32
(shipping) -> convert fp16 -> validate.

Shape freezing covers the ENCODER outputs and the mask count only. The encoder
declares high_res_feats_0/1 with symbolic dims even though the decoder declares
them static, so ORT would size its arena for an unbounded worst case. Concrete
dims are probed from a real run, never assumed. num_points is deliberately left
dynamic — see the note below.

fp32 is what ships: ORT-Web's WebGPU fp16 kernels produce a materially wrong
embedding for this model (spikes/sam21/FINDINGS.md §1.2). fp16 is still built and
validated so the swap is one line if that is fixed. There is no .ort step —
measured, .ort fails to load where the plain .onnx works (FINDINGS.md §1.3).

Usage:
  python scripts/export-sam21.py                      # small
  python scripts/export-sam21.py --scale tiny
  python scripts/export-sam21.py --block LayerNormalization Softmax
  python scripts/export-sam21.py --inspect            # print graph IO, exit
  python scripts/export-sam21.py --skip-validate      # skip fp32-vs-fp16 check
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnxconverter_common import float16

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models" / "sam21"
SPIKE = ROOT / "spikes" / "sam21"
RELEASE = "release-5.6.0"
BASE = f"https://github.com/darktable-org/darktable-ai/releases/download/{RELEASE}"

# num_points stays DYNAMIC. Freezing it to a max and padding with label -1 is
# not the no-op it looks like: the graph already appends its own padding point,
# and every extra -1 slot adds another not_a_point_embed token to the prompt
# attention. Measured on bus.jpg — 2 real clicks padded to 8 slots score
# predIoU 0.0001 and land IoU 0.227 against the same clicks unpadded.
#   slots  1      2      3      4      6      8      (1 real point)
#   predIoU 0.899 0.777  0.590  0.538  0.111  0.0001
# So the caller passes exactly the clicks it has, and this dim is left symbolic.

# Gates. NOT binary mask IoU at 256²: measured, that scores the threshold rather
# than the conversion. A 345px mask has a 949px 1px-wide boundary band, so ~12px
# of sub-pixel jitter reads as IoU 0.965 while the logit field is cosine
# 0.999997 identical — and §10 re-snaps that boundary with a guided filter over
# upsampled logits anyway. Gate the logit field, the mask choice, and
# disagreements that are NOT boundary jitter.
COS_GATE = 0.9999       # per-click logit-field cosine, fp32 vs fp16
OFFBAND_GATE = 0.02     # disagreeing px >1px from the fp32 boundary, / mask area

ap = argparse.ArgumentParser()
ap.add_argument("--scale", default="small", choices=["tiny", "small", "base-plus"])
ap.add_argument("--block", nargs="*", default=[], help="ops kept in fp32 (default: none)")
ap.add_argument("--inspect", action="store_true")
ap.add_argument("--skip-validate", action="store_true")
args = ap.parse_args()

SPIKE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
MODEL_ID = f"mask-object-sam21-{args.scale}"
pkg = SPIKE / f"{MODEL_ID}.dtmodel"
src = SPIKE / MODEL_ID


def mb(p):
    return p.stat().st_size / 1048576


def fetch(url, dst):
    subprocess.run(["curl", "-fL", "--progress-bar", "-o", str(dst), url], check=True)


# --- fetch + verify + unpack (.dtmodel is a plain DEFLATE zip; darktable_ai/package.py)
versions_path = SPIKE / "versions.json"
if not versions_path.exists():
    fetch(f"{BASE}/versions.json", versions_path)
versions = json.loads(versions_path.read_text())
entry = versions.get(MODEL_ID) or versions.get("models", {}).get(MODEL_ID) or {}
want = str(entry.get("sha256", "")).removeprefix("sha256:")
version = entry.get("version", "unknown")

if not pkg.exists():
    print(f"[sam21] downloading {pkg.name} ...")
    fetch(f"{BASE}/{MODEL_ID}.dtmodel", pkg)

# Verify before unpacking: a truncated fetch must fail here, not as a broken
# ORT session hours later.
digest = hashlib.sha256(pkg.read_bytes()).hexdigest()
if want and digest != want:
    raise SystemExit(f"[sam21] sha256 MISMATCH\n  want {want}\n  got  {digest}")
print(f"[sam21] {MODEL_ID} v{version} sha256 {'verified' if want else 'UNPINNED'} ({digest[:16]}…)")

if not src.exists():
    with zipfile.ZipFile(pkg) as zf:
        zf.extractall(SPIKE)
    if not src.exists():
        raise SystemExit(f"[sam21] unexpected archive layout in {pkg}")

enc_src, dec_src = src / "encoder.onnx", src / "decoder.onnx"
for p in (enc_src, dec_src):
    if not p.exists():
        raise SystemExit(f"[sam21] missing {p}")


def describe(path):
    m = onnx.load(str(path), load_external_data=False)

    def io(vals):
        out = []
        for v in vals:
            d = v.type.tensor_type
            dims = [x.dim_param or x.dim_value for x in d.shape.dim]
            out.append(f"{v.name}{dims}:{onnx.TensorProto.DataType.Name(d.elem_type)}")
        return out
    return io(m.graph.input), io(m.graph.output)


for name, p in (("encoder", enc_src), ("decoder", dec_src)):
    i, o = describe(p)
    print(f"[sam21] {name} {mb(p):.1f}MB\n  in : {i}\n  out: {o}")
if args.inspect:
    sys.exit(0)


# --- probe: one real fp32 run gives both the concrete shapes to freeze and the
#     reference outputs to validate against.
import onnxruntime as ort  # noqa: E402  (heavy; only needed past --inspect)

CPU = ["CPUExecutionProvider"]

# Validate on a PHOTOGRAPH, never on noise. On noise the mask logits hover at
# zero across the whole frame, so any fp16 delta flips the sign and IoU collapses
# — it measures the threshold, not the conversion. On a real image the logits are
# bimodal and only a thin boundary band is near zero.
FIXTURE = ROOT / "spikes" / "yoloe" / "bus.jpg"
CLICKS = [(512, 470), (300, 560), (720, 520), (170, 830)]  # bus body, windows, kerbside figure
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def fixture_tensor():
    from PIL import Image
    im = Image.open(FIXTURE).convert("RGB").resize((1024, 1024), Image.BICUBIC)
    x = (np.asarray(im, np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)[None].copy()


enc_names = ["high_res_feats_0", "high_res_feats_1", "image_embed"]
enc_feed = {"image": fixture_tensor()}
probe_cache = SPIKE / f"probe-{args.scale}.npz"
if probe_cache.exists():  # a 1024² fp32 CPU encode is ~30s; cache it across runs
    z = np.load(probe_cache)
    enc_ref = [z[n] for n in enc_names]
    print(f"[sam21] probe: reusing {probe_cache.name}")
else:
    print(f"[sam21] probing encoder (fp32, CPU) on {FIXTURE.name} ...")
    enc32 = ort.InferenceSession(str(enc_src), providers=CPU)
    enc_names = [o.name for o in enc32.get_outputs()]
    enc_ref = enc32.run(None, enc_feed)
    np.savez(probe_cache, **dict(zip(enc_names, enc_ref)))
    del enc32
enc_shapes = {n: list(v.shape) for n, v in zip(enc_names, enc_ref)}
for n in enc_names:
    print(f"  {n} -> {enc_shapes[n]}")


# Exactly as many slots as clicks — no -1 padding; see the num_points note above.
def decoder_feed(ref, click=CLICKS[0], dtype=np.float32):
    pts = click if isinstance(click, list) else [click]
    coords = np.array([pts], np.float32)
    labels = np.ones((1, len(pts)), np.float32)
    feed = {
        "image_embed": ref[enc_names.index("image_embed")],
        "high_res_feats_0": ref[enc_names.index("high_res_feats_0")],
        "high_res_feats_1": ref[enc_names.index("high_res_feats_1")],
        "point_coords": coords,
        "point_labels": labels,
        "mask_input": np.zeros((1, 1, 256, 256), np.float32),
        "has_mask_input": np.zeros((1,), np.float32),
    }
    return {k: v.astype(dtype) for k, v in feed.items()}


print("[sam21] probing decoder (fp32, CPU) ...")
dec32 = ort.InferenceSession(str(dec_src), providers=CPU)
dec_feed = decoder_feed(enc_ref)
dec_ref = dec32.run(None, dec_feed)
dec_names = [o.name for o in dec32.get_outputs()]
dec_shapes = {n: list(v.shape) for n, v in zip(dec_names, dec_ref)}
for n in dec_names:
    print(f"  {n} -> {dec_shapes[n]}")


# --- freeze shapes to the probed values
def set_dims(vi, shape):
    dim = vi.type.tensor_type.shape.dim
    if len(dim) != len(shape):
        raise SystemExit(f"[sam21] rank mismatch freezing {vi.name}: {len(dim)} vs {shape}")
    for d, v in zip(dim, shape):
        d.ClearField("dim_param")
        d.dim_value = int(v)


def freeze(path, dst, out_shapes, dim_params=()):
    m = onnx.load(str(path))
    for name, value in dim_params:  # symbolic INPUT dims (decoder num_points)
        for vi in list(m.graph.input) + list(m.graph.value_info):
            for d in vi.type.tensor_type.shape.dim:
                if d.dim_param == name:
                    d.ClearField("dim_param")
                    d.dim_value = value
    for vi in m.graph.output:
        if vi.name in out_shapes:
            set_dims(vi, out_shapes[vi.name])
    onnx.save(m, str(dst))
    return dst


enc_static = freeze(enc_src, SPIKE / "encoder.static.onnx", enc_shapes)
dec_static = freeze(dec_src, SPIKE / "decoder.static.onnx", dec_shapes)
print('[sam21] shapes frozen (encoder outputs + mask count; num_points stays dynamic)')


# --- mixed-precision fp16
FP32, FP16 = onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16


def repair_casts(m):
    """onnxconverter-common retypes a pre-existing Cast node's *declared output*
    to fp16 but leaves its `to` attribute at FLOAT, so ORT rejects the model
    ("Type (tensor(float16)) ... does not match expected type (tensor(float))").
    The graph declaration is what ORT type-checks, so make the attribute agree."""
    decl = {v.name: v.type.tensor_type.elem_type
            for v in list(m.graph.value_info) + list(m.graph.output)}
    fixed = []
    for n in m.graph.node:
        if n.op_type != "Cast":
            continue
        want = decl.get(n.output[0])
        for a in n.attribute:
            if a.name == "to" and want in (FP16, FP32) and a.i != want:
                a.i = want
                fixed.append(n.name)
    return fixed


def to_fp16(src_path, dst_path, block):
    m = onnx.load(str(src_path))
    conv = float16.convert_float_to_float16(
        m, keep_io_types=False, op_block_list=block or None, disable_shape_infer=False,
    )
    fixed = repair_casts(conv)
    if fixed:
        print(f"[sam21]   repaired {len(fixed)} Cast node(s): {', '.join(fixed[:3])}…")
    onnx.checker.check_model(conv, full_check=False)
    onnx.save(conv, str(dst_path), save_as_external_data=False)
    return dst_path


# fp32 is what SHIPS — ORT-Web's WebGPU fp16 kernels produce a wrong embedding
# for this model (spikes/sam21/FINDINGS.md §1.2). These are the frozen graphs,
# unconverted; fp16 is still built below so the swap is one line when fixed.
shutil.copyfile(enc_static, OUT / "encoder.fp32.onnx")
shutil.copyfile(dec_static, OUT / "decoder.fp32.onnx")
print(f"[sam21] fp32 (shipping): encoder {mb(OUT / 'encoder.fp32.onnx'):.1f}MB "
      f"decoder {mb(OUT / 'decoder.fp32.onnx'):.1f}MB")

block = args.block or []
print(f"[sam21] fp16 block list: {block or '(none — blanket fp16)'}")
enc_fp16 = to_fp16(enc_static, OUT / "encoder.fp16.onnx", block)
dec_fp16 = to_fp16(dec_static, OUT / "decoder.fp16.onnx", block)
print(f"[sam21] encoder {mb(enc_src):.1f} -> {mb(enc_fp16):.1f}MB")
print(f"[sam21] decoder {mb(dec_src):.1f} -> {mb(dec_fp16):.1f}MB")


# --- numeric check: onnxconverter-common has mistyped heads before (see
#     export-yoloe.py), so never ship a conversion that was not compared.
iou, ship = None, None
if not args.skip_validate:
    MI, II = dec_names.index("masks"), dec_names.index("iou_predictions")

    def dilate(m):
        o = m.copy()
        o[1:, :] |= m[:-1, :]; o[:-1, :] |= m[1:, :]
        o[:, 1:] |= m[:, :-1]; o[:, :-1] |= m[:, 1:]
        return o

    def boundary_band(m):
        """Pixels within 1 of the mask boundary — where sub-pixel jitter lives."""
        return dilate(m) & ~(~dilate(~m))

    def agreement(ref_out, got_out):
        """Compare like with like: SAM emits 3 ambiguity candidates and the
        shipped mask is the highest predicted-IoU one, so score fp16 on fp32's
        chosen index. A different argmax is a different mask, not a rounding
        error, and is reported separately."""
        k = int(ref_out[II].astype(np.float32).ravel().argmax())
        rl = ref_out[MI].astype(np.float32)[0, k]
        gl = got_out[MI].astype(np.float32)[0, k]
        rm, gm = rl > 0, gl > 0
        union = (rm | gm).sum()
        cos = float(rl.ravel() @ gl.ravel()
                    / (np.linalg.norm(rl) * np.linalg.norm(gl) + 1e-12))
        off = int(((rm ^ gm) & ~boundary_band(rm)).sum())
        return {
            "iou": float((rm & gm).sum() / union) if union else 1.0,
            "cos": cos,
            "offBand": off / max(int(rm.sum()), 1),
            "sameMask": k == int(got_out[II].astype(np.float32).ravel().argmax()),
        }

    enc16 = ort.InferenceSession(str(enc_fp16), providers=CPU)
    enc_got = enc16.run(None, {k: v.astype(np.float16) for k, v in enc_feed.items()})
    print("[sam21] encoder fp32-vs-fp16:")
    for n, a, b in zip(enc_names, enc_ref, enc_got):
        a, b = a.astype(np.float32), b.astype(np.float32)
        cos = float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"  {n:18s} {tuple(a.shape)} cos={cos:.6f} maxabs={np.abs(a - b).max():.5f}")

    # Which side loses precision? Cross the two encoders with the two decoders:
    # the shipped pair is fp16/fp16, the other two localise any regression.
    dec16 = ort.InferenceSession(str(dec_fp16), providers=CPU)
    enc16_as32 = [v.astype(np.float32) for v in enc_got]
    lanes = {
        "fp16 enc -> fp32 dec": (dec32, enc16_as32, np.float32),
        "fp32 enc -> fp16 dec": (dec16, enc_ref, np.float16),
        "fp16 enc -> fp16 dec": (dec16, enc_got, np.float16),
    }
    print(f"[sam21] vs fp32/fp32, {len(CLICKS)} clicks on {FIXTURE.name}:")
    scores = {}
    for label, (sess, ref_in, dt) in lanes.items():
        per = [agreement(dec32.run(None, decoder_feed(enc_ref, c, np.float32)),
                         sess.run(None, decoder_feed(ref_in, c, dt))) for c in CLICKS]
        scores[label] = {
            "cos": min(p["cos"] for p in per),
            "offBand": max(p["offBand"] for p in per),
            "sameMask": sum(p["sameMask"] for p in per),
            "iou": min(p["iou"] for p in per),
        }
        s = scores[label]
        print(f"  {label}  logit-cos={s['cos']:.6f} off-band={s['offBand']:.4f} "
              f"same-mask={s['sameMask']}/{len(per)} (raw-IoU={s['iou']:.4f})")

    ship = scores["fp16 enc -> fp16 dec"]
    iou = ship["iou"]
    fails = []
    if ship["cos"] < COS_GATE:
        fails.append(f"logit cosine {ship['cos']:.6f} < {COS_GATE}")
    if ship["offBand"] > OFFBAND_GATE:
        fails.append(f"off-band {ship['offBand']:.4f} > {OFFBAND_GATE}")
    if ship["sameMask"] != len(CLICKS):
        fails.append(f"mask choice differs on {len(CLICKS) - ship['sameMask']} click(s)")
    if fails:
        raise SystemExit("[sam21] fp16 REJECTED: " + "; ".join(fails)
                         + ". Populate --block and re-run.")
    print("[sam21] fp16 accepted")


# --- No .ort step. §7 step 5 called it mandatory ("ORT-Web's graph optimizer
#     crashes on SAM2's transpose pass"); measured on the pinned runtime the
#     reverse holds — the .onnx loads and both .ort styles fail with
#     "[Transpose] blocks.0/attn/Transpose_1 … no GPU data for input", and Fixed
#     style also inflates 78 MB -> 156 MB by undoing fp16. See FINDINGS.md §1.3.

# --- provenance, so a cached model's origin is inspectable at runtime
(OUT / "model.json").write_text(json.dumps({
    "id": MODEL_ID,
    "release": RELEASE,
    "version": version,
    "sha256": digest,
    "scale": args.scale,
    "precision": "fp32 (shipping) + fp16 (kept, see FINDINGS §1.2)",
    "fp16BlockList": block,
    "numPoints": "dynamic",  # padding degrades masks; pass exactly the clicks
    "encoderInput": [1, 3, 1024, 1024],
    "shapes": {**enc_shapes, **dec_shapes},
    "masks": dec_shapes["masks"][1],  # ambiguity candidates; pick by iou_predictions
    "validation": ship,
}, indent=2) + "\n")

for p in sorted(OUT.glob("*")):
    print(f"[sam21] {mb(p):8.1f}MB  {p.name}")
print("[sam21] done ->", OUT)
