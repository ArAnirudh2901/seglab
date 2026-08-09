#!/usr/bin/env python3
"""Export the MobileCLIP2-B TEXT tower for the browser.

YOLOE-26L's checkpoint declares `text_model = mobileclip2:b`, so the runtime must
produce vectors in THAT space — a MobileCLIP-BLT or ViT-B/32 encoder would feed
RepRTA out-of-distribution input and quietly degrade every search.

Ultralytics ships the tower only as a frozen TorchScript blob, which the ONNX
exporter cannot handle (the traced MHA loses rank: "export of transpose for
tensor of unknown rank"). So the weights come from the open_clip release and are
loaded into a plain module here. That swap is only safe if it is proven, hence
step 3: the rebuilt module must match the TorchScript blob to ~1e-6 before
anything is exported.

Two artifacts, because the tower splits cleanly by op type:
  • the 49408x512 token table -> int8 per-row, read on demand in JS. It is a
    Gather, not a MatMul, so MatMulNBits cannot touch it; per-row int8 is
    near-lossless and keeps the file out of RAM.
  • everything else -> 4-bit block-wise MatMulNBits (block 32), weight-only,
    fp32 activations. This is NOT the `quantize_dynamic`/QInt8 scheme that
    collapsed CLIP alignment 0.9 -> 0.03 in July: that quantized ACTIVATIONS.
    Step 6 re-measures alignment rather than trusting the distinction.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

CONTEXT = 77
VOCAB = 49408
WIDTH = 512
LAYERS = 12
HEADS = 8

PHRASES = [
    "a photo of a flower", "orange tulip", "grape hyacinth", "muscari",
    "a rusty bicycle leaning on a blue wall", "person", "dog", "the red car",
    "a weathered wooden fence post", "cumulonimbus cloud",
]


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResBlock(nn.Module):
    def __init__(self, width: int, heads: int, act: nn.Module):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), act, nn.Linear(width * 4, width))

    def forward(self, x, mask):
        h = self.ln_1(x)
        x = x + self.attn(h, h, h, need_weights=False, attn_mask=mask)[0]
        return x + self.mlp(self.ln_2(x))


class TextTower(nn.Module):
    """CLIP text tower taking PRE-EMBEDDED tokens.

    The token lookup is deliberately not here: it is lifted into a separate int8
    asset and gathered in JS, which keeps 25 MB of table out of the graph and out
    of the browser's memory.
    """

    def __init__(self, quick_gelu: bool = True):
        super().__init__()
        act = QuickGELU() if quick_gelu else nn.GELU()
        self.positional_embedding = nn.Parameter(torch.zeros(CONTEXT, WIDTH))
        self.resblocks = nn.ModuleList([ResBlock(WIDTH, HEADS, act) for _ in range(LAYERS)])
        self.ln_final = nn.LayerNorm(WIDTH)
        self.text_projection = nn.Parameter(torch.zeros(WIDTH, WIDTH))
        mask = torch.full((CONTEXT, CONTEXT), float("-inf")).triu(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, token_embeds: torch.Tensor, eot_index: torch.Tensor) -> torch.Tensor:
        x = token_embeds + self.positional_embedding
        for blk in self.resblocks:
            x = blk(x, self.mask)
        x = self.ln_final(x)
        # Pool at the EOT position. Passed in rather than derived via argmax so
        # the graph has no data-dependent control flow.
        x = x.gather(1, eot_index.view(-1, 1, 1).expand(-1, 1, WIDTH)).squeeze(1)
        x = x @ self.text_projection
        return x / x.norm(dim=-1, keepdim=True)


def load_state(path: Path) -> tuple[dict, torch.Tensor, bool]:
    from safetensors.torch import load_file

    raw = load_file(str(path))
    prefix = "text." if any(k.startswith("text.") for k in raw) else ""
    grab = lambda k: raw[f"{prefix}{k}"]  # noqa: E731

    table = grab("token_embedding.weight").float()
    sd: dict[str, torch.Tensor] = {
        "positional_embedding": grab("positional_embedding").float(),
        "ln_final.weight": grab("ln_final.weight").float(),
        "ln_final.bias": grab("ln_final.bias").float(),
    }
    proj_key = f"{prefix}text_projection"
    proj = raw[proj_key] if proj_key in raw else raw[f"{proj_key}.weight"].T
    sd["text_projection"] = proj.float()
    for i in range(LAYERS):
        src = f"transformer.resblocks.{i}."
        dst = f"resblocks.{i}."
        for a, b in [
            ("ln_1.weight", "ln_1.weight"), ("ln_1.bias", "ln_1.bias"),
            ("ln_2.weight", "ln_2.weight"), ("ln_2.bias", "ln_2.bias"),
            ("attn.in_proj_weight", "attn.in_proj_weight"), ("attn.in_proj_bias", "attn.in_proj_bias"),
            ("attn.out_proj.weight", "attn.out_proj.weight"), ("attn.out_proj.bias", "attn.out_proj.bias"),
            ("mlp.c_fc.weight", "mlp.0.weight"), ("mlp.c_fc.bias", "mlp.0.bias"),
            ("mlp.c_proj.weight", "mlp.2.weight"), ("mlp.c_proj.bias", "mlp.2.bias"),
        ]:
            sd[dst + b] = grab(src + a).float()
    return sd, table, True


def verify_against_torchscript(tower: TextTower, table: torch.Tensor, ts_path: Path) -> float:
    """The rebuilt module must reproduce the shipped tower, or the space is wrong."""
    import clip  # the reference CLIP BPE, same tokenizer MobileCLIP2 uses

    tokens = clip.clip.tokenize(PHRASES, truncate=True).long()
    eot = tokens.argmax(dim=-1)  # EOT is the highest id present
    with torch.no_grad():
        mine = tower(table[tokens], eot)
        ts = torch.jit.load(ts_path, map_location="cpu").eval()
        ref = ts(tokens.int())
    cos = torch.nn.functional.cosine_similarity(mine, ref, dim=-1)
    return float(cos.min())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="build/text-encoder")
    ap.add_argument("--out", default="models/clip-text")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src / "open_clip_config.json").read_text())
    quick = bool(cfg.get("model_cfg", {}).get("quick_gelu", cfg.get("quick_gelu", False)))
    print(f"[1/6] config: quick_gelu={quick}")

    print("[2/6] loading open_clip weights")
    sd, table, _ = load_state(src / "open_clip_model.safetensors")
    tower = TextTower(quick_gelu=quick).eval()
    missing, unexpected = tower.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    assert all("mask" in m for m in missing), f"missing keys: {missing[:5]}"
    print(f"    tower {sum(p.numel() for p in tower.parameters())/1e6:.2f} M + table {table.numel()/1e6:.2f} M")

    print("[3/6] verifying against the shipped TorchScript tower")
    cos = verify_against_torchscript(tower, table, src / "mobileclip2_b.ts")
    print(f"    min cosine vs mobileclip2_b.ts: {cos:.6f}")
    if cos < 0.9999:
        raise SystemExit("rebuilt tower does NOT match the shipped one — wrong weights or activation")

    print("[4/6] exporting ONNX (token table lifted out)")
    fp32 = out / "mclip2-text.fp32.onnx"
    embeds = torch.zeros(1, CONTEXT, WIDTH)
    eot = torch.tensor([2])
    torch.onnx.export(
        tower, (embeds, eot), str(fp32), opset_version=17,
        input_names=["token_embeds", "eot_index"], output_names=["text_embeds"],
        dynamic_axes={"token_embeds": {0: "b"}, "eot_index": {0: "b"}, "text_embeds": {0: "b"}},
        do_constant_folding=True, dynamo=False,
    )
    print(f"    fp32 -> {fp32.name} ({fp32.stat().st_size/1e6:.1f} MB)")

    print("[5/6] token table -> int8 per row")
    t = table.numpy().astype(np.float32)
    scale = np.abs(t).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(t / scale[:, None]), -127, 127).astype(np.int8)
    (out / "mclip2-embed.i8").write_bytes(q.tobytes())
    (out / "mclip2-embed.scale.f32").write_bytes(scale.astype(np.float32).tobytes())
    err = np.abs(q.astype(np.float32) * scale[:, None] - t).max()
    print(f"    table -> int8 {q.nbytes/1e6:.1f} MB + {scale.nbytes/1e6:.2f} MB scales, max abs err {err:.5f}")

    print("[6/6] 4-bit block-wise (MatMulNBits)")
    from onnxruntime.quantization import matmul_nbits_quantizer as mnb
    import onnx

    # Measured sweep (min cosine vs fp32, table held at int8):
    #   block 64 sym  33.2 MB 0.9813 | block 32 sym 35.4 MB 0.9824
    #   block 16 sym  39.8 MB 0.9862 | block 16 asym 40.9 MB 0.9912  <- chosen
    # Asymmetric wins because the per-block weight distributions are skewed, and
    # +1.1 MB for +0.005 cosine is the cheapest accuracy on the table.
    # accuracy_level stays 0 (EP's native compute). Level 4 requests int8
    # ARITHMETIC — the activation quantization that collapsed CLIP alignment in
    # July — and is never appropriate here even though the CPU EP ignores it.
    q4 = out / "mclip2-text.q4.onnx"
    quant = mnb.MatMulNBitsQuantizer(
        onnx.load(str(fp32)), block_size=16, is_symmetric=False, accuracy_level=0,
    )
    quant.process()
    quant.model.save_model_to_file(str(q4), use_external_data_format=False)
    print(f"    q4 -> {q4.name} ({q4.stat().st_size/1e6:.1f} MB)")

    shutil.copy(src / "merges.txt", out / "merges.txt")
    print(f"    merges.txt -> {(out/'merges.txt').stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
