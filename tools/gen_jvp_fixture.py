#!/usr/bin/env python3
"""Generates the forward-AD golden fixture: a small DiT-like graph whose
weights, inputs, tangents, forward output, and `torch.func.jvp` result are
saved to `tools/fixtures/jvp.safetensors`. The Rust golden test
(`tests/golden.rs`) rebuilds the identical graph in candle and compares
`candle_core::forward_ad::jvp` against the saved reference (issue #8).

Usage:  python3 tools/gen_jvp_fixture.py
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

torch.manual_seed(0)

B, T, D_IN, D_H = 2, 6, 16, 32

weights = {
    "w1": torch.randn(D_H, D_IN) * 0.3,
    "b1": torch.randn(D_H) * 0.1,
    "wq": torch.randn(D_H, D_H) * 0.2,
    "bq": torch.randn(D_H) * 0.1,
    "wk": torch.randn(D_H, D_H) * 0.2,
    "bk": torch.randn(D_H) * 0.1,
    "wv": torch.randn(D_H, D_H) * 0.2,
    "bv": torch.randn(D_H) * 0.1,
    "w2": torch.randn(D_IN, D_H) * 0.2,
    "b2": torch.randn(D_IN) * 0.1,
}


def f(x, t):
    """Mini DiT-ish graph: linear -> silu -> layernorm -> self-attention
    (softmax) -> residual -> timestep modulation -> tanh -> linear.
    Mirrored 1:1 in tests/golden.rs."""
    h = x @ weights["w1"].T + weights["b1"]
    h = F.silu(h)
    h = F.layer_norm(h, (D_H,), eps=1e-5)
    q = h @ weights["wq"].T + weights["bq"]
    k = h @ weights["wk"].T + weights["bk"]
    v = h @ weights["wv"].T + weights["bv"]
    a = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(D_H), dim=-1)
    h = h + a @ v
    h = h * (1.0 + t)[:, None, None]
    h = torch.tanh(h)
    return h @ weights["w2"].T + weights["b2"]


x = torch.randn(B, T, D_IN)
t = torch.rand(B)
vx = torch.randn(B, T, D_IN)
vt = torch.ones(B)

y_ref, jvp_ref = torch.func.jvp(f, (x, t), (vx, vt))

out = Path(__file__).parent / "fixtures" / "jvp.safetensors"
out.parent.mkdir(parents=True, exist_ok=True)
tensors = {k: v.contiguous() for k, v in weights.items()}
tensors.update(
    {
        "x": x,
        "t": t,
        "vx": vx,
        "vt": vt,
        "y_ref": y_ref.contiguous(),
        "jvp_ref": jvp_ref.contiguous(),
    }
)
save_file(tensors, str(out))
print(f"wrote {out} ({out.stat().st_size} bytes)")
print(f"torch {torch.__version__}; |y|_max={y_ref.abs().max():.4f} |jvp|_max={jvp_ref.abs().max():.4f}")
