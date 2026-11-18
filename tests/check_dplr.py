"""Check the chunked DPLR engine against the plain loop, then a forward/backward pass of every model.

python3 tests/check_dplr.py
Run this first on a new machine. It prints the largest output and gradient differences for
each recurrence type (linear attention, scalar decay, vector decay, delta rule, RWKV-7
style) and exits non-zero if any difference is above 1e-4 (float64).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from env.dplr import dplr_recurrent, dplr_chunked  # noqa: E402
from env.config import resolve, ALL_METHODS  # noqa: E402
from env.model import LM  # noqa: E402

torch.manual_seed(0)
B, H, L, dk, dv = 2, 2, 45, 8, 6           # L not a multiple of the chunk: tests padding


def case(kind):
    q, k = torch.randn(B, H, L, dk), torch.randn(B, H, L, dk)
    v = torch.randn(B, H, L, dv)
    b = c = None
    if kind == "linear":
        logw = torch.zeros(B, H, L, 1)
    elif kind == "scalar":
        logw = -F.softplus(torch.randn(B, H, L, 1))
    elif kind == "vector":
        logw = -F.softplus(torch.randn(B, H, L, dk))
    elif kind == "delta":                      # Gated DeltaNet: b = alpha*beta*k, c = k (unit k)
        k = F.normalize(k, dim=-1)
        beta = torch.rand(B, H, L, 1)
        logw = -F.softplus(torch.randn(B, H, L, 1))
        c = k
        b = torch.exp(logw) * beta * k
        k = beta * k
    elif kind == "rwkv7":                      # vector decay, b = a*kappa, c = kappa
        logw = -0.6 * torch.sigmoid(torch.randn(B, H, L, dk))
        kap = F.normalize(torch.randn(B, H, L, dk), dim=-1)
        a = torch.rand(B, H, L, dk)
        b, c = a * kap, kap
    t = [x.double().requires_grad_() if x is not None else None for x in (q, k, v, logw, b, c)]
    return t


def grads(fn, t):
    o, S = fn(*t)
    loss = (o * torch.linspace(-1, 1, o.shape[-1], dtype=o.dtype)).sum() + S.square().sum()
    g = torch.autograd.grad(loss, [x for x in t if x is not None])
    return o.detach(), S.detach(), g


worst = 0.0
for kind in ("linear", "scalar", "vector", "delta", "rwkv7"):
    t = case(kind)
    o1, S1, g1 = grads(lambda *a: dplr_recurrent(*a), t)
    for chunk in (8, 16, 64):
        o2, S2, g2 = grads(lambda *a: dplr_chunked(*a, chunk=chunk), t)
        do = (o1 - o2).abs().max().item()
        dS = (S1 - S2).abs().max().item()
        dg = max((a - b).abs().max().item() for a, b in zip(g1, g2))
        worst = max(worst, do, dS, dg)
        print(f"{kind:7s} chunk {chunk:3d}  out {do:.1e}  state {dS:.1e}  grad {dg:.1e}")

for m in ALL_METHODS:
    cfg = resolve(dict(method=m))
    model = LM(cfg)
    x = torch.randint(0, 512, (2, 64))
    out = model(x)
    out.sum().backward()
    ok = torch.isfinite(out).all().item() and all(torch.isfinite(p.grad).all().item()
                                                  for p in model.parameters() if p.grad is not None)
    print(f"{m:15s} forward/backward {'ok' if ok else 'NON-FINITE'}  params {sum(p.numel() for p in model.parameters())}")
    worst = worst if ok else float("inf")

print("PASS" if worst < 1e-4 else f"FAIL (worst difference {worst:.1e})")
sys.exit(0 if worst < 1e-4 else 1)
