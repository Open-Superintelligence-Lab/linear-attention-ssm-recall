"""RWKV-7 "Goose" time mixing (arXiv 2503.14456), laptop size.

Keeps: token shift (static per-channel lerp with the previous token), vector decay
log w = -exp(-0.5) * sigmoid(w0 + lora(x)), in-context learning rate a = sigmoid(a0 + lora(x))
(vector), removal key kappa = L2norm(k * k_k), replacement (write) key k~ = k * (1 + (a - 1) * k_a),
generalized delta transition diag(w) - (a * kappa) kappa^T, bonus term (r . (u * k~)) v,
per-head LayerNorm on the output, low-rank sigmoid output gate, receptance r as the query.
Drops: value residual across layers, channel mixing (all models share the same SwiGLU MLP),
the CUDA kernel.
Knobs: removal_key = rwkv7 (default) | tied (kappa = L2norm(k~)) | separate (own projection);
decay=scalar (one decay per head); token_shift=0 removes the
shift; conv > 0 adds a short conv on r, k, v; decay_init moves the w0 ramp.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common import ShortConv, split, merge, l2norm
from env.dplr import dplr

W0_RAMP = {"short": (-3.0, 2.0), "default": (-6.0, -1.0), "long": (-9.0, -4.0)}


class Mixer(nn.Module):
    def __init__(self, d, cfg):
        super().__init__()
        H, dk, dv = int(cfg["n_heads"]), int(cfg["head_k"]), int(cfg["head_v"])
        self.H, self.dk, self.dv = H, dk, dv
        self.vector = cfg["decay"] == "vector"
        self.removal = cfg["removal_key"] if cfg["removal_key"] in ("rwkv7", "tied", "separate") else "rwkv7"
        self.token_shift = int(cfg["token_shift"])
        self.chunk, self.mode = int(cfg["chunk"]), cfg["dplr_mode"]
        self.mu = nn.Parameter(torch.full((6, d), 0.5))           # r, w, k, v, a, g
        self.wr = nn.Linear(d, H * dk, bias=False)
        self.wk = nn.Linear(d, H * dk, bias=False)
        self.wv = nn.Linear(d, H * dv, bias=False)
        self.wkk = nn.Linear(d, H * dk, bias=False) if self.removal == "separate" else None
        kc = int(cfg["conv"])
        self.conv = ShortConv(H * (2 * dk + dv), kc, act=False) if kc > 0 else None
        nw = H * dk if self.vector else H
        lo, hi = W0_RAMP[cfg["decay_init"]]
        self.w0 = nn.Parameter(torch.linspace(lo, hi, nw))
        self.w1 = nn.Linear(d, 16, bias=False)
        self.w2 = nn.Linear(16, nw, bias=False)
        nn.init.zeros_(self.w2.weight)
        self.a0 = nn.Parameter(torch.zeros(H * dk))
        self.a1 = nn.Linear(d, 16, bias=False)
        self.a2 = nn.Linear(16, H * dk, bias=False)
        nn.init.zeros_(self.a2.weight)
        self.g1 = nn.Linear(d, 16, bias=False)
        self.g2 = nn.Linear(16, H * dv, bias=False)
        self.k_k = nn.Parameter(torch.full((H * dk,), 0.85))
        self.k_a = nn.Parameter(torch.ones(H * dk))
        self.u = nn.Parameter(torch.zeros(H, 1, dk))
        self.onorm = nn.LayerNorm(dv, bias=False)
        self.out = nn.Linear(H * dv, d, bias=False)

    def state_floats(self, L):
        return self.H * self.dk * self.dv

    def forward(self, x):
        B, L, _ = x.shape
        H, dk, dv = self.H, self.dk, self.dv
        if self.token_shift:
            xx = F.pad(x, (0, 0, 1, 0))[:, :-1] - x
            xr, xw, xk, xv, xa, xg = (x + xx * self.mu[i] for i in range(6))
        else:
            xr = xw = xk = xv = xa = xg = x
        r, k, v = self.wr(xr), self.wk(xk), self.wv(xv)
        if self.conv is not None:
            r, k, v = self.conv(torch.cat([r, k, v], -1)).split([H * dk, H * dk, H * dv], -1)
        logw = -math.exp(-0.5) * torch.sigmoid(self.w0 + self.w2(torch.tanh(self.w1(xw))))
        a = torch.sigmoid(self.a0 + self.a2(self.a1(xa)))
        g = self.g2(torch.sigmoid(self.g1(xg)))
        kt = k * (1 + (a - 1) * self.k_a)                          # write key
        if self.removal == "tied":
            kap = l2norm(split(kt, H))
        elif self.removal == "separate":
            kap = l2norm(split(self.wkk(xk), H))
        else:
            kap = l2norm(split(k * self.k_k, H))
        r, kt, v, a = split(r, H), split(kt, H), split(v, H), split(a, H)
        logw = split(logw, H)
        o, _ = dplr(r, kt, v, logw, b=a * kap, c=kap, chunk=self.chunk, mode=self.mode)
        o = o + (r * self.u * kt).sum(-1, keepdim=True) * v
        return self.out(merge(self.onorm(o)) * g)


def build(d, cfg):
    return Mixer(d, cfg)
