"""Gated Linear Attention (GLA, arXiv 2312.06635), laptop size.

Keeps: per-key-channel data-dependent decay alpha_t = sigmoid(low-rank(x))^(1/16)
(log w = logsigmoid(.) / 16), q scaled by dk^-0.5, per-head RMSNorm on the output,
SiLU output gate, matrix state dk x dv per head.
Drops: the hardware-efficient two-level chunk kernel (this repo's generic engine is used).
Knobs: decay=scalar (one gate per head), conv > 0 adds a short conv + SiLU on q, k, v,
qk_norm=1 L2-normalizes q and k, decay_init sets the gate bias (short -3, default 0, long +4).
"""
import torch.nn as nn
import torch.nn.functional as F

from baselines.common import RMSNorm, ShortConv, split, merge, l2norm
from env.dplr import dplr

GATE_BIAS = {"short": -3.0, "default": 0.0, "long": 4.0}


class Mixer(nn.Module):
    def __init__(self, d, cfg):
        super().__init__()
        H, dk, dv = int(cfg["n_heads"]), int(cfg["head_k"]), int(cfg["head_v"])
        self.H, self.dk, self.dv = H, dk, dv
        self.vector = cfg["decay"] == "vector"
        self.qk_norm = int(cfg["qk_norm"])
        self.chunk, self.mode = int(cfg["chunk"]), cfg["dplr_mode"]
        self.qkv = nn.Linear(d, H * (2 * dk + dv), bias=False)
        kc = int(cfg["conv"])
        self.conv = ShortConv(H * (2 * dk + dv), kc, act=kc > 0)
        self.gk1 = nn.Linear(d, 16, bias=False)
        self.gk2 = nn.Linear(16, H * dk if self.vector else H, bias=True)
        nn.init.constant_(self.gk2.bias, GATE_BIAS[cfg["decay_init"]])
        self.g = nn.Linear(d, H * dv, bias=False)
        self.onorm = RMSNorm(dv)
        self.out = nn.Linear(H * dv, d, bias=False)

    def state_floats(self, L):
        return self.H * self.dk * self.dv

    def forward(self, x):
        B, L, _ = x.shape
        H, dk, dv = self.H, self.dk, self.dv
        q, k, v = self.conv(self.qkv(x)).split([H * dk, H * dk, H * dv], -1)
        q, k, v = split(q, H), split(k, H), split(v, H)
        if self.qk_norm:
            q, k = l2norm(q), l2norm(k)
        q = q * dk ** -0.5
        logw = F.logsigmoid(self.gk2(self.gk1(x))) / 16.0
        logw = split(logw, H)                                   # (B, H, L, dk or 1)
        o, _ = dplr(q, k, v, logw, chunk=self.chunk, mode=self.mode)
        return self.out(merge(self.onorm(o)) * F.silu(self.g(x)))


def build(d, cfg):
    return Mixer(d, cfg)
