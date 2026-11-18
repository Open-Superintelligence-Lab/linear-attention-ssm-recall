"""Gated DeltaNet (arXiv 2412.06464), laptop size.

Keeps: gated delta rule S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T,
L2-normalized q and k, short causal conv + SiLU on q, k, v, beta = sigmoid(W x),
Mamba-2 style decay log alpha = -exp(A_log) * softplus(W x + dt_bias), per-head RMSNorm
on the output, SiLU output gate.
Drops: the WY/UT chunk kernel (this repo's generic engine solves the same triangular
system), the hybrid H1/H2 attention layers (use the hybrid knob instead).
Knobs: beta_range=2 allows beta in (0, 2) (negative eigenvalues, arXiv 2411.12537),
decay=vector gives a per-key-channel alpha, removal_key=separate removes along a key from
its own projection instead of the write key.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common import RMSNorm, ShortConv, split, merge, l2norm, dt_init, a_range
from env.dplr import dplr


class Mixer(nn.Module):
    def __init__(self, d, cfg):
        super().__init__()
        H, dk, dv = int(cfg["n_heads"]), int(cfg["head_k"]), int(cfg["head_v"])
        self.H, self.dk, self.dv = H, dk, dv
        self.vector = cfg["decay"] == "vector"
        self.beta_range = float(cfg["beta_range"])
        self.separate = cfg["removal_key"] == "separate"
        self.chunk, self.mode = int(cfg["chunk"]), cfg["dplr_mode"]
        n_k = 3 if self.separate else 2
        self.qkv = nn.Linear(d, H * (n_k * dk + dv), bias=False)
        self.conv = ShortConv(H * (n_k * dk + dv), int(cfg["conv"]), act=True)
        self.n_k = n_k
        self.wb = nn.Linear(d, H, bias=False)
        na = H * dk if self.vector else H
        self.wa = nn.Linear(d, na, bias=False)
        self.dt_bias = nn.Parameter(dt_init(na, cfg["decay_init"]))
        self.A_log = nn.Parameter(torch.log(torch.empty(na).uniform_(*a_range(cfg))))
        self.g = nn.Linear(d, H * dv, bias=False)
        self.onorm = RMSNorm(dv)
        self.out = nn.Linear(H * dv, d, bias=False)

    def state_floats(self, L):
        return self.H * self.dk * self.dv

    def forward(self, x):
        B, L, _ = x.shape
        H, dk, dv = self.H, self.dk, self.dv
        parts = self.conv(self.qkv(x)).split([H * dk] * self.n_k + [H * dv], -1)
        q, k = l2norm(split(parts[0], H)), l2norm(split(parts[1], H))
        kr = l2norm(split(parts[2], H)) if self.separate else k
        v = split(parts[-1], H)
        q = q * dk ** -0.5
        beta = split(torch.sigmoid(self.wb(x)) * self.beta_range, H)            # (B, H, L, 1)
        logw = split(-torch.exp(self.A_log) * F.softplus(self.wa(x) + self.dt_bias), H)
        alpha = torch.exp(logw)
        o, _ = dplr(q, beta * k, v, logw, b=alpha * beta * kr, c=kr, chunk=self.chunk, mode=self.mode)
        return self.out(merge(self.onorm(o)) * F.silu(self.g(x)))


def build(d, cfg):
    return Mixer(d, cfg)
