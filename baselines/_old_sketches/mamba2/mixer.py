"""Mamba-2 (SSD, arXiv 2405.21060), laptop size.

Keeps: input-dependent time step dt, scalar decay per head exp(dt * A) with A = -exp(A_log),
B and C shared across heads (one group), short causal conv on (x, B, C) + SiLU, D skip,
gated RMSNorm (norm(y * silu(z))), state N x P per head (N = head_k, P = head_v).
Drops: the fused kernel, expand factor (inner width is n_heads * head_v), dt clamping.
Knobs: decay=vector gives a per-state-channel dt (dt scales B per channel instead of x).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common import RMSNorm, ShortConv, split, merge, l2norm, dt_init, a_range
from env.dplr import dplr


class Mixer(nn.Module):
    def __init__(self, d, cfg):
        super().__init__()
        H, N, P = int(cfg["n_heads"]), int(cfg["head_k"]), int(cfg["head_v"])
        self.H, self.N, self.P = H, N, P
        self.vector = cfg["decay"] == "vector"
        self.qk_norm = int(cfg["qk_norm"])
        self.chunk, self.mode = int(cfg["chunk"]), cfg["dplr_mode"]
        self.ndt = H * N if self.vector else H
        self.in_proj = nn.Linear(d, 2 * H * P + 2 * N + self.ndt, bias=False)
        self.conv = ShortConv(H * P + 2 * N, int(cfg["conv"]), act=True)
        self.dt_bias = nn.Parameter(dt_init(self.ndt, cfg["decay_init"]))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.ndt).uniform_(*a_range(cfg))))
        self.D = nn.Parameter(torch.ones(H))
        self.norm = RMSNorm(H * P)
        self.out = nn.Linear(H * P, d, bias=False)

    def state_floats(self, L):
        return self.H * self.N * self.P

    def forward(self, x):
        B, L, _ = x.shape
        H, N, P = self.H, self.N, self.P
        z, xbc, dt = torch.split(self.in_proj(x), [H * P, H * P + 2 * N, self.ndt], -1)
        xs, Bm, Cm = torch.split(self.conv(xbc), [H * P, N, N], -1)
        dt = F.softplus(dt + self.dt_bias)                     # (B, L, ndt)
        logw = dt * (-torch.exp(self.A_log))
        xh = split(xs, H)                                      # (B, H, L, P)
        k = Bm.unsqueeze(1).expand(B, H, L, N)
        q = Cm.unsqueeze(1).expand(B, H, L, N)
        if self.qk_norm:
            q, k = l2norm(q), l2norm(k)
        if self.vector:
            logw = logw.view(B, L, H, N).transpose(1, 2)
            k = k * dt.view(B, L, H, N).transpose(1, 2)
            v = xh
        else:
            logw = logw.view(B, L, H, 1).transpose(1, 2)
            v = xh * dt.view(B, L, H, 1).transpose(1, 2)
        o, _ = dplr(q, k, v, logw, chunk=self.chunk, mode=self.mode)
        o = o + self.D.view(1, H, 1, 1) * xh
        return self.out(self.norm(merge(o) * F.silu(z)))


def build(d, cfg):
    return Mixer(d, cfg)
