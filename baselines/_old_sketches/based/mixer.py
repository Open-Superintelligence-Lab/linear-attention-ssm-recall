"""Based (arXiv 2402.18668), laptop size.

Keeps: linear attention with the 2nd-order Taylor feature map on small projected q, k
(feature dim 1 + f + f^2 for projection size f = feat_dim), normalized read
(phi(q).S / phi(q).z), no decay, plus a small sliding-window softmax attention.
Changes: Based stacks separate linear-attention, sliding-window and gated-conv layers; the
laptop version puts the linear-attention branch and a sliding-window branch in the same
mixer (outputs added) and replaces the BaseConv layers with a short conv on q, k, v.
Published Based uses f = 16 (273 features) and window 64; the laptop default is f = 4
(21 features) and window 32 so its state is near the other families' (see state_floats).
Knobs: feature (identity, elu1, relu, taylor, hedgehog), feat_dim, window (0 = no SWA
branch), normalize (1 = divide by phi(q).sum_j phi(k_j); positive feature maps only).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.common import RMSNorm, ShortConv, split, merge, feature_map, feature_dim, rope, sliding_mask
from env.dplr import dplr


class Mixer(nn.Module):
    def __init__(self, d, cfg):
        super().__init__()
        H, dv, f = int(cfg["n_heads"]), int(cfg["head_v"]), int(cfg["feat_dim"])
        self.H, self.dv, self.f = H, dv, f
        self.kind = cfg["feature"]
        self.F = feature_dim(self.kind, f)
        self.normalize = int(cfg["normalize"]) and self.kind != "identity"
        self.window = int(cfg["window"])
        self.chunk, self.mode = int(cfg["chunk"]), cfg["dplr_mode"]
        self.qkv = nn.Linear(d, H * (2 * f + dv), bias=False)
        self.conv = ShortConv(H * (2 * f + dv), int(cfg["conv"]), act=False)
        self.hh = nn.Linear(f, f, bias=False) if self.kind == "hedgehog" else None
        if self.window > 0:
            self.swa_qkv = nn.Linear(d, 3 * H * dv, bias=False)
            self.qn, self.kn = RMSNorm(dv), RMSNorm(dv)
        self.out = nn.Linear(H * dv, d, bias=False)

    def state_floats(self, L):
        lin = self.H * self.F * (self.dv + (1 if self.normalize else 0))
        swa = 2 * min(L, self.window) * self.H * self.dv if self.window > 0 else 0
        return lin + swa

    def forward(self, x):
        B, L, _ = x.shape
        H, f, dv = self.H, self.f, self.dv
        q, k, v = self.conv(self.qkv(x)).split([H * f, H * f, H * dv], -1)
        q, k, v = split(q, H), split(k, H), split(v, H)
        q, k = feature_map(q, self.kind, self.hh), feature_map(k, self.kind, self.hh)
        if self.kind == "identity":
            q = q * f ** -0.5
        logw = q.new_zeros(B, H, L, 1)
        if self.normalize:
            o, _ = dplr(q, k, torch.cat([v, torch.ones_like(v[..., :1])], -1), logw, chunk=self.chunk, mode=self.mode)
            o = o[..., :dv] / (o[..., dv:] + 1e-6)
        else:
            o, _ = dplr(q, k, v, logw, chunk=self.chunk, mode=self.mode)
        y = merge(o)
        if self.window > 0:
            q2, k2, v2 = self.swa_qkv(x).split(H * dv, -1)
            q2, k2, v2 = split(q2, H), split(k2, H), split(v2, H)
            q2, k2 = rope(self.qn(q2)), rope(self.kn(k2))
            if self.window < L:
                o2 = F.scaled_dot_product_attention(q2, k2, v2, attn_mask=sliding_mask(L, self.window, x.device))
            else:
                o2 = F.scaled_dot_product_attention(q2, k2, v2, is_causal=True)
            y = y + merge(o2)
        return self.out(y)


def build(d, cfg):
    return Mixer(d, cfg)
