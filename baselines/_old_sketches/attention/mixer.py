"""Reference mixers: full causal softmax attention and sliding-window attention (SWA).

Modern defaults: RoPE, QK-norm (RMSNorm per head on q and k), no biases.
Head size is head_v. window = 0 means full attention.
"""
import torch.nn as nn
import torch.nn.functional as F

from baselines.common import RMSNorm, split, merge, rope, sliding_mask


class Mixer(nn.Module):
    def __init__(self, d, cfg, window=0):
        super().__init__()
        self.H, self.dh = int(cfg["n_heads"]), int(cfg["head_v"])
        self.window = int(window)
        self.qkv = nn.Linear(d, 3 * self.H * self.dh, bias=False)
        self.qn, self.kn = RMSNorm(self.dh), RMSNorm(self.dh)
        self.out = nn.Linear(self.H * self.dh, d, bias=False)

    def state_floats(self, L):
        n = min(L, self.window) if self.window > 0 else L
        return 2 * n * self.H * self.dh

    def forward(self, x):
        B, L, _ = x.shape
        q, k, v = self.qkv(x).split(self.H * self.dh, -1)
        q, k, v = split(q, self.H), split(k, self.H), split(v, self.H)
        q, k = rope(self.qn(q)), rope(self.kn(k))
        if self.window > 0 and self.window < L:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=sliding_mask(L, self.window, x.device))
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(merge(o))


def build(d, cfg):
    return Mixer(d, cfg, window=0)


def build_swa(d, cfg):
    return Mixer(d, cfg, window=int(cfg["window"]))
