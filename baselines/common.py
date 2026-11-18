"""Shared parts for all mixers: norms, short causal conv, head reshapes, feature maps, RoPE."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class ShortConv(nn.Module):
    """Depthwise causal conv over time. kernel <= 0 means identity. act applies SiLU after."""

    def __init__(self, dim, kernel, act=False):
        super().__init__()
        self.kernel = int(kernel)
        self.act = act
        if self.kernel > 0:
            self.conv = nn.Conv1d(dim, dim, self.kernel, groups=dim, padding=self.kernel - 1, bias=False)

    def forward(self, x):                      # (B, L, dim)
        if self.kernel > 0:                    # shift-and-add: same result as the padded conv,
            K, L = self.kernel, x.shape[1]     # much faster than depthwise conv1d backward on CPU
            w = self.conv.weight[:, 0, :]      # (dim, K)
            xp = F.pad(x, (0, 0, K - 1, 0))
            x = sum(xp[:, i:i + L] * w[:, i] for i in range(K))
        return F.silu(x) if self.act else x


def split(x, H):                               # (B, L, H*d) -> (B, H, L, d)
    B, L, _ = x.shape
    return x.view(B, L, H, -1).transpose(1, 2)


def merge(x):                                  # (B, H, L, d) -> (B, L, H*d)
    B, H, L, d = x.shape
    return x.transpose(1, 2).reshape(B, L, H * d)


def l2norm(x, eps=1e-6):
    return F.normalize(x, dim=-1, eps=eps)


def feature_dim(kind, d):
    if kind == "taylor":
        return 1 + d + d * d
    if kind == "hedgehog":
        return 2 * d
    return d


def feature_map(x, kind, proj=None):
    """Feature maps for linear attention. proj is the learned map for hedgehog."""
    if kind == "identity":
        return x
    if kind == "elu1":
        return F.elu(x) + 1
    if kind == "relu":
        return F.relu(x)
    if kind == "taylor":                       # Based: phi(q).phi(k) = 1 + s + s^2/2, s = q.k/sqrt(d)
        d = x.shape[-1]
        x = x / d ** 0.25
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)).flatten(-2) / math.sqrt(2)
        return torch.cat([torch.ones_like(x[..., :1]), x, x2], -1)
    if kind == "hedgehog":                     # Hedgehog 2402.04347: softmax over [Wx, -Wx]
        z = proj(x)
        return torch.softmax(torch.cat([z, -z], -1), -1)
    raise ValueError(f"unknown feature map {kind}")


def rope(x, base=10000.0):
    """Rotary position embedding on (B, H, L, d), d even."""
    B, H, L, d = x.shape
    half = d // 2
    freqs = base ** (-torch.arange(half, device=x.device, dtype=x.dtype) / half)
    ang = torch.arange(L, device=x.device, dtype=x.dtype)[:, None] * freqs[None]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:2 * half]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)
    return torch.cat([out, x[..., 2 * half:]], -1) if d % 2 else out


def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))


def dt_init(n, level):
    """Mamba-style time-step bias init, log-uniform in a range set by decay_init."""
    lo, hi = {"short": (1e-2, 1.0), "default": (1e-3, 1e-1), "long": (1e-4, 1e-2)}[level]
    dt = torch.exp(torch.rand(n) * (math.log(hi) - math.log(lo)) + math.log(lo))
    return inv_softplus(dt)


def a_range(cfg):
    """Init range of A (decay rate per unit dt) for Mamba-2 and Gated DeltaNet. Published: 1-16."""
    lo, hi = (float(v) for v in str(cfg.get("a_range", "1,16")).split(","))
    return lo, hi


def sliding_mask(L, window, device):
    i = torch.arange(L, device=device)
    m = i[:, None] >= i[None, :]
    if window and window > 0:
        m = m & (i[:, None] - i[None, :] < window)
    return m
