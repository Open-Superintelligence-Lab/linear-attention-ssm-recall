"""The language model around the mixers: token embedding, pre-norm blocks (RMSNorm, mixer,
RMSNorm, SwiGLU MLP), final RMSNorm, untied output head. No biases, no position table
(attention layers use RoPE; recurrent layers need no positions)."""
import importlib
import torch.nn as nn
import torch.nn.functional as F

from baselines.common import RMSNorm
from env.mqar import VOCAB


def build_mixer(name, d, cfg):
    if name in ("attention", "attn"):
        return importlib.import_module("baselines._old_sketches.attention.mixer").build(d, cfg)
    if name == "swa":
        return importlib.import_module("baselines._old_sketches.attention.mixer").build_swa(d, cfg)
    return importlib.import_module(f"baselines._old_sketches.{name}.mixer").build(d, cfg)


def parse_hybrid(s, n_layers):
    """'attn@1' or 'swa@0,2' -> {layer_index: mixer_name}."""
    out = {}
    for part in [p for p in str(s).split(";") if p]:
        name, idx = part.split("@")
        for i in idx.split(","):
            i = int(i)
            if i < 0 or i >= n_layers:
                raise ValueError(f"hybrid layer {i} out of range for {n_layers} layers")
            out[i] = name
    return out


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.up = nn.Linear(d, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        a, b = self.up(x).chunk(2, -1)
        return self.down(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, d, mixer, mlp):
        super().__init__()
        self.n1, self.mixer = RMSNorm(d), mixer
        self.n2 = RMSNorm(d) if mlp else None
        self.mlp = SwiGLU(d, 2 * d) if mlp else None

    def forward(self, x):
        x = x + self.mixer(self.n1(x))
        if self.mlp is not None:
            x = x + self.mlp(self.n2(x))
        return x


class LM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, n = int(cfg["d_model"]), int(cfg["n_layers"])
        hyb = parse_hybrid(cfg["hybrid"], n)
        self.layer_types = [hyb.get(i, cfg["method"]) for i in range(n)]
        self.emb = nn.Embedding(VOCAB, d)
        self.blocks = nn.ModuleList(Block(d, build_mixer(t, d, cfg), int(cfg["mlp"])) for t in self.layer_types)
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        nn.init.normal_(self.emb.weight, std=0.02)

    def state_floats(self, L):
        """Recurrent state (or KV cache for attention layers) per sequence, summed over layers."""
        return sum(b.mixer.state_floats(L) for b in self.blocks)

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))
