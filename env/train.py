"""Training loop and optimizers.

opt=muon (default): Muon (Newton-Schulz orthogonalized momentum) on the 2-D weight matrices
inside blocks, Adam on embeddings, head, norms, convs, biases and scalars.
opt=adam: Adam on everything (no weight decay).
Learning rate: linear warmup, then cosine to 10% of the peak. lr_mult scales both peaks.
"""
import math
import time
import numpy as np
import torch
import torch.nn.functional as F

from env import mqar
from env.config import lengths
from env.model import LM


def _ns(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    tr = X.shape[0] > X.shape[1]
    if tr:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tr else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"]
                buf.mul_(g["momentum"]).add_(p.grad)
                u = _ns(p.grad.add(buf, alpha=g["momentum"]))
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                p.add_(u, alpha=-g["lr"] * scale)


def make_optimizers(model, cfg):
    m = float(cfg["lr_mult"])
    if cfg["opt"] == "adam":
        return [torch.optim.Adam(model.parameters(), lr=cfg["lr_adam"] * m, betas=(0.9, 0.95))]
    mat, rest = [], []
    for name, p in model.named_parameters():
        if p.ndim == 2 and name.startswith("blocks.") and min(p.shape) > 1:
            mat.append(p)
        else:
            rest.append(p)
    return [Muon(mat, lr=cfg["lr_muon"] * m), torch.optim.Adam(rest, lr=cfg["lr_adam"] * m, betas=(0.9, 0.95))]


def schedule(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))


def length_for_step(step, cfg):
    Ls = lengths(cfg["train_lengths"])
    if cfg["curriculum"] == "short_to_long":           # equal steps per length, shortest first
        stage = min(len(Ls) - 1, step * len(Ls) // int(cfg["steps"]))
        return sorted(Ls)[stage]
    return Ls[step % len(Ls)]                          # mixed: round robin


def loss_fn(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=mqar.IGNORE)


def train(cfg, log=None):
    torch.manual_seed(int(cfg["seed"]))
    torch.set_num_threads(int(cfg["threads"]))
    rng = np.random.default_rng(int(cfg["seed"]))
    model = LM(cfg)
    opts = make_optimizers(model, cfg)
    peaks = [[g["lr"] for g in o.param_groups] for o in opts]
    total, curve = int(cfg["steps"]), []
    t0 = time.time()
    for step in range(total):
        L = length_for_step(step, cfg)
        n = max(1, int(cfg["batch_tokens"]) // L)
        x, y, _ = mqar.make(cfg["task"], rng, n, L, cfg)
        loss = loss_fn(model, torch.from_numpy(x), torch.from_numpy(y))
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["clip"]))
        f = schedule(step, total, int(cfg["warmup"]))
        for o, pk in zip(opts, peaks):
            for g, lr in zip(o.param_groups, pk):
                g["lr"] = lr * f
            o.step()
        if not math.isfinite(loss.item()):
            return model, dict(diverged=True, step=step, curve=curve, train_secs=time.time() - t0)
        ee = int(cfg["eval_every"])
        if ee and (step + 1) % ee == 0:
            curve.append([step + 1, round(loss.item(), 4)])
            if log:
                log(f"step {step + 1} L {L} loss {loss.item():.4f}")
    return model, dict(diverged=False, final_loss=loss.item(), curve=curve, train_secs=time.time() - t0)
