"""MQAR with the official zoology generator and model, at laptop size. Prints one JSON line.

python3 -I benchmark/mqar_zoology.py method=based
python3 -I benchmark/mqar_zoology.py method=gated_deltanet d_model=64 lr=3e-3 steps=1500

Data: zoology.data.multiquery_ar.multiquery_ar (HazyResearch/zoology, pinned in data/download.sh),
vocab 8,192, power_a 0.01, random non-queries: the generator of the Zoology and Based papers.
Model: zoology.model.LanguageModel as built in zoology/experiments/models_repo.py
(baselines/official/zoology_lm.py). Training follows zoology/train.py: cross-entropy on the
query positions, its optimizer with weight decay 0.1, cosine decay to 0. Score: zoology's
compute_metrics accuracy (mean over examples of the fraction of queries answered).

Laptop changes against original_mqar_configs.py (written in the output JSON):
  - training examples / steps: `steps` batches of `batch` (published: 32 epochs over 180k examples)
  - train mix: (L 64, 4 pairs), (128, 8), (256, 16) by default; published adds (256, 32) and (256, 64)
  - test: (64, 4/8/16), (128, 32), (256, 64); published also tests (512, 128) and (1024, 256)
  - logits only at query positions (the same loss; saves memory with vocab 8,192)
"""
import json
import math
import os
import resource
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from baselines.official import fla_cpu, zoology_lm  # noqa: E402

DEFAULTS = dict(method="based", d_model=64, lr=3e-3, wd=0.1, steps=1500, batch=64, seed=123,
                train="64:4,128:8,256:16", test="64:4,64:8,64:16,128:32,256:64",
                n_train=20000, n_test=500, vocab=8192, scan="engine", threads=4, mixer_kw="")


def parse(argv):
    cfg = dict(DEFAULTS)
    for a in argv:
        k, v = a.split("=", 1)
        cfg[k] = type(DEFAULTS[k])(v) if k in DEFAULTS and not isinstance(DEFAULTS[k], str) else v
    return cfg


def segs(spec):
    return [tuple(int(t) for t in s.split(":")) for s in spec.split(",")]


def make(gen, vocab, L, kv, n, seed, chunk=1000):
    """Generate in chunks of 1,000 (own seed each): the generator's peak memory grows ~65 MB
    per 1,000 examples at L 256, and the runs must stay under 2 GB."""
    xs, ys = [], []
    for c in range(0, n, chunk):
        d = gen(vocab_size=vocab, num_examples=min(chunk, n - c), input_seq_len=L,
                seed=seed * 1000 + c // chunk, num_kv_pairs=kv)
        xs.append(d.inputs); ys.append(d.labels)
    return torch.cat(xs), torch.cat(ys)


def loss_and_preds(net, x, y):
    h = net.backbone(x)
    m = y != -100
    logits = net.lm_head(h[m])
    return torch.nn.functional.cross_entropy(logits, y[m]), logits.argmax(-1), m


def state_size(net, lmax):
    try:
        return int(sum(b.sequence_mixer.state_size(sequence_length=lmax) for b in net.backbone.layers))
    except Exception:
        return None


def main():
    cfg = parse(sys.argv[1:])
    torch.set_num_threads(cfg["threads"])
    torch.manual_seed(cfg["seed"])
    fla_cpu.set_scan(cfg["scan"], 32)
    zoology_lm.zoology()
    from zoology.data.multiquery_ar import multiquery_ar
    t0 = time.time()
    tr = [make(multiquery_ar, cfg["vocab"], L, kv, cfg["n_train"], cfg["seed"] + i)
          for i, (L, kv) in enumerate(segs(cfg["train"]))]
    te = [(L, kv, *make(multiquery_ar, cfg["vocab"], L, kv, cfg["n_test"], 10_000 + cfg["seed"] + i))
          for i, (L, kv) in enumerate(segs(cfg["test"]))]
    lmax = max(L for L, _ in segs(cfg["train"]) + segs(cfg["test"]))
    kw = json.loads(cfg["mixer_kw"]) if cfg["mixer_kw"] else {}
    net = zoology_lm.build(cfg["method"], d_model=cfg["d_model"], l_max=lmax, vocab=cfg["vocab"], **kw)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, cfg["steps"]) / cfg["steps"])))
    rng = np.random.default_rng(cfg["seed"])
    curve, t1 = [], time.time()
    net.train()
    for step in range(cfg["steps"]):
        x, y = tr[step % len(tr)]
        idx = torch.from_numpy(rng.integers(0, len(x), cfg["batch"]))
        loss, _, _ = loss_and_preds(net, x[idx], y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if not math.isfinite(loss.item()):
            break
        if step % 100 == 0 or step == cfg["steps"] - 1:
            curve.append([step, round(loss.item(), 4)])
    train_secs = time.time() - t1
    net.eval()
    acc = {}
    with torch.no_grad():
        for L, kv, x, y in te:
            per = []
            for i in range(0, len(x), 50):
                _, pred, m = loss_and_preds(net, x[i:i + 50], y[i:i + 50])
                hit = torch.zeros_like(y[i:i + 50], dtype=torch.float)
                hit[m] = (pred == y[i:i + 50][m]).float()
                per.append(hit.sum(1) / m.sum(1))
            acc[f"L{L}_kv{kv}"] = round(float(torch.cat(per).mean()), 4)
    out = dict(benchmark="mqar_zoology", config=cfg, params=sum(p.numel() for p in net.parameters()),
               state_size=state_size(net, lmax),
               acc=acc, score=round(float(np.mean(list(acc.values()))), 4), final_loss=curve[-1][1] if curve else None,
               curve=curve, train_secs=round(train_secs, 1), secs=round(time.time() - t0, 1),
               peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
