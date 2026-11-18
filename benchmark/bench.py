"""The fixed benchmark MQAR-LR and the scoring code.

MQAR-LR: multi-query associative recall at lengths 128, 256, 512, 1024 with D = L/8 stored
pairs (16, 32, 64, 128), constant filler, queries at random slots after the pairs, every
stored key queried once. 256 sequences per length, generated from seed 10000 + L.
Score per length = accuracy of the predicted value over all queries; MQAR-LR score = mean
of the four per-length accuracies. Every run scores exactly these 1,024 sequences
(IDs and token hashes in eval_items.json, checked by eval_items.py --check).

`evaluate()` also scores task variants (noise filler, overwrite, needle, other lengths) on
seeds 20000 + L, 128 sequences per length, for runs that change the task.
"""
import hashlib
import numpy as np
import torch

from env import mqar

BENCH_LENGTHS = (128, 256, 512, 1024)
BENCH_PAIRS_FRAC = 0.125
N_PER_LENGTH = 256
BENCH_SEED = 10000
TASK_SEED = 20000
N_TASK = 128
BENCH_CFG = dict(pairs_frac=BENCH_PAIRS_FRAC, pairs=0, filler="zero", layout="spread",
                 overwrite=0.0, query_order="random")
DIST_BUCKETS = (0, 64, 256, 1024, 1 << 30)


def bench_set(L):
    rng = np.random.default_rng(BENCH_SEED + L)
    return mqar.make("mqar", rng, N_PER_LENGTH, L, BENCH_CFG)


def token_hash(x, y):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.int64).tobytes())
    return h.hexdigest()


@torch.no_grad()
def score(model, x, y, meta, batch_tokens=16384):
    """Accuracy at scored positions, plus accuracy by write-to-query distance and on re-assigned keys."""
    model.eval()
    L = x.shape[1]
    bs = max(1, batch_tokens // L)
    correct = np.zeros(y.shape, bool)
    for i in range(0, len(x), bs):
        pred = model(torch.from_numpy(x[i:i + bs])).argmax(-1).numpy()
        correct[i:i + bs] = pred == y[i:i + bs]
    model.train()
    m = y != mqar.IGNORE
    out = dict(acc=float(correct[m].mean()), n_queries=int(m.sum()))
    d = meta["dist"]
    for lo, hi in zip(DIST_BUCKETS[:-1], DIST_BUCKETS[1:]):
        mm = m & (d >= lo) & (d < hi)
        if mm.any():
            out[f"acc_d{lo}"] = float(correct[mm].mean())
    u = m & meta["updated"]
    if u.any():
        out["acc_updated"] = float(correct[u].mean())
        out["acc_kept"] = float(correct[m & ~meta["updated"]].mean())
    return out


def evaluate(model, cfg):
    """Fixed benchmark always; task-variant eval when the run's eval task or lengths differ."""
    res = {}
    accs = []
    for L in BENCH_LENGTHS:
        x, y, meta = bench_set(L)
        r = score(model, x, y, meta)
        res[f"bench_L{L}"] = round(r["acc"], 4)
        accs.append(r["acc"])
    res["bench_score"] = round(float(np.mean(accs)), 4)
    task_keys = ("pairs_frac", "pairs", "filler", "layout", "overwrite", "query_order")
    same_task = cfg["eval_task"] == "mqar" and all(cfg[k] == BENCH_CFG[k] for k in task_keys)
    eval_L = [int(v) for v in str(cfg["eval_lengths"]).split(",") if v]
    if same_task and not eval_L:
        return res
    for L in eval_L or BENCH_LENGTHS:
        rng = np.random.default_rng(TASK_SEED + L)
        try:
            x, y, meta = mqar.make(cfg["eval_task"], rng, N_TASK, L, dict(cfg, dense_train=0))
        except ValueError:
            continue                    # setting does not fit this length
        r = score(model, x, y, meta)
        for k, v in r.items():
            if k != "n_queries":
                res[f"task_L{L}_{k}"] = round(v, 4)
    ta = [v for k, v in res.items() if k.startswith("task_L") and k.endswith("_acc")]
    if ta:
        res["task_score"] = round(float(np.mean(ta)), 4)
    return res
