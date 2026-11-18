"""Write (default) or check (--check) the list of benchmark eval items.

python3 benchmark/eval_items.py           # writes benchmark/eval_items.json
python3 benchmark/eval_items.py --check   # regenerates and compares token hashes

Each item ID is L<length>-<index>; the file stores per-length seed, pairs and a SHA-256 of
the generated tokens and targets so every run scores identical sequences.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from benchmark.bench import bench_set, token_hash, BENCH_LENGTHS, BENCH_SEED, N_PER_LENGTH, BENCH_CFG  # noqa: E402

PATH = os.path.join(HERE, "eval_items.json")


def build():
    out = dict(benchmark="MQAR-LR", config=BENCH_CFG, n_per_length=N_PER_LENGTH, lengths={})
    for L in BENCH_LENGTHS:
        x, y, _ = bench_set(L)
        out["lengths"][str(L)] = dict(seed=BENCH_SEED + L, pairs=int(L * BENCH_CFG["pairs_frac"]),
                                      queries=int((y != -100).sum()), sha256=token_hash(x, y),
                                      ids=[f"L{L:04d}-{i:03d}" for i in range(len(x))])
    return out


if __name__ == "__main__":
    new = build()
    if "--check" in sys.argv:
        old = json.load(open(PATH))
        bad = [L for L in new["lengths"] if new["lengths"][L]["sha256"] != old["lengths"][L]["sha256"]]
        print("eval items match" if not bad else f"MISMATCH at lengths {bad}")
        sys.exit(1 if bad else 0)
    json.dump(new, open(PATH, "w"), indent=1)
    print(sum(len(v["ids"]) for v in new["lengths"].values()), "items written")
