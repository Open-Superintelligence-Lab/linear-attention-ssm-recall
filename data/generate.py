"""Optional: write the benchmark sets and sample training batches to data/ as .npz files.

python3 data/generate.py            # data/bench_L0128.npz ... data/bench_L1024.npz
python3 data/generate.py --train 4  # also 4 sample training batches per length

Training never reads these files: env/train.py generates every batch on the fly from the
run seed, and benchmark/bench.py regenerates the fixed sets from their seeds. This script is
for inspecting sequences or loading them into other code. data/ is gitignored except this file.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402

from benchmark.bench import bench_set, BENCH_LENGTHS  # noqa: E402
from env import mqar  # noqa: E402
from env.config import resolve  # noqa: E402


def main(argv):
    n_train = int(argv[argv.index("--train") + 1]) if "--train" in argv else 0
    for L in BENCH_LENGTHS:
        x, y, meta = bench_set(L)
        np.savez_compressed(os.path.join(HERE, f"bench_L{L:04d}.npz"), x=x, y=y, **meta)
        print(f"bench L={L}: {x.shape[0]} sequences, {(y != mqar.IGNORE).sum()} queries")
    if n_train:
        cfg = resolve({})
        rng = np.random.default_rng(0)
        for L in BENCH_LENGTHS:
            n = cfg["batch_tokens"] // L
            for b in range(n_train):
                x, y, meta = mqar.make("mqar", rng, n, L, cfg)
                np.savez_compressed(os.path.join(HERE, f"train_L{L:04d}_b{b}.npz"), x=x, y=y, **meta)
        print(f"{n_train} training batches per length written")


if __name__ == "__main__":
    main(sys.argv[1:])
