#!/usr/bin/env bash
# Reproduce BASELINES.md: fetch the pinned official code, check the CPU ports against it,
# then train and score the 6 baselines on MQAR (zoology) and the RULER-style NIAH tasks.
# Runs stay under 2 GB each; JOBS sets how many run at once (default 2).
set -euo pipefail
cd "$(dirname "$0")"
JOBS="${JOBS:-2}"

bash data/download.sh
pip3 install -q einops opt_einsum transformers
python3 tests/check_official.py

mkdir -p data/runs/mqar data/runs/ruler
# MQAR: lr 1e-2 for all, plus 3e-3 for Based and GLA (BASELINES.md lists the best per method)
{
  for m in attention based gated_deltanet gla rwkv7 mamba2; do echo "$m 1e-2 run_$m"; done
  echo "based 3e-3 run_based_lr3e-3"; echo "gla 3e-3 run_gla_lr3e-3"
} | xargs -P "$JOBS" -L1 sh -c 'python3 -I benchmark/mqar_zoology.py method=$0 lr=$1 steps=1500 train=64:4,128:8 threads=3 | grep "^{" > data/runs/mqar/$2.json'

# RULER-style NIAH: lr 1e-2 for attention, 3e-3 for the rest
{
  echo "attention 1e-2"
  for m in based gated_deltanet gla rwkv7 mamba2; do echo "$m 3e-3"; done
} | xargs -P "$JOBS" -L1 sh -c 'python3 -I benchmark/ruler_niah.py method=$0 lr=$1 train_len=128 test_len=128,256 batch=64 steps=3000 | grep "^{" > data/runs/ruler/run_$0.json'

python3 benchmark/official_table.py
python3 baselines/make_manifest.py
