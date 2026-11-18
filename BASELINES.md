# Baselines

All baselines come from official code at pinned commits (`data/download.sh`, checksums in
`data/checksums.sha256`). Every entry, with its changes, match check and scores, is in
`baselines/manifest.json` (written by `python3 baselines/make_manifest.py`).

| Method | Source (pinned) | License | Settings | Match check |
|---|---|---|---|---|
| Mamba-2 | state-spaces/mamba `Mamba2` (e9594ce) | Apache-2.0 | d_state 16, headdim 64, expand 2, both layers | CPU scan vs `ssd_chunk_scan_combined_ref`: out 1.7e-7, grads 4.8e-7 |
| Gated DeltaNet | fla `GatedDeltaNet` (cc4648a) | MIT | 2 heads, expand_v 2, short conv 4 | CPU scan vs fla torch reference: out 3.0e-7, grads 1.1e-6 |
| GLA | fla `GatedLinearAttention` (cc4648a) | MIT | 2 heads, expand_k 0.5 | out 1.3e-6, grads 5.5e-7 |
| RWKV-7 | fla `RWKV7Attention` (cc4648a) | MIT | head_dim 64, low-rank 16/64/16/16 | out 3.9e-7, grads 1.0e-6 |
| Based | zoology `Based` (1ad20d1), HazyResearch's MQAR layer | Apache-2.0 | Taylor map, feature_dim 16, 1 head | equals HazyResearch/based `LinearAttention` (32c1bd4) exactly: out and grads 0.0 |
| Attention | zoology `MHA` (1ad20d1) | Apache-2.0 | 2 heads, dropout 0.1 | zoology's own layer, unchanged |

How they run on the Mac (`baselines/official/`):
- Every model is zoology's MQAR model from `zoology/experiments/models_repo.py`: 2 blocks, LayerNorm,
  no MLP, d_model 64, layer 0 zoology `BaseConv` (kernel 3), layer 1 the method. Mamba-2 uses
  Mamba-2 in both layers, as zoology does.
- The fla and mamba modules are imported unchanged. A Triton stub lets them import; their Triton
  calls go to the repos' own torch references (`scan=naive`) or to the chunked engine `env/dplr.py`
  (`scan=engine`, used for the runs). `python3 tests/check_official.py` compares both on a small
  input (B=2, T=100), outputs and every gradient (`baselines/match_check_layers.json`).

## MQAR (zoology generator)

`python3 -I benchmark/mqar_zoology.py method=<m> lr=<lr> steps=1500 train=64:4,128:8 threads=3`.
Train (64,4) and (128,8), 1,500 steps of batch 64; test 500 sequences per setting. Score = mean of
the 5 test columns. Each method shows its best of the lr values run (1e-2 for all; 3e-3 also for
Based and GLA). Run 2026-11-18 on the MacBook CPU, AC power, shared with other runs.

<!-- mqar -->
| Method | lr | Params | L64_kv4 | L64_kv8 | L64_kv16 | L128_kv32 | L256_kv64 | Score | Train s | Peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| attention | 0.01 | 545,984 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 652 | 0.51 |
| based | 0.003 | 539,584 | 0.999 | 1.000 | 0.997 | 0.979 | 0.843 | **0.963** | 327 | 0.88 |
| gated_deltanet | 0.01 | 555,268 | 1.000 | 1.000 | 0.997 | 0.966 | 0.705 | **0.933** | 1506 | 0.64 |
| gla | 0.003 | 547,328 | 0.948 | 0.809 | 0.501 | 0.146 | 0.027 | **0.486** | 2137 | 1.02 |
| rwkv7 | 0.01 | 560,960 | 1.000 | 0.999 | 0.998 | 0.960 | 0.609 | **0.913** | 3184 | 1.43 |
| mamba2 | 0.01 | 580,300 | 0.941 | 0.856 | 0.641 | 0.291 | 0.076 | **0.561** | 4177 | 0.61 |
<!-- /mqar -->

Published, for comparison (the Zoology, Based and GLA papers print MQAR as figures only):

| Source | Setting | Published | Ours |
|---|---|---|---|
| RWKV-7 paper (2503.14456) Table 6 | d_model 64, (64,4), (128,8), (256,16) | >99% each (best of 3 lr) | (64,4) 1.000; (128,8) not tested; (64,16) 0.998 |
| RWKV-7 paper Table 6 | d_model 64, (512,64) | 98.43 | not tested; (256,64) 0.609 |
| Zoology / Based papers (figures) | d_model 64 | attention solves MQAR; Based the best sub-quadratic model | attention 1.000; Based 0.963, second |
| Gated DeltaNet paper (2412.06464) | 1.3B, retrieval | Gated DeltaNet above Mamba-2 | 0.933 vs 0.561: holds |

Published order on MQAR (attention first; Based strongest sub-quadratic at small state; Gated
DeltaNet above Mamba-2; GLA below attention) holds in our table. Laptop differences against the
papers: 96k training examples instead of 32 epochs over 180k, 1-2 lr values instead of 4, and no
(512,128) or (1024,256) test.

**Not run yet:** MQAR at lr 3e-3 for Gated DeltaNet, RWKV-7 and Mamba-2 (each 25-70 min), and the
long published tests (512,128) and (1024,256), which take more than 5 minutes per method.

## RULER-style NIAH (tokens)

`python3 -I benchmark/ruler_niah.py method=<m> lr=<lr> train_len=128 test_len=128,256 batch=64 steps=3000`.
The 8 NIAH configs of RULER as tokens (DATA.md), trained on all 8 at length 128, tested at 128
and 256, 100 sequences per task and length. Task columns are at length 128; `all@L` averages the 8 tasks.

<!-- ruler -->
| Method | lr | Params | multikey_1 | multikey_2 | multikey_3 | multiquery | multivalue | single_1 | single_2 | single_3 | all@128 | all@256 | Score | Train s | Peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| attention | 0.01 | 545,984 | 0.990 | 0.920 | 0.480 | 0.985 | 0.172 | 1.000 | 1.000 | 1.000 | 0.818 | 0.781 | **0.799** | 929 | 0.47 |
<!-- /ruler -->

**Not run yet** (2026-11-18): RULER-style runs for Based, Gated DeltaNet, GLA, RWKV-7
and Mamba-2 (3,000 steps each, 15-70 min on the shared laptop CPU; commands in `reproduce.sh`).

Published (pretrained 1.3B models, real text, Gated DeltaNet paper Table 2, S-NIAH = RULER
niah_single_1/2/3): Gated DeltaNet 98.4 / 100.0 / 86.6 at 1K, Mamba-2 99.2 / 99.4 / 64.4 at 1K;
at 4K Gated DeltaNet 91.4 / 92.2 / 27.6 vs Mamba-2 65.4 / 56.2 / 4.6. Not comparable to the
laptop numbers; the comparable fact is the order (Gated DeltaNet above Mamba-2, uuid values hardest).

## Extra: MQAR-LR with the old sketch layers

The first version of this environment used hand-written layers on one engine (now in
`baselines/_old_sketches/`, not baselines). The MQAR-LR task (`env/mqar.py`, `benchmark/run.py`)
stays as an extra task; its table below is from those sketch layers.

### Old laptop table (MQAR-LR, sketch layers)

Run 2026-11-18 on the MacBook CPU (AC power), `python3 -m benchmark.run method=<m>` at defaults:
d_model 64, 2 layers, 1,500 steps of 8,192 tokens, lengths 128-1024 round robin. Chance is
1/192 = 0.0052.

| Method | State floats (L=1024) | Params | L128 | L256 | L512 | L1024 | MQAR-LR score | Train time |
|---|---|---|---|---|---|---|---|---|
| based | 10,964 | 158,784 | 0.862 | 0.609 | 0.289 | 0.122 | **0.470** | 555 s |
| attention | 262,144 | 147,904 | 0.160 | 0.102 | 0.059 | 0.035 | 0.089 | 327 s |
| gla | 2,048 | 150,976 | 0.074 | 0.047 | 0.029 | 0.019 | 0.042 | 582 s |
| swa | 8,192 | 147,904 | 0.039 | 0.015 | 0.007 | 0.006 | 0.017 | 343 s |
| mamba2 | 2,048 | 144,844 | 0.009 | 0.009 | 0.007 | 0.007 | 0.008 | 352 s |
| gated_deltanet | 2,048 | 149,384 | 0.004 | 0.005 | 0.005 | 0.006 | 0.005 | 555 s |
| rwkv7 | 2,048 | 150,976 | not run | | | | | ~15 min (over the 10 min tool limit) |

Only Based learns the task in 1,500 steps. It is the only method with both a short conv and no
decay, plus a 32-token attention window. Mamba-2 and Gated DeltaNet stay at chance (loss = ln 192)
even with a slower decay init (`a_range=0.05,0.5`: 0.005 and 0.007). Full attention, which has no
short conv, learns only slowly. So 1,500 steps is too short for most methods.
