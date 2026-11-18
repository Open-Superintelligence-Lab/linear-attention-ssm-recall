# Data

## Downloads (pinned)

`data/download.sh` clones 5 repos into `data/ext/` (gitignored) at fixed commits and checks the
SHA-256 of the 10 files the environment imports (`data/checksums.sha256`). Code that reads them runs
with `python3 -I`.

| Repo | Commit | License | Used for |
|---|---|---|---|
| HazyResearch/zoology | 1ad20d1 | Apache-2.0 | MQAR generator (`zoology/data/multiquery_ar.py`), model (`zoology/model.py`), Based and BaseConv layers |
| NVIDIA/RULER | db43f92 | Apache-2.0 | NIAH task configs (`scripts/data/synthetic/niah.py`, `synthetic.yaml`) |
| fla-org/flash-linear-attention | cc4648a | MIT | Gated DeltaNet, GLA, RWKV-7 layers |
| state-spaces/mamba | e9594ce | Apache-2.0 | Mamba-2 layer and its reference scan |
| HazyResearch/based | 32c1bd4 | Apache-2.0 | Based LinearAttention (match check for zoology's Based) |

No dataset files are downloaded: both benchmarks are generators, run from the pinned code.

## Benchmark 1: MQAR (zoology generator)

`benchmark/mqar_zoology.py` calls `zoology.data.multiquery_ar.multiquery_ar` (vocab 8,192,
power_a 0.01, random non-query tokens), the generator of the Zoology and Based papers. Laptop
settings against zoology's `original_mqar_configs.py`:

| | Published | Here |
|---|---|---|
| train (L, pairs) | (64,4) 100k, (128,8) 20k, (256,16/32/64) 20k each | (64,4), (128,8), 20k each (seeds 123, 124) |
| training | 32 epochs, batch 256 | 1,500 steps, batch 64 (96k examples seen) |
| lr | best of 4 (1e-3 to 3e-2) | 1e-2 |
| test (L, pairs) | (64,4/8/16), (128,32), (256,64), (512,128), (1024,256), 1k each | (64,4/8/16), (128,32), (256,64), 500 each (seed 10,123+) |

Score: zoology's accuracy (mean over examples of the fraction of queries answered), averaged over
the 5 test settings.

## Benchmark 2: RULER-style NIAH (tokens)

RULER (NVIDIA) builds text for a pretrained LLM and needs its tokenizer; no pretrained model is
allowed here (REDO-BRIEF rule 7). `benchmark/ruler_niah.py` builds the same 8 NIAH task configs
from `synthetic.yaml` directly as tokens and trains a from-scratch model on all 8 at once
(mapping table in the file's docstring):

| Task | Haystack | Key | Value | Keys / values / queries |
|---|---|---|---|---|
| niah_single_1 | noise (one repeated sentence) | word | number | 1 / 1 / 1 |
| niah_single_2 | essay (Zipf filler sentences) | word | number | 1 / 1 / 1 |
| niah_single_3 | essay | word | uuid (4 tokens) | 1 / 1 / 1 |
| niah_multikey_1 | essay | word | number | 4 / 1 / 1 |
| niah_multikey_2 | distractor needles | word | number | 1 / 1 / 1 |
| niah_multikey_3 | distractor needles | uuid | uuid | 1 / 1 / 1 |
| niah_multivalue | essay | word | number | 1 / 4 / 1 |
| niah_multiquery | essay | word | number | 4 / 1 / 4 |

Vocab 8,192: 0 pad, 1 needle marker, 2 "is", 3 query marker, 10-27 noise sentence, 32-47 hex,
100-1099 keys, 1100-5099 numbers, 5100-8099 filler words. Score: the fraction of answer values
whose tokens are all predicted (RULER's string_match_all), averaged over tasks and test lengths.

## Extra task: MQAR-LR (the first version of this environment)

Every sequence is generated with NumPy from a seed by `env/mqar.py`.

### Tokens

Vocabulary of 512: `0` filler, `1-192` keys, `193-384` values, `385-511` noise tokens.

### Tasks

**MQAR** (multi-query associative recall, Zoology style). A sequence of length L holds
D key-value pairs `k1 v1 k2 v2 ... kD vD`, then a query region. In the query region each
queried key appears again followed by its value; the model is scored on predicting the value at
the key's position. All other positions are not scored. Knobs:

| Knob | Values | Meaning |
|---|---|---|
| `pairs_frac` / `pairs` | 0.125 default / fixed D | number of stored pairs |
| `filler` | zero, noise | free positions hold token 0 or random noise tokens |
| `layout` | spread, gap | queries at random slots, or packed at the end (maximum distance) |
| `overwrite` | 0-1 | fraction of keys given a new value in an update block after the pairs |
| `query_order` | random, same, reverse | order of queries relative to storage order |

**Needle.** A noise haystack with `n_needles` pairs at random positions; all needles are
queried at the end.

Every generator also returns, per scored position, `dist` (tokens since the key's latest write)
and `updated` (whether the key was re-assigned), used for the distance and overwrite breakdowns.

### Training data

Generated on the fly, one batch per step, from the run seed (`seed`). With the defaults a run
sees 1,500 x 8,192 = 12.3M tokens, lengths 128/256/512/1024 in round robin.

### Test data

Fixed sets, regenerated from seed `10000 + L` at score time: lengths 128, 256, 512, 1024,
256 sequences each (1,024 items), D = L/8. Item IDs and a SHA-256 of the tokens per length are
committed in `benchmark/eval_items.json`. Check your machine generates the same sequences:

```bash
python3 benchmark/eval_items.py --check
```

Task variants (noise, overwrite, needle, other lengths) are scored on seed `20000 + L`, 128
sequences per length. Training seeds are small integers (0, 1, ...), so train and eval never share
a random stream.

## Files on disk

`data/` is gitignored except `data/download.sh`, `data/checksums.sha256` and `data/generate.py`
(writes the MQAR-LR sets as `.npz` for inspection). Run outputs go to `data/runs/`.
