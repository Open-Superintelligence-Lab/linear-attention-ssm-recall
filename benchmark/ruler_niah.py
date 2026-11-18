"""RULER needle-in-a-haystack tasks, ported to tokens for from-scratch models. Prints one JSON line.

python3 -I benchmark/ruler_niah.py method=gated_deltanet
python3 -I benchmark/ruler_niah.py method=attention train_len=256 test_len=256,512

Source: NVIDIA/RULER scripts/data/synthetic/niah.py and synthetic.yaml (pinned in data/download.sh).
RULER builds text for a pretrained LLM (Paul Graham essays, an LLM tokenizer, a chat template) and
scores with string_match_all. No pretrained model is allowed here, so the same 8 task configs are
built directly from tokens and a from-scratch model is trained on them (the closest standard
from-scratch version; REDO-BRIEF rule 7):

  RULER text                                     token port
  needle "One of the special magic numbers       [NEEDLE, key, IS, value]
          for {key} is: {value}."
  key word (adjective-noun)                      1 token from 1,000 key tokens
  value number (7 digits)                        1 token from 4,000 value tokens
  uuid (key or value)                            4 tokens from 16 hex tokens
  haystack noise (one repeated sentence)         one fixed 18-token sentence, repeated
  haystack essay (Paul Graham essays)            sentences of 8-24 tokens drawn from a Zipf(1.1)
                                                 over 3,000 filler tokens
  haystack needle (distractor needles)           needles with random keys and values
  needle depths: 40 depths 0..100% (essay);      same rules, at the sentence level
    random sentence index (noise, needle)
  question "What are all the special magic       [QUERY, key, IS] then the answer tokens
    numbers for {key}?"
  string_match_all (fraction of answers found)   fraction of answer values whose tokens are all
                                                 predicted (teacher-forced argmax)

Tasks (synthetic.yaml): single_1 noise/words/numbers; single_2 essay/words/numbers;
single_3 essay/words/uuids; multikey_1 essay, 4 keys; multikey_2 needle haystack, words/numbers;
multikey_3 needle haystack, uuids/uuids; multivalue essay, 4 values per key; multiquery essay,
4 keys asked. Multivalue answers are asked in the order the needles appear (RULER's metric is
order-free; next-token training needs a fixed order). RULER tests 4k-128k tokens with an LLM;
here lengths are laptop sizes (train 256, test 256 and 512 by default).
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
from benchmark.mqar_zoology import loss_and_preds, state_size  # noqa: E402

DEFAULTS = dict(method="based", d_model=64, lr=1e-2, wd=0.1, steps=1000, batch=64, seed=123,
                train_len=256, test_len="256,512", n_train=2500, n_test=100, vocab=8192,
                scan="engine", threads=3, mixer_kw="")

PAD, NEEDLE, IS, QUERY = 0, 1, 2, 3
NOISE = np.arange(10, 28)                 # the fixed noise sentence (18 tokens)
HEX = np.arange(32, 48)                   # uuid alphabet
KEYS = np.arange(100, 1100)               # key words
VALUES = np.arange(1100, 5100)            # 7-digit numbers
FILLER = np.arange(5100, 8100)            # essay words
DEPTHS = np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int)

TASKS = {   # haystack, key type, value type, keys, values per key, queries
    "niah_single_1": ("noise", "words", "numbers", 1, 1, 1),
    "niah_single_2": ("essay", "words", "numbers", 1, 1, 1),
    "niah_single_3": ("essay", "words", "uuids", 1, 1, 1),
    "niah_multikey_1": ("essay", "words", "numbers", 4, 1, 1),
    "niah_multikey_2": ("needle", "words", "numbers", 1, 1, 1),
    "niah_multikey_3": ("needle", "uuids", "uuids", 1, 1, 1),
    "niah_multivalue": ("essay", "words", "numbers", 1, 4, 1),
    "niah_multiquery": ("essay", "words", "numbers", 1, 1, 4),
}


def parse(argv):
    cfg = dict(DEFAULTS)
    for a in argv:
        k, v = a.split("=", 1)
        cfg[k] = type(DEFAULTS[k])(v) if k in DEFAULTS and not isinstance(DEFAULTS[k], str) else v
    return cfg


class Gen:
    def __init__(self, seed):
        self.r = np.random.default_rng(seed)
        z = 1.0 / np.arange(1, len(FILLER) + 1) ** 1.1
        self.zipf = z / z.sum()

    def item(self, kind):
        if kind == "words":
            return [int(self.r.choice(KEYS))]
        if kind == "numbers":
            return [int(self.r.choice(VALUES))]
        return [int(t) for t in self.r.choice(HEX, 4)]   # uuids

    def sentence(self, hay, kt, vt, avoid=()):
        if hay == "noise":
            return [int(t) for t in NOISE]
        if hay == "essay":
            return [int(t) for t in self.r.choice(FILLER, int(self.r.integers(8, 25)), p=self.zipf)]
        k = self.item(kt)
        while k in avoid:                                         # a distractor never reuses a real key
            k = self.item(kt)
        return [NEEDLE, *k, IS, *self.item(vt)]                   # distractor needle

    def example(self, task, L):
        hay, kt, vt, nk, nv, nq = TASKS[task]
        nk = max(nk, nq)                                          # niah.py line 77
        keys, needles = [], []
        while len(keys) < nk:                                     # distinct keys
            k = self.item(kt)
            if k not in keys:
                keys.append(k)
        vals = {}
        for i, k in enumerate(keys):
            vals[i] = []
            for _ in range(nv):
                v = self.item(vt)
                vals[i].append(v)
                needles.append((i, [NEEDLE, *k, IS, *v]))
        self.r.shuffle(needles)
        ask = list(self.r.choice(nk, nq, replace=False))
        qlen = sum(2 + len(keys[i]) + nv * len(vals[i][0]) for i in ask)
        budget = L - qlen - sum(len(n) for _, n in needles)
        sents = []
        while True:                                               # fill the haystack up to L
            s = self.sentence(hay, kt, vt, keys)
            if sum(map(len, sents)) + len(s) > budget:
                break
            sents.append(s)
        if hay == "essay":
            pos = sorted(int(len(sents) * d / 100) for d in self.r.choice(DEPTHS, len(needles), replace=False))
        else:
            pos = sorted(self.r.choice(len(sents) + 1, len(needles), replace=False))
        x, order, j = [], [], 0
        for p in range(len(sents) + 1):
            while j < len(needles) and pos[j] == p:
                order.append(needles[j]); x += needles[j][1]; j += 1
            if p < len(sents):
                x += sents[p]
        y = [-100] * len(x)
        for i in ask:                                             # question + answers
            q = [QUERY, *keys[i], IS]
            ans = [v for (ki, n) in order if ki == i for v in [n[2 + len(keys[i]):]]]
            x += q; y += [-100] * len(q)
            for v in ans:
                x += v; y += v
        x = [PAD] * (L - len(x)) + x                              # left pad to L
        y = [-100] * (L - len(y)) + y
        # next-token targets: position t predicts token t+1
        inp = torch.tensor(x[:-1] + [PAD])
        lab = torch.tensor(y[1:] + [-100])
        return inp, lab, len(ans[0]) if ans else 1

    def batch(self, task, L, n):
        xs, ys, w = zip(*(self.example(task, L) for _ in range(n)))
        return torch.stack(xs), torch.stack(ys), w[0]


def main():
    cfg = parse(sys.argv[1:])
    torch.set_num_threads(cfg["threads"])
    torch.manual_seed(cfg["seed"])
    fla_cpu.set_scan(cfg["scan"], 32)
    zoology_lm.zoology()
    t0 = time.time()
    g = Gen(cfg["seed"])
    tr = [g.batch(t, cfg["train_len"], cfg["n_train"])[:2] for t in TASKS]
    gt = Gen(10_000 + cfg["seed"])
    tlens = [int(s) for s in cfg["test_len"].split(",")]
    te = [(t, L, *gt.batch(t, L, cfg["n_test"])) for L in tlens for t in TASKS]
    lmax = max(tlens + [cfg["train_len"]])
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
        for task, L, x, y, w in te:
            per = []
            for i in range(0, len(x), 50):
                xb, yb = x[i:i + 50], y[i:i + 50]
                _, pred, m = loss_and_preds(net, xb, yb)
                hit = torch.zeros_like(yb, dtype=torch.float)
                hit[m] = (pred == yb[m]).float()
                for b in range(len(xb)):            # a value counts when all its tokens are right
                    h = hit[b][m[b]].view(-1, w)
                    per.append(float(h.min(1).values.mean()))
            acc[f"{task}@{L}"] = round(float(np.mean(per)), 4)
    by_len = {L: round(float(np.mean([v for k, v in acc.items() if k.endswith(f"@{L}")])), 4) for L in tlens}
    out = dict(benchmark="ruler_niah_tokens", config=cfg, params=sum(p.numel() for p in net.parameters()),
               state_size=state_size(net, lmax), acc=acc, by_len=by_len,
               score=round(float(np.mean(list(acc.values()))), 4),
               final_loss=curve[-1][1] if curve else None, curve=curve,
               train_secs=round(train_secs, 1), secs=round(time.time() - t0, 1),
               peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
