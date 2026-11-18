"""Write baselines/manifest.json from the pinned sources, the match checks and the benchmark runs.

python3 baselines/make_manifest.py
"""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FLA = ("https://github.com/fla-org/flash-linear-attention", "cc4648a3b35c27c6bc6b3cfb0079a14d57bcbc5b", "MIT")
MAMBA = ("https://github.com/state-spaces/mamba", "e9594ce1c732d97440f0332fdc43170a2294dbfa", "Apache-2.0")
ZOO = ("https://github.com/HazyResearch/zoology", "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb", "Apache-2.0")
BASED = ("https://github.com/HazyResearch/based", "32c1bd4e09c8f7f718b0ed3f2e0a333908297879", "Apache-2.0")

MODEL = ("Inside zoology's MQAR model (zoology/experiments/models_repo.py): 2 TransformerBlocks, "
         "LayerNorm pre-norm, no MLP, layer 0 BaseConv(kernel 3), layer 1 the method; d_model 64, vocab 8,192.")
CPU = ("Runs on CPU: Triton kernels replaced by the repo's own torch reference (scan=naive) or by "
       "env/dplr.py (scan=engine, used for the runs); the module code itself is unchanged.")

# Published numbers that exist for a setting close to ours. Published MQAR results for the
# Zoology / Based / GLA papers are figures only; the closest printed table is RWKV-7's.
PUBLISHED = {
    "rwkv7": {"mqar": "RWKV-7 paper (2503.14456) Table 6, d_model 64: >99% at (64,4), (128,8), (256,16); "
                      "98.43 at (512,64); best of 3 lr"},
    "gated_deltanet": {"ruler": "Gated DeltaNet paper (2412.06464) Table 2, 1.3B pretrained, S-NIAH-1/2/3 at 1K: "
                                "98.4 / 100.0 / 86.6"},
    "mamba2": {"ruler": "Gated DeltaNet paper Table 2, Mamba2 1.3B, S-NIAH-1/2/3 at 1K: 99.2 / 99.4 / 64.4"},
    "attention": {"mqar": "Zoology (2312.04927) and Based (2402.18668) figures: attention solves MQAR at every "
                          "d_model tested (figure only)"},
    "based": {"mqar": "Based paper figure: Based is the strongest sub-quadratic model on MQAR at equal state "
                      "(figure only)"},
    "gla": {"mqar": "GLA paper (2312.06635) MQAR figure: GLA below attention, above Mamba-1 (figure only)"},
}

# Published order checked on our MQAR table: the papers put attention first, and Gated DeltaNet
# above Mamba-2 (Gated DeltaNet paper, in-context retrieval and S-NIAH).
ORDER_CHECKS = {
    "attention": lambda s: s["attention"] >= max(s.values()) - 1e-9,
    "gated_deltanet": lambda s: s["gated_deltanet"] > s["mamba2"],
    "mamba2": lambda s: s["gated_deltanet"] > s["mamba2"],
    "rwkv7": lambda s: s["rwkv7"] > s["mamba2"],
    "based": lambda s: s["based"] > s["gla"] and s["based"] > s["mamba2"],
    "gla": lambda s: s["attention"] > s["gla"],
}

ENTRIES = {
    "attention": ("zoology.mixers.attention.MHA", ZOO, "zoology's MHA as zoology runs it (2 heads, dropout 0.1). " + MODEL,
                  None),
    "based": ("zoology.mixers.based.Based", ZOO,
              "zoology's own Based layer (Taylor map, feature_dim 16, 1 head, quadratic train view). " + MODEL,
              "based_zoology_vs_hazyresearch_based"),
    "gated_deltanet": ("fla.layers.GatedDeltaNet", FLA,
                       "num_heads 2, head_dim 32, expand_v 2, short conv 4, no output gate (zoology's settings). "
                       + CPU + " " + MODEL, "gated_deltanet"),
    "gla": ("fla.layers.GatedLinearAttention", FLA,
            "num_heads 2, expand_k 0.5, expand_v 1, no short conv. " + CPU + " " + MODEL, "gla"),
    "rwkv7": ("fla.layers.RWKV7Attention", FLA,
              "head_dim 64, low-rank dims 16/64/16/16; built as layer 1 of 2 and run with v_first = v "
              "(zoology's copy with v_first None does the same). " + CPU + " " + MODEL, "rwkv7_layer0"),
    "mamba2": ("mamba_ssm.modules.mamba2.Mamba2", MAMBA,
               "d_state 16, headdim 64, expand 2, chunk 32, use_mem_eff_path False (nn.Conv1d path); Mamba2 in "
               "both layers, no BaseConv, LayerNorm blocks (zoology's Mamba2Block uses RMSNorm). " + CPU, "mamba2"),
}


def best(pattern):
    out = {}
    for f in glob.glob(os.path.join(ROOT, pattern)):
        try:
            o = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        m = o["config"]["method"]
        if m not in out or o["score"] > out[m]["score"]:
            out[m] = o
    return out


def main():
    checks = json.load(open(os.path.join(ROOT, "baselines/match_check_layers.json")))["results"]
    mq, ru = best("data/runs/mqar/run_*.json"), best("data/runs/ruler/run_*.json")
    mq_scores = {m: o["score"] for m, o in mq.items()}
    rows = []
    for m, (src, (repo, commit, lic), changes, key) in ENTRIES.items():
        if key is None:
            mc, md = "not needed", "zoology's own attention layer, used unchanged"
        else:
            c = checks.get(key, {})
            ok = c.get("pass", c.get("ok"))
            mc = "pass" if ok else ("fail" if ok is False else "see detail")
            what = ("zoology Based vs HazyResearch/based LinearAttention (TaylorExp 16), same weights"
                    if key.startswith("based") else
                    "official layer with the CPU scan vs the official torch reference, same weights")
            md = (what + ", B=2 T=100 d=64: "
                  + ", ".join(f"{k} {v:.1e}" for k, v in c.items() if k.endswith("rel_err")))
        score = {}
        if m in mq:
            score["mqar_zoology"] = {"score": mq[m]["score"], "lr": mq[m]["config"]["lr"], "acc": mq[m]["acc"]}
        if m in ru:
            score["ruler_niah_tokens"] = {"score": ru[m]["score"], "lr": ru[m]["config"]["lr"], "by_len": ru[m]["by_len"]}
        holds = None
        if len(mq_scores) == len(ENTRIES):
            holds = bool(ORDER_CHECKS[m](mq_scores))
        rows.append(dict(method=m, source=src, repo=repo, commit=commit, license=lic, changes=changes,
                         match_check=mc, match_detail=md,
                         benchmark="mqar_zoology (laptop size); ruler_niah_tokens (RULER NIAH configs as tokens)",
                         our_score=score, published_score=PUBLISHED.get(m, {}),
                         published_order_holds=holds))
    json.dump(rows, open(os.path.join(ROOT, "baselines/manifest.json"), "w"), indent=1)
    for r in rows:
        print(r["method"], r["match_check"], {k: v["score"] for k, v in r["our_score"].items()}, r["published_order_holds"])


if __name__ == "__main__":
    main()
