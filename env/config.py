"""Run config: every knob, its default, and the per-method defaults.

A run is `python3 -m benchmark.run key=value ...`. Values of -1 or "method" mean "use the
method's own default" from METHOD_DEFAULTS.
"""

DEFAULTS = dict(
    # model
    method="gated_deltanet",      # mamba2 | gla | based | gated_deltanet | rwkv7 | attention | swa
    d_model=64, n_layers=2, n_heads=2, head_k=16, head_v=32, mlp=1,
    hybrid="",                    # e.g. "attn@1" or "swa@0,2": these layers use that mixer instead
    conv=-1,                      # short causal conv kernel, 0 = none
    qk_norm=-1,                   # L2-normalize q and k
    decay="method",               # scalar | vector
    decay_init="default",         # short | default | long
    a_range="1,16",               # mamba2 / gated_deltanet: A init range (published 1-16)
    beta_range=1.0,               # Gated DeltaNet: beta in (0, beta_range)
    removal_key="method",         # gated_deltanet: tied | separate; rwkv7: rwkv7 | tied | separate
    token_shift=1,                # rwkv7
    feature="taylor", feat_dim=4, normalize=1,   # based
    window=32,                    # swa and the based SWA branch
    chunk=-1, dplr_mode="chunked",   # chunk size: speed only, same math; -1 = method default
    # task
    task="mqar", eval_task="",    # eval_task "" = same as task
    pairs_frac=0.125, pairs=0,    # stored pairs D = L * pairs_frac, or a fixed D if pairs > 0
    filler="zero", layout="spread", overwrite=0.0, query_order="random",
    n_needles=1, needle_filler="noise", dense_train=0,   # dense_train=1: fill the query region with queries (training only)
    train_lengths="128,256,512,1024", eval_lengths="", curriculum="mixed",
    # training
    steps=1500, batch_tokens=8192, opt="muon", lr_mult=1.0, lr_muon=0.02, lr_adam=3e-3,
    warmup=100, clip=1.0, seed=0, threads=1, eval_every=0,
)

METHOD_DEFAULTS = dict(
    mamba2=dict(conv=4, qk_norm=0, decay="scalar", removal_key="tied"),
    gla=dict(conv=0, qk_norm=0, decay="vector", removal_key="tied"),
    based=dict(conv=4, qk_norm=0, decay="scalar", removal_key="tied"),
    gated_deltanet=dict(conv=4, qk_norm=1, decay="scalar", removal_key="tied"),
    rwkv7=dict(conv=0, qk_norm=0, decay="vector", removal_key="rwkv7", chunk=16),
    attention=dict(conv=0, qk_norm=1, decay="scalar", removal_key="tied"),
    swa=dict(conv=0, qk_norm=1, decay="scalar", removal_key="tied"),
)

RECURRENT = ("mamba2", "gla", "based", "gated_deltanet", "rwkv7")
ALL_METHODS = RECURRENT + ("attention", "swa")


def _cast(v, ref):
    if isinstance(ref, bool):
        return v in ("1", "true", "True")
    if isinstance(ref, int):
        return int(float(v))
    if isinstance(ref, float):
        return float(v)
    return v


def resolve(overrides):
    """Merge defaults, method defaults and overrides (strings from the command line or values from grid rows)."""
    cfg = dict(DEFAULTS)
    for k, v in overrides.items():
        if k in DEFAULTS:
            cfg[k] = _cast(v, DEFAULTS[k]) if isinstance(v, str) else v
        else:
            cfg[k] = v                 # bookkeeping keys (id, question, context) pass through
    md = METHOD_DEFAULTS[cfg["method"]]
    for k in ("conv", "qk_norm", "chunk"):
        if cfg[k] == -1:
            cfg[k] = md.get(k, 32)
    for k in ("decay", "removal_key"):
        if cfg[k] == "method":
            cfg[k] = md[k]
    if not cfg["eval_task"]:
        cfg["eval_task"] = cfg["task"]
    return cfg


def lengths(s):
    return [int(x) for x in str(s).split(",") if x]
