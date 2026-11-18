"""Match check for the official fla layers on CPU (baselines/official/fla_cpu.py).

For each official layer (Mamba2 from state-spaces/mamba; GatedDeltaNet, GatedLinearAttention,
RWKV7Attention from flash-linear-attention; both pinned): build it once, run the same input through
  (a) scan="naive": the repos' own PyTorch references (fla naive_*, mamba ssd_chunk_scan_combined_ref) + nn.Conv1d, and
  (b) scan="engine": this repo's chunked DPLR engine + shift-add conv,
and compare outputs and parameter gradients (float32). Writes baselines/match_check.json.
Also: Based. zoology.mixers.based.Based (the MQAR Based layer used as the baseline) against
HazyResearch/based LinearAttention (the Based release), same weights, quadratic view.
Run: python3 tests/check_official.py
"""
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from baselines.official import fla_cpu, mamba_cpu  # noqa: E402

torch.manual_seed(0)


def run(layer, x, mode, chunk, **kw):
    fla_cpu.set_scan(mode, chunk)
    layer.zero_grad()
    xx = x.clone().requires_grad_(True)
    out = layer(xx, **kw)
    o = out[0] if isinstance(out, tuple) else out
    (o * torch.linspace(-1, 1, o.numel()).view_as(o)).sum().backward()
    grads = {n: p.grad.clone() for n, p in layer.named_parameters() if p.grad is not None}
    return o.detach(), xx.grad.clone(), grads


def rel(a, b):
    return float((a - b).abs().max() / (b.abs().max() + 1e-12))


def check_based(x):
    """zoology Based vs HazyResearch/based LinearAttention with the same projection weights."""
    from baselines.official import zoology_lm
    zoology_lm.zoology()
    from zoology.mixers.based import Based
    bpath = os.path.join(ROOT, "data", "ext", "based")
    if bpath not in sys.path:
        sys.path.insert(0, bpath)
    import types
    gen = types.ModuleType("based.generation")   # needs an old transformers; only a type hint here
    gen.InferenceParams = object
    sys.modules.setdefault("based.generation", gen)
    from based.models.mixers.linear_attention import LinearAttention, TaylorExp
    d = x.shape[-1]
    z = Based(d_model=d, l_max=128, feature_dim=16, num_key_value_heads=1, num_heads=1,
              feature_name="taylor_exp", train_view="quadratic")
    h = LinearAttention(d_model=d, feature_map=TaylorExp(16), feature_dim=16, num_heads=1,
                        parallel_implementation="quadratic")
    with torch.no_grad():
        for a, b in (("proj_q", "proj_q"), ("proj_k", "proj_k"), ("proj_v", "proj_v"), ("proj_o", "out_proj")):
            getattr(h, b).weight.copy_(getattr(z, a).weight)
    oz, xz, gz = run(z, x, "naive", 32)
    oh, xh, gh = run(h, x, "naive", 32)
    gh = {k.replace("out_proj", "proj_o"): v for k, v in gh.items()}
    gerr = max(rel(gz[k], gh[k]) for k in gh)
    r = {"out_rel_err": rel(oz, oh), "x_grad_rel_err": rel(xz, xh), "param_grad_max_rel_err": gerr,
         "n_param_grads": len(gh), "out_abs_max": float(oh.abs().max())}
    r["pass"] = max(r["out_rel_err"], r["x_grad_rel_err"], gerr) < 1e-4
    print("based", {k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in r.items()})
    return r


def main():
    C = dict(fla_cpu.load(), **mamba_cpu.load())
    d, T = 64, 100                         # T not a multiple of the chunk: tests padding
    layers = {
        "mamba2": C["Mamba2"](d_model=d, d_state=16, d_conv=4, expand=2, headdim=16, chunk_size=32,
                              use_mem_eff_path=False),
        "gated_deltanet": C["GatedDeltaNet"](hidden_size=d, num_heads=2, head_dim=16, expand_v=2, mode="chunk"),
        "gated_deltanet_negeig": C["GatedDeltaNet"](hidden_size=d, num_heads=2, head_dim=16, expand_v=2,
                                                    mode="chunk", allow_neg_eigval=True),
        "gla": C["GatedLinearAttention"](hidden_size=d, num_heads=2, expand_k=0.5, expand_v=1.0, mode="chunk"),
        "rwkv7_layer0": C["RWKV7Attention"](hidden_size=d, head_dim=16, layer_idx=0, num_hidden_layers=2),
        "rwkv7_layer1": C["RWKV7Attention"](hidden_size=d, head_dim=16, layer_idx=1, num_hidden_layers=2),
    }
    for name in ("rwkv7_layer0", "rwkv7_layer1"):   # official init zeros o_proj; randomize it so outputs are non-zero
        torch.nn.init.normal_(layers[name].o_proj.weight, std=0.1)
    res = {}
    x = torch.randn(2, T, d)
    v_first = torch.randn(2, T, d)
    for name, layer in layers.items():
        kw = {"v_first": v_first} if name == "rwkv7_layer1" else {}
        t0 = time.time(); on, xn, gn = run(layer, x, "naive", 32, **kw); tn = time.time() - t0
        t0 = time.time(); oe, xe, ge = run(layer, x, "engine", 32, **kw); te = time.time() - t0
        gerr = max(rel(ge[k], gn[k]) for k in gn)
        r = {"out_rel_err": rel(oe, on), "x_grad_rel_err": rel(xe, xn), "param_grad_max_rel_err": gerr,
             "n_param_grads": len(gn), "out_abs_max": float(on.abs().max()),
             "naive_secs": round(tn, 3), "engine_secs": round(te, 3)}
        r["pass"] = max(r["out_rel_err"], r["x_grad_rel_err"], gerr) < 1e-4
        res[name] = r
        print(name, {k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in r.items()})
    res["based_zoology_vs_hazyresearch_based"] = check_based(x)
    out = {"what": "official layer, scan=engine vs scan=naive (the repos' torch references), float32 (the fla references compute in float32), B=2 T=100 d=64",
           "tolerance": "max relative error < 1e-4", "results": res}
    os.makedirs(os.path.join(ROOT, "baselines"), exist_ok=True)
    p = os.path.join(ROOT, "baselines", "match_check_layers.json")
    json.dump(out, open(p, "w"), indent=1)
    print("all pass" if all(r["pass"] for r in res.values()) else "FAIL", "->", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
