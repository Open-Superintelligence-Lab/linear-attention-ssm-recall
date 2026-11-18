"""Build zoology's own MQAR model (HazyResearch/zoology, pinned) around an official mixer.

Same layout as zoology/experiments/models_repo.py: vocab 8,192, 2 TransformerBlocks, no MLP
(state_mixer Identity), no position embeddings, layer 0 = BaseConv(kernel 3), layer 1 = method.
Mamba-2 uses the method in both layers (zoology's Mamba2 setup).
"""
import os
import sys

from baselines.official import fla_cpu

ZOO = os.path.join(fla_cpu.ROOT, "data", "ext", "zoology")


def zoology():
    if ZOO not in sys.path:
        sys.path.insert(0, ZOO)
    import types
    stub = "wandb" not in sys.modules
    if stub:                                  # zoology.utils imports wandb for logging only
        sys.modules["wandb"] = types.ModuleType("wandb")
    import zoology.model as zm
    import zoology.config as zc
    if stub:                                  # drop it again so other packages see no wandb
        del sys.modules["wandb"]
    f = zm.TokenEmbeddings.__init__          # embeddings default to device='cuda'; use the CPU
    f.__defaults__ = tuple("cpu" if d == "cuda" else d for d in f.__defaults__)
    return zm, zc


MIXERS = {   # name -> (module path, kwargs) ; values from zoology models_repo.py
    "attention": ("zoology.mixers.attention.MHA", {"dropout": 0.1, "num_heads": 2}),
    "based": ("zoology.mixers.based.Based", {"feature_dim": 16, "feature_name": "taylor_exp",
                                             "num_key_value_heads": 1, "num_heads": 1,
                                             "train_view": "quadratic"}),
    "gated_deltanet": ("baselines.official.mixers.GatedDeltaNet", {"num_heads": 2, "use_gate": False,
                                                                  "use_short_conv": True, "conv_size": 4}),
    "gla": ("baselines.official.mixers.GatedLinearAttention", {"num_heads": 2, "use_short_conv": False}),
    "rwkv7": ("baselines.official.mixers.RWKV7Attention", {"head_dim": 64, "decay_low_rank_dim": 16,
                                                          "gate_low_rank_dim": 64, "a_low_rank_dim": 16,
                                                          "v_low_rank_dim": 16}),
    "mamba2": ("baselines.official.mixers.Mamba2", {"d_state": 16}),
}


def build(method, d_model=64, l_max=1024, vocab=8192, n_layers=2, **override):
    zm, zc = zoology()
    name, kw = MIXERS[method]
    kw = {**kw, **override}
    if method in ("based",):
        kw["l_max"] = l_max
    mixer = {"name": name, "kwargs": kw}
    if method != "mamba2":
        conv = {"name": "zoology.mixers.base_conv.BaseConv",
                "kwargs": {"l_max": l_max, "kernel_size": 3, "implicit_long_conv": True}}
        mixer = {"name": "zoology.mixers.hybrid.Hybrid", "kwargs": {"configs": [conv, mixer]}}
    cfg = zc.ModelConfig(d_model=d_model, n_layers=n_layers, vocab_size=vocab, max_position_embeddings=0,
                         sequence_mixer=mixer, state_mixer={"name": "torch.nn.Identity", "kwargs": {}},
                         block_type="TransformerBlock", name=method)
    return zm.LanguageModel(cfg)
