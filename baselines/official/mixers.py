"""Official sequence mixers with the call signature zoology's model expects: Mixer(d_model, layer_idx, **kw).

zoology (HazyResearch/zoology, the MQAR benchmark code) builds each baseline as
LanguageModel -> 2 TransformerBlocks (LayerNorm pre-norm, no MLP: state_mixer = Identity),
where layer 0 is zoology's BaseConv and layer 1 is the method (zoology.mixers.hybrid.Hybrid).
zoology keeps its own copies of the fla layers; here layer 1 is the pinned upstream class instead:

  gated_deltanet  fla.layers.GatedDeltaNet        head_dim = d/num_heads, expand_v 2 (zoology's copy)
  gla             fla.layers.GatedLinearAttention expand_k 0.5, expand_v 1
  rwkv7           fla.layers.RWKV7Attention       initialized as layer 1 of 2; forward runs as the
                                                  first RWKV layer (v_first = v), the same function
                                                  as zoology's copy with v_first=None
  mamba2          mamba_ssm.modules.mamba2.Mamba2 both layers, no BaseConv, headdim 64 (as zoology);
                                                  zoology's Mamba2Block uses RMSNorm, here LayerNorm
  based           zoology.mixers.based.Based      HazyResearch's own MQAR Based layer (Taylor map,
                                                  quadratic train view); checked against
                                                  HazyResearch/based LinearAttention in
                                                  tests/check_official.py
  attention       zoology.mixers.attention.MHA

All fla / mamba layers run through baselines/official/{fla_cpu,mamba_cpu}.py on a CPU.
"""
import torch.nn as nn

from baselines.official import fla_cpu, mamba_cpu


def _first(out):
    return out[0] if isinstance(out, tuple) else out


class _Wrap(nn.Module):
    def __init__(self, layer, state_floats):
        super().__init__()
        self.layer = layer
        self._state = state_floats

    def forward(self, x, *a, **k):
        return _first(self.layer(x))

    def state_size(self, sequence_length=2048, **kw):
        return self._state


def GatedDeltaNet(d_model, layer_idx=None, num_heads=2, use_gate=False, use_short_conv=True,
                  conv_size=4, expand_v=2, allow_neg_eigval=False, **kw):
    cls = fla_cpu.load()["GatedDeltaNet"]
    hd = d_model // num_heads
    layer = cls(hidden_size=d_model, expand_v=expand_v, head_dim=hd, num_heads=num_heads,
                use_gate=use_gate, use_short_conv=use_short_conv, conv_size=conv_size,
                allow_neg_eigval=allow_neg_eigval, layer_idx=layer_idx)
    return _Wrap(layer, num_heads * hd * hd * expand_v)


def GatedLinearAttention(d_model, layer_idx=None, num_heads=2, use_short_conv=False,
                         expand_k=0.5, expand_v=1.0, **kw):
    cls = fla_cpu.load()["GatedLinearAttention"]
    layer = cls(hidden_size=d_model, expand_k=expand_k, expand_v=expand_v, num_heads=num_heads,
                use_short_conv=use_short_conv, layer_idx=layer_idx)
    dk, dv = int(d_model * expand_k), int(d_model * expand_v)
    return _Wrap(layer, dk * dv // num_heads)


def RWKV7Attention(d_model, layer_idx=None, head_dim=64, decay_low_rank_dim=16, gate_low_rank_dim=64,
                   a_low_rank_dim=16, v_low_rank_dim=16, **kw):
    cls = fla_cpu.load()["RWKV7Attention"]
    layer = cls(hidden_size=d_model, head_dim=head_dim,
                decay_low_rank_dim=decay_low_rank_dim, gate_low_rank_dim=gate_low_rank_dim,
                a_low_rank_dim=a_low_rank_dim, v_low_rank_dim=v_low_rank_dim,
                layer_idx=1, num_hidden_layers=2)   # init as layer 1 of 2, its place in the model
    layer.layer_idx = 0   # forward: v_first = v (zoology's copy computes lerp(v, v, .) = v here)
    return _Wrap(layer, d_model * head_dim)


def Mamba2(d_model, layer_idx=None, d_state=16, headdim=64, expand=2, chunk_size=32, **kw):
    cls = mamba_cpu.load()["Mamba2"]
    layer = cls(d_model=d_model, d_state=d_state, headdim=headdim, expand=expand,
                chunk_size=chunk_size, use_mem_eff_path=False, layer_idx=layer_idx)
    return _Wrap(layer, expand * d_model * d_state)
