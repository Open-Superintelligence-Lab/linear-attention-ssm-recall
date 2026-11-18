"""Run the official Mamba-2 layer (state-spaces/mamba, mamba_ssm/modules/mamba2.py) on a CPU.

Source: state-spaces/mamba, pinned commit in data/download.sh (Apache-2.0).
The Mamba2 class is imported unchanged (parameters, A/dt/D init, in_proj split, conv, gate,
norm, out_proj). It runs with use_mem_eff_path=False, the module's own non-fused branch, and
without the causal_conv1d package, so the conv is the module's own nn.Conv1d fallback.
Two Triton calls remain and are replaced:

  mamba_chunk_scan_combined     scan="naive": ssd_chunk_scan_combined_ref (the repo's own torch
                                reference, mamba_ssm/ops/triton/ssd_combined.py)
                                scan="engine": this repo's chunked DPLR engine (env/dplr.py)
  RMSNorm (gated) forward       mamba_ssm.ops.triton.layernorm_gated.rms_norm_ref

The scan switch is shared with baselines/official/fla_cpu.py (fla_cpu.set_scan).
tests/check_official.py compares engine against naive on this layer.
"""
import os
import sys
import types

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from baselines.official import fla_cpu

ROOT = fla_cpu.ROOT
MAMBA = os.path.join(ROOT, "data", "ext", "mamba")
_LOADED = {}


def load():
    if _LOADED:
        return _LOADED
    from baselines.official.triton_stub import install
    install()
    if MAMBA not in sys.path:
        sys.path.insert(0, MAMBA)
    if "mamba_ssm" not in sys.modules:       # skip the package __init__ (it imports CUDA extensions)
        pkg = types.ModuleType("mamba_ssm")
        pkg.__path__ = [os.path.join(MAMBA, "mamba_ssm")]
        sys.modules["mamba_ssm"] = pkg
    import mamba_ssm.modules.mamba2 as M
    import mamba_ssm.ops.triton.layernorm_gated as LNG
    from mamba_ssm.ops.triton.ssd_combined import ssd_chunk_scan_combined_ref
    from env.dplr import dplr

    assert M.causal_conv1d_fn is None        # so Mamba2 uses its nn.Conv1d fallback

    def norm_fwd(self, x, z=None):
        return LNG.rms_norm_ref(x, self.weight, self.bias, z=z, eps=self.eps, group_size=self.group_size,
                                norm_before_gate=self.norm_before_gate, upcast=True).to(x.dtype)

    LNG.RMSNorm.forward = norm_fwd

    def scan(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, dt_softplus=False,
             initial_states=None, seq_idx=None, cu_seqlens=None, dt_limit=(0.0, float("inf")),
             return_final_states=False, return_varlen_states=False, **kw):
        assert initial_states is None and seq_idx is None and cu_seqlens is None
        assert not return_final_states and not return_varlen_states
        assert dt_limit == (0.0, float("inf"))
        Bsz, T, H, P = x.shape
        if fla_cpu.SCAN["mode"] == "naive":
            pad = (-T) % chunk_size           # the reference needs T to be a multiple of chunk_size
            p = lambda t: F.pad(t, (0, 0, 0, 0, 0, pad)) if t is not None else None
            out = ssd_chunk_scan_combined_ref(p(x), F.pad(dt, (0, 0, 0, pad)), A, p(B), p(C), chunk_size,
                                              D=D, z=p(z), dt_bias=dt_bias, dt_softplus=dt_softplus)
            return out[:, :T]
        if dt_bias is not None:
            dt = dt + dt_bias
        if dt_softplus:
            dt = F.softplus(dt)
        G = B.shape[2]
        Bh = repeat(B, "b l g n -> b (g h) l n", h=H // G)
        Ch = repeat(C, "b l g n -> b (g h) l n", h=H // G)
        dth = rearrange(dt, "b l h -> b h l 1")
        xh = rearrange(x, "b l h p -> b h l p")
        # S_t = exp(dt A) S_{t-1} + (dt B_t) x_t^T,  y_t = S_t^T C_t + D x_t
        o, _ = dplr(Ch, dth * Bh, xh, dth * A.view(1, H, 1, 1), chunk=fla_cpu.SCAN["chunk"])
        y = rearrange(o, "b h l p -> b l h p")
        if D is not None:
            y = y + x * (D.view(H, 1) if D.dim() == 1 else D)
        return y if z is None else y * F.silu(z)

    M.mamba_chunk_scan_combined = scan
    _LOADED.update(Mamba2=M.Mamba2)
    return _LOADED
