"""Run the official flash-linear-attention layer classes on a CPU (no Triton).

Source: fla-org/flash-linear-attention, pinned commit in data/download.sh (MIT license).
The layer classes GatedDeltaNet, GatedLinearAttention and RWKV7Attention are imported unchanged
from data/ext/flash-linear-attention. Their __init__ (parameters, init) and forward (projections,
gates, conv, norms, output path) are the official code. Only the calls that launch Triton kernels
are replaced, each by fla's own PyTorch reference function where one exists:

  Triton call                               replaced by
  RMSNorm / GroupNorm forward               fla.modules.norm.layernorm.rms_norm_ref / group_norm_ref
  FusedRMSNormGated forward (swish)         fla.modules.norm.layernorm_gated.rms_norm_ref(z=g)
  FusedRMSNormGated forward (sigmoid)       rms_norm_ref(x) * sigmoid(g)          (no fla ref; 1 line)
  ShortConvolution / causal_conv1d          nn.Conv1d.forward (the class's own parent) + SiLU
  l2norm / l2_norm                          x / sqrt(sum x^2 + 1e-6), the kernel's formula (1 line)
  token_shift                               fla.modules.token_shift.token_shift_ref
  fused_addcmul_rwkv7                       fla.ops.rwkv7.fused_addcmul.torch_addcmul_rwkv7
  fused_k_rwkv7                             fla.ops.rwkv7.fused_k_update.k_update_ref
  gate_output_correction                    fla.ops.rwkv7.naive.naive_gate_output_correction
  GDN in-kernel gate                        fla.ops.gated_delta_rule.gate.naive_gdn_gate
  chunk/fused_recurrent_gated_delta_rule    scan (below)
  chunk/fused_recurrent/fused_chunk_gla     scan (below)
  chunk_rwkv7 / fused_mul_recurrent_rwkv7   scan (below)

scan="naive": fla's own token-by-token references (naive_recurrent_gated_delta_rule,
naive_recurrent_gla, generalized_delta_rule.dplr.naive.dplr_recurrence). Exact, slow.
scan="engine": this repo's chunked DPLR engine (env/dplr.py), same math, about 10-50x faster
on a CPU. tests/check_official.py checks engine against naive on the official layers
(forward outputs and gradients) before any run uses the engine. In engine mode the conv is a
shift-and-add (same result as nn.Conv1d, faster backward on CPU); the check covers it.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FLA = os.path.join(ROOT, "data", "ext", "flash-linear-attention")

SCAN = {"mode": "engine", "chunk": 32}
_LOADED = {}


def set_scan(mode, chunk=None):
    assert mode in ("engine", "naive")
    SCAN["mode"] = mode
    if chunk:
        SCAN["chunk"] = int(chunk)


def _bthk_to_bhtk(*xs):
    return [x.transpose(1, 2) for x in xs]


def _l2(x, eps=1e-6):                            # same formula as fla l2norm_fwd_kernel
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + eps)


def _conv_engine(x, w, act):                     # x (B, T, D), w (D, K)
    K, T = w.shape[1], x.shape[1]
    xp = F.pad(x, (0, 0, K - 1, 0))
    y = sum(xp[:, i:i + T] * w[:, i] for i in range(K))
    return F.silu(y) if act else y


def _conv(x, weight, bias, act, module=None):
    if SCAN["mode"] == "engine" and bias is None:
        return _conv_engine(x, weight.reshape(weight.shape[0], -1), act)
    T = x.shape[1]
    if module is not None:
        y = nn.Conv1d.forward(module, x.transpose(1, 2))[..., :T].transpose(1, 2)
    else:
        w = weight.reshape(weight.shape[0], 1, -1)
        y = F.conv1d(x.transpose(1, 2), w, bias, padding=w.shape[-1] - 1, groups=w.shape[0])[..., :T].transpose(1, 2)
    return F.silu(y) if act else y


def load():
    """Import fla under the Triton stub and patch the kernel calls. Returns the three classes."""
    if _LOADED:
        return _LOADED
    if FLA not in sys.path:
        sys.path.insert(0, FLA)
    from baselines.official.triton_stub import install
    install()
    import warnings
    warnings.filterwarnings("ignore", message=".*FLA implementation of RWKV.*")

    import fla.layers.gated_deltanet as L_gdn
    import fla.layers.gla as L_gla
    import fla.layers.rwkv7 as L_rwkv
    import fla.modules.norm.layernorm as LN
    import fla.modules.norm.fused_norm_gate as FNG
    import fla.modules.causal_conv1d.ops as CONV
    from fla.modules.norm.layernorm_gated import rms_norm_ref as rms_norm_gated_ref
    from fla.modules.token_shift import token_shift_ref
    from fla.ops.rwkv7.fused_addcmul import torch_addcmul_rwkv7
    from fla.ops.rwkv7.fused_k_update import k_update_ref
    from fla.ops.rwkv7.naive import naive_gate_output_correction
    from fla.ops.gated_delta_rule.gate import naive_gdn_gate
    from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
    from fla.ops.gla.naive import naive_recurrent_gla
    from fla.ops.generalized_delta_rule.dplr.naive import dplr_recurrence
    from env.dplr import dplr

    # ---- norms
    def rmsnorm_fwd(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        w = self.weight
        if getattr(self, "zero_centered_weight", False) and w is not None:
            w = w.float() + 1
        if w is None:
            w = torch.ones(x.shape[-1], dtype=x.dtype)
        return LN.rms_norm_ref(x, w, self.bias, residual=residual, eps=self.eps, prenorm=prenorm)

    def groupnorm_fwd(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return LN.group_norm_ref(x, self.weight, self.bias, self.num_groups, residual=residual,
                                 eps=self.eps, is_rms_norm=self.is_rms_norm, prenorm=prenorm)

    def gated_fwd(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
        assert residual is None and not prenorm
        w = self.weight if self.weight is not None else torch.ones(x.shape[-1], dtype=x.dtype)
        if self.activation in ("swish", "silu"):
            return rms_norm_gated_ref(x, w, self.bias, z=g, eps=self.eps, norm_before_gate=True, upcast=False)
        return LN.rms_norm_ref(x, w, self.bias, eps=self.eps) * torch.sigmoid(g)

    LN.RMSNorm.forward = rmsnorm_fwd
    LN.GroupNorm.forward = groupnorm_fwd
    FNG.FusedRMSNormGated.forward = gated_fwd

    # ---- short conv
    def shortconv_fwd(self, x, residual=None, mask=None, cache=None, output_final_state=False,
                      cu_seqlens=None, chunk_indices=None, **kw):
        assert residual is None and cache is None and cu_seqlens is None and mask is None
        return _conv(x, self.weight, self.bias, self.activation is not None, module=self), None

    def causal_conv1d(x, weight=None, bias=None, residual=None, initial_state=None,
                      output_final_state=False, activation=None, backend=None, cu_seqlens=None, **kw):
        assert residual is None and initial_state is None and cu_seqlens is None
        return _conv(x, weight, bias, activation is not None), None

    CONV.ShortConvolution.forward = shortconv_fwd
    L_gdn.causal_conv1d = causal_conv1d

    # ---- Gated DeltaNet scan (inputs [B, T, H, K]; in-kernel l2norm, beta sigmoid, gate)
    def gdn_rule(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
                 use_qk_l2norm_in_kernel=False, use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
                 use_gate_in_kernel=False, A_log=None, dt_bias=None, cu_seqlens=None, **kw):
        assert initial_state is None and cu_seqlens is None
        if use_qk_l2norm_in_kernel:
            q, k = _l2(q), _l2(k)
        if use_beta_sigmoid_in_kernel:
            beta = beta.sigmoid() * (2.0 if allow_neg_eigval else 1.0)
        if use_gate_in_kernel:
            g = naive_gdn_gate(g, A_log, dt_bias)
        K = q.shape[-1]
        scale = K ** -0.5 if scale is None else scale
        if SCAN["mode"] == "naive":
            return naive_recurrent_gated_delta_rule(q, k, v, beta, g, scale=scale,
                                                    output_final_state=output_final_state)
        q, k, v = _bthk_to_bhtk(q, k, v)
        g, beta = g.transpose(1, 2)[..., None], beta.transpose(1, 2)[..., None]   # (B, H, T, 1)
        bk = beta * k
        o, S = dplr(q * scale, bk, v, g, b=g.exp() * bk, c=k, chunk=SCAN["chunk"])
        return o.transpose(1, 2), (S if output_final_state else None)

    L_gdn.chunk_gated_delta_rule = gdn_rule
    L_gdn.fused_recurrent_gated_delta_rule = gdn_rule

    # ---- GLA scan (vector log-decay gk, scale K^-0.5)
    def gla_rule(q, k, v, g=None, gk=None, scale=None, initial_state=None, output_final_state=False,
                 cu_seqlens=None, **kw):
        assert initial_state is None and cu_seqlens is None
        gk = g if gk is None else gk
        if SCAN["mode"] == "naive":
            assert scale is None
            return naive_recurrent_gla(q, k, v, gk, output_final_state=output_final_state)
        K = q.shape[-1]
        scale = K ** -0.5 if scale is None else scale
        q, k, v, gk = _bthk_to_bhtk(q, k, v, gk)
        o, S = dplr(q * scale, k, v, gk, chunk=SCAN["chunk"])
        return o.transpose(1, 2), (S if output_final_state else None)

    L_gla.chunk_gla = gla_rule
    L_gla.fused_chunk_gla = gla_rule
    L_gla.fused_recurrent_gla = gla_rule

    # ---- RWKV-7 scan: S_t = S_{t-1} diag(e^w) + b (a^T S_{t-1}) + k v^T  (fla DPLR form)
    def rwkv_rule(r, w, k, v, a, b, scale=1.0, initial_state=None, output_final_state=False,
                  cu_seqlens=None, **kw):
        assert initial_state is None and cu_seqlens is None
        r, w, k, v, a, b = _bthk_to_bhtk(r, w, k, v, a, b)
        if SCAN["mode"] == "naive":
            K = r.shape[-1]          # dplr_recurrence multiplies q by K^-0.5 itself
            o, S = dplr_recurrence(r * (scale * K ** 0.5), k, v, a, b, w, output_final_state=output_final_state)
        else:                        # engine: S_t = diag(w) S - b_e (c^T S) + k v^T, so b_e = -b, c = a
            o, S = dplr(r * scale, k, v, w, b=-b, c=a, chunk=SCAN["chunk"])
        return o.transpose(1, 2), (S if output_final_state else None)

    def rwkv_rule_recurrent(r, w, k, v, kk, a, scale=1.0, initial_state=None, output_final_state=False,
                            cu_seqlens=None, **kw):
        return rwkv_rule(r, w, k, v, -kk, kk * a, scale, initial_state, output_final_state, cu_seqlens)

    L_rwkv.chunk_rwkv7 = rwkv_rule
    L_rwkv.fused_mul_recurrent_rwkv7 = rwkv_rule_recurrent
    L_rwkv.token_shift = lambda x, cu_seqlens=None, output_cache=False, cache=None, **kw: \
        (token_shift_ref(x, cu_seqlens), None) if output_cache else token_shift_ref(x, cu_seqlens)
    L_rwkv.fused_addcmul_rwkv7 = torch_addcmul_rwkv7
    L_rwkv.fused_k_rwkv7 = lambda k, a, ka, **kw: k_update_ref(k, a, ka)
    L_rwkv.gate_output_correction = naive_gate_output_correction
    L_rwkv.l2_norm = lambda x, *a, **kw: _l2(x)

    _LOADED.update(GatedDeltaNet=L_gdn.GatedDeltaNet, GatedLinearAttention=L_gla.GatedLinearAttention,
                   RWKV7Attention=L_rwkv.RWKV7Attention)
    return _LOADED
