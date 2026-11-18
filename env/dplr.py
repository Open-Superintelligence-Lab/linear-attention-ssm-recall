"""One recurrence engine for all five recurrent baselines.

Every recurrent layer in this repo is a special case of the diagonal-plus-low-rank (DPLR)
recurrence, per head, with state S of shape (dk, dv):

    S_t = diag(w_t) S_{t-1} - b_t (c_t^T S_{t-1}) + k_t v_t^T        o_t = S_t^T q_t

    linear attention (Based)   w = 1,         no b, c
    Mamba-2                    w = scalar,    no b, c        (k = dt*B or B, v = x or dt*x)
    GLA                        w = vector,    no b, c
    Gated DeltaNet             w = alpha,     b = alpha*beta*k, c = k, write key beta*k
    RWKV-7                     w = vector,    b = a*kappa,      c = kappa, write key k~

`dplr_recurrent` is the plain loop (reference, slow). `dplr_chunked` is the chunkwise
parallel form used for training. Derivation of the chunked form:

  Let r_t = S_{t-1}^T c_t (the part the low-rank term removes). Then
      S_t = diag(w_t) S_{t-1} + k_t v_t^T - b_t r_t^T,
  a purely diagonal-decay recurrence with two writes per step. Inside a chunk, with
  G_t = sum_{i<=t} log w_i (G_0 = 0 at the chunk start, state S_0):
      r_t = (c_t * e^{G_{t-1}})^T S_0 + sum_{j<t} <c_t, k_j>_{t-1,j} v_j - sum_{j<t} <c_t, b_j>_{t-1,j} r_j
  where <x, y>_{t,j} = sum_d x_d y_d exp(G_{t,d} - G_{j,d}). This is a unit lower-triangular
  system (I + M) R = H, solved with one triangular solve per chunk. Outputs and the chunk-end
  state then follow in closed form. All exponents are differences G_t - G_j with j <= t, so
  they are <= 0 and never overflow.

Shapes: q, k, b, c (B, H, L, dk); v (B, H, L, dv); logw (B, H, L, dk) or (B, H, L, 1) for a
scalar decay (log w <= 0). Returns o (B, H, L, dv) and the final state (B, H, dk, dv).

`tests/check_dplr.py` compares the two forms (outputs and gradients). Run it first.
"""
import torch
import torch.nn.functional as F


def dplr_recurrent(q, k, v, logw, b=None, c=None, S0=None):
    B, H, L, dk = k.shape
    dv = v.shape[-1]
    S = q.new_zeros(B, H, dk, dv) if S0 is None else S0
    outs = []
    for t in range(L):
        Sn = torch.exp(logw[:, :, t]).unsqueeze(-1) * S
        if b is not None:
            r = torch.einsum("bhk,bhkv->bhv", c[:, :, t], S)
            Sn = Sn - b[:, :, t].unsqueeze(-1) * r.unsqueeze(-2)
        S = Sn + k[:, :, t].unsqueeze(-1) * v[:, :, t].unsqueeze(-2)
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], S))
    return torch.stack(outs, 2), S


def _pair(x, y, Gx, Gy, mask):
    """sum_d x[..., t, d] y[..., j, d] exp(Gx[..., t, d] - Gy[..., j, d]) where mask[t, j], else 0."""
    if Gx.shape[-1] == 1:
        d = Gx - Gy.transpose(-1, -2)
        d = torch.where(mask, d, torch.full_like(d, float("-inf")))
        return (x @ y.transpose(-1, -2)) * torch.exp(d)
    d = Gx.unsqueeze(-2) - Gy.unsqueeze(-3)                    # (..., C, C, dk)
    d = torch.where(mask[..., None], d, torch.full_like(d, float("-inf")))
    return (x.unsqueeze(-2) * y.unsqueeze(-3) * torch.exp(d)).sum(-1)


def dplr_chunked(q, k, v, logw, b=None, c=None, S0=None, chunk=32):
    B, H, L, dk = k.shape
    dv = v.shape[-1]
    C = int(chunk)
    pad = (-L) % C
    delta = b is not None
    if pad:
        p4 = lambda t: F.pad(t, (0, 0, 0, pad))
        q, k, v, logw = p4(q), p4(k), p4(v), p4(logw)
        if delta:
            b, c = p4(b), p4(c)
    N = (L + pad) // C
    rs = lambda t: t.reshape(B, H, N, C, t.shape[-1])
    q, k, v, logw = rs(q), rs(k), rs(v), rs(logw)
    if delta:
        b, c = rs(b), rs(c)
    G = logw.cumsum(-2)                     # inclusive: G_t
    Gp = G - logw                           # exclusive: G_{t-1}
    Ge = G[..., -1:, :]                     # chunk end
    ar = torch.arange(C, device=q.device)
    incl = ar[:, None] >= ar[None, :]
    strict = ar[:, None] > ar[None, :]

    A_qk = _pair(q, k, G, G, incl)
    q_dec = q * torch.exp(G)
    k_end = k * torch.exp(Ge - G)
    if delta:
        A_qb = _pair(q, b, G, G, incl)
        A_ck = _pair(c, k, Gp, G, strict)
        A_cb = _pair(c, b, Gp, G, strict)
        eye = torch.eye(C, device=q.device, dtype=q.dtype)
        Tm = eye + A_cb
        c_dec = c * torch.exp(Gp)
        b_end = b * torch.exp(Ge - G)

    # Everything that does not depend on the carried state is computed for all chunks at once:
    #   R = Tinv (c_dec S + A_ck v) = U S + W, with Tinv = (I + A_cb)^-1
    #   o_n = Qs_n S_n + Ov_n,   S_{n+1} = diag(e^Ge_n) S_n + M_n S_n + Z_n
    # so the serial loop over chunks is one (dk x dk) @ (dk x dv) product per chunk.
    Qs = q_dec
    Ov = A_qk @ v
    Z = k_end.transpose(-1, -2) @ v
    if delta:
        Tinv = torch.linalg.solve_triangular(Tm, eye.expand_as(Tm), upper=False, unitriangular=True)
        U = Tinv @ c_dec                     # (B, H, N, C, dk)
        W = Tinv @ (A_ck @ v)                # (B, H, N, C, dv)
        Qs = Qs - A_qb @ U
        Ov = Ov - A_qb @ W
        bT = b_end.transpose(-1, -2)         # (B, H, N, dk, C)
        M = -(bT @ U)                        # (B, H, N, dk, dk)
        Z = Z - bT @ W
    decay_end = torch.exp(Ge).transpose(-1, -2)   # (B, H, N, dk or 1, 1)
    S = q.new_zeros(B, H, dk, dv) if S0 is None else S0
    states = []
    for n in range(N):
        states.append(S)
        S_next = decay_end[:, :, n] * S + Z[:, :, n]
        if delta:
            S_next = S_next + M[:, :, n] @ S
        S = S_next
    Sst = torch.stack(states, 2)             # state at each chunk start (B, H, N, dk, dv)
    o = (Qs @ Sst + Ov).reshape(B, H, N * C, dv)[:, :, :L]
    return o, S


def dplr(q, k, v, logw, b=None, c=None, chunk=32, mode="chunked"):
    if mode == "recurrent":
        return dplr_recurrent(q, k, v, logw, b, c)
    return dplr_chunked(q, k, v, logw, b, c, chunk=chunk)
