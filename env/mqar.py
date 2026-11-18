"""Recall tasks, generated with NumPy from a fixed seed (nothing is downloaded).

Token layout (vocab 512):
  0          filler token ("zero" filler)
  1..192     keys
  193..384   values
  385..511   noise tokens ("noise" filler)

MQAR (multi-query associative recall, Zoology 2312.04927 style):
  k1 v1 k2 v2 ... kD vD | [update pairs] | query region
  In the query region each queried key k appears followed by its value v; the model is
  scored on predicting v at the position of k (next-token target). Other positions are
  ignored (-100). Filler fills the free positions of the query region.

Needle: noise haystack with n_needles key-value pairs at random positions, then all
needle keys are queried at the end of the sequence.

Every function returns (x, y, meta): x int64 (n, L) inputs, y int64 (n, L) targets with
-100 where not scored, meta dict of (n, L) arrays: dist (tokens from the key's latest
write to the query, -1 where not scored) and updated (query asks a re-assigned key).
"""
import numpy as np

VOCAB = 512
FILLER = 0
N_KEYS = 192
KEY0 = 1
VAL0 = KEY0 + N_KEYS     # 193
NOISE0 = VAL0 + N_KEYS   # 385
IGNORE = -100


def _fill(rng, shape, filler):
    if filler == "zero":
        return np.zeros(shape, np.int64)
    if filler == "noise":
        return rng.integers(NOISE0, VOCAB, size=shape).astype(np.int64)
    raise ValueError(f"filler must be zero or noise, got {filler}")


def mqar(rng, n, L, D, filler="zero", layout="spread", overwrite=0.0, query_order="random", n_queries=None, dense=False):
    """n sequences of length L with D stored pairs.

    layout: spread = queries at random slots of the query region (filler between them);
            gap = filler first, then all queries packed at the end (controls distance).
    overwrite: fraction of keys re-assigned a new value in an update block right after the pairs.
    query_order: random | same (order the pairs were stored) | reverse.
    dense: training only. Every free slot of the query region holds a query (keys drawn with
           replacement), so a sequence gives (L - 2D - 2U) / 2 scored tokens instead of D.
    """
    U = int(round(overwrite * D))
    Q = D if n_queries is None else int(n_queries)
    if D > N_KEYS or (Q > D and not dense):
        raise ValueError(f"need D <= {N_KEYS} and n_queries <= D (D={D}, Q={Q})")
    if dense:
        Q = (L - 2 * D - 2 * U) // 2
    if 2 * D + 2 * U + 2 * Q > L:
        raise ValueError(f"L={L} too short for D={D}, U={U}, Q={Q}")
    x = _fill(rng, (n, L), filler)
    y = np.full((n, L), IGNORE, np.int64)
    dist = np.full((n, L), -1, np.int64)
    upd_mask = np.zeros((n, L), bool)
    for i in range(n):
        keys = rng.choice(N_KEYS, size=D, replace=False) + KEY0
        vals = rng.integers(0, N_KEYS, size=D) + VAL0
        x[i, 0:2 * D:2] = keys
        x[i, 1:2 * D:2] = vals
        write_pos = np.arange(D) * 2           # position of each key's latest write
        vals_now = vals.copy()
        updated = np.zeros(D, bool)
        pos = 2 * D
        if U:
            upd = rng.choice(D, size=U, replace=False)
            newv = rng.integers(0, N_KEYS, size=U) + VAL0
            same = newv == vals[upd]
            newv[same] = (newv[same] - VAL0 + 1) % N_KEYS + VAL0
            x[i, pos:pos + 2 * U:2] = keys[upd]
            x[i, pos + 1:pos + 2 * U:2] = newv
            vals_now[upd] = newv
            write_pos[upd] = pos + 2 * np.arange(U)
            updated[upd] = True
            pos += 2 * U
        if dense:
            qi = np.concatenate([rng.permutation(D), rng.integers(0, D, size=max(0, Q - D))])[:Q]
            rng.shuffle(qi)
        elif query_order == "random":
            qi = rng.permutation(D)[:Q]
        elif query_order == "same":
            qi = np.arange(D)[:Q]
        elif query_order == "reverse":
            qi = np.arange(D)[::-1][:Q]
        else:
            raise ValueError(query_order)
        slots = (L - pos) // 2
        if layout == "spread":
            s = np.sort(rng.choice(slots, size=Q, replace=False))
        elif layout == "gap":
            s = np.arange(slots - Q, slots)
        else:
            raise ValueError(layout)
        p = pos + 2 * s
        x[i, p] = keys[qi]
        x[i, p + 1] = vals_now[qi]
        y[i, p] = vals_now[qi]
        dist[i, p] = p - write_pos[qi]
        upd_mask[i, p] = updated[qi]
    return x, y, dict(dist=dist, updated=upd_mask)


def needle(rng, n, L, n_needles=1, filler="noise"):
    """Noise haystack with n_needles pairs at random places; all needles are queried at the end."""
    m = int(n_needles)
    if 4 * m > L:
        raise ValueError("L too short")
    x = _fill(rng, (n, L), filler)
    y = np.full((n, L), IGNORE, np.int64)
    dist = np.full((n, L), -1, np.int64)
    upd_mask = np.zeros((n, L), bool)
    hay = L - 2 * m
    for i in range(n):
        keys = rng.choice(N_KEYS, size=m, replace=False) + KEY0
        vals = rng.integers(0, N_KEYS, size=m) + VAL0
        s = np.sort(rng.choice(hay // 2, size=m, replace=False)) * 2
        x[i, s] = keys
        x[i, s + 1] = vals
        order = rng.permutation(m)
        p = hay + 2 * np.arange(m)
        x[i, p] = keys[order]
        x[i, p + 1] = vals[order]
        y[i, p] = vals[order]
        dist[i, p] = p - s[order]
    return x, y, dict(dist=dist, updated=upd_mask)


def make(task, rng, n, L, cfg):
    """Dispatch by task name with the run config (see env/config.py for the knobs)."""
    if task == "mqar":
        D = int(cfg["pairs"]) if int(cfg.get("pairs", 0)) > 0 else max(1, int(L * float(cfg["pairs_frac"])))
        return mqar(rng, n, L, D, filler=cfg["filler"], layout=cfg["layout"],
                    overwrite=float(cfg["overwrite"]), query_order=cfg["query_order"],
                    dense=bool(int(cfg.get("dense_train", 0))))
    if task == "needle":
        return needle(rng, n, L, n_needles=int(cfg["n_needles"]), filler=cfg.get("needle_filler", "noise"))
    raise ValueError(f"unknown task {task}")
