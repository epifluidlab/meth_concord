"""Link-shuffle null: a contact keeps its left end; its partner is replaced
by a contact end from ANOTHER cell of the same stratum.

The donor is an end of a read pair at least ``min_distance`` long -- the
kind of end the partner it stands in for is -- on the partner's side of the
anchor (to its right), at a distance in the contact's own log-distance bin,
and optionally inside given bounds (step 5: the anchor's bin, where the real
partner is).  Where the donor's own partner sits does not matter.  Because
the donor comes from another cell it can never re-form a read pair.

WHY ANOTHER CELL.  Within one cell, at single-cell depth, a contact has one
candidate donor or none: on synthetic cells with six contacts per Mb, 42% of
the contacts that had any had exactly one and 28% had two.  Every draw then
repeats the same donor, the null comes out far narrower than a fresh
re-pairing would be, and a real value with no coordination in it can sit
four standard deviations above it.  Pooled over the cells of a stratum (cell
type, optionally x batch) the candidates are many, each draw is a new
partner, and nearly every contact gets one.

A contact whose own distance bin holds no donor may borrow from up to
``max_widen`` neighbouring distance bins on each side; how often that
happened is returned, so a run can say how much of its null is matched
exactly.
"""

from __future__ import annotations

import numpy as np

#: Genome coordinate = chromosome code * stride + position.
GENOME_STRIDE = np.int64(1) << np.int64(33)


def log_dbins(min_d: int, max_d: int, per_decade: int = 4) -> np.ndarray:
    """Log-spaced distance-bin edges across [min_d, max_d]."""
    lo, hi = np.log10(max(min_d, 1)), np.log10(max(max_d, min_d + 1))
    n = max(1, int(np.ceil((hi - lo) * per_decade)))
    return np.logspace(lo, hi, n + 1)


def donor_ends(chrom_codes, pos1, pos2, p1, p2, inf1, inf2,
               min_distance: int):
    """Genome coordinates and posteriors of every informative end of a read
    pair at least ``min_distance`` long (its partner's call does not
    matter)."""
    pos1 = np.asarray(pos1, np.int64)
    pos2 = np.asarray(pos2, np.int64)
    codes = np.asarray(chrom_codes, np.int64)
    lr = np.abs(pos2 - pos1) >= int(min_distance)
    m1, m2 = lr & inf1, lr & inf2
    g = np.concatenate([codes[m1] * GENOME_STRIDE + pos1[m1],
                        codes[m2] * GENOME_STRIDE + pos2[m2]])
    p = np.concatenate([np.asarray(p1, float)[m1], np.asarray(p2, float)[m2]])
    return g, p


def build_pool(ends_by_cell: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One stratum's donor pool from ``{cell_id: (g, p)}``: genome
    coordinates sorted, their posteriors, and the cell each came from."""
    if not ends_by_cell:
        return (np.empty(0, np.int64), np.empty(0), np.empty(0, np.int32))
    g = np.concatenate([e[0] for e in ends_by_cell.values()])
    p = np.concatenate([e[1] for e in ends_by_cell.values()])
    cell = np.concatenate([np.full(e[0].size, cid, np.int32)
                           for cid, e in ends_by_cell.items()])
    order = np.argsort(g, kind="mergesort")
    return g[order], p[order], cell[order]


def draw_link_shuffle(anchor_g, d, own_cell: int, pool, own_g, dbins, rng,
                      n_draws: int = 1, bounds=None, max_widen: int = 2,
                      max_rounds: int = 32):
    """Donor posteriors for one cell's contacts.

    ``anchor_g``: the contacts' left ends as genome coordinates; ``d``:
    their lengths; ``pool``: (g, p, cell) from :func:`build_pool`;
    ``own_g``: this cell's own donor ends, sorted, which are never used.
    ``bounds``: optional (lo, hi) per contact the donor must lie in.

    Returns (donor_p, level): ``donor_p`` is n_draws x contacts, NaN where a
    contact has no donor; ``level`` is how many distance bins its window was
    widened by (0 = its own bin), -1 for none.  The window -- and so whether
    a contact has a donor -- is fixed before drawing, so it does not depend
    on the draw.
    """
    anchor_g = np.asarray(anchor_g, np.int64)
    n = anchor_g.size
    level = np.full(n, -1, np.int64)
    donor_p = np.full((max(1, int(n_draws)), n), np.nan)
    pool_g, pool_p, pool_cell = pool
    if n == 0 or pool_g.size == 0:
        return donor_p, level
    own_g = np.asarray(own_g, np.int64)
    nb = len(dbins) - 1
    k = np.clip(np.searchsorted(dbins, np.asarray(d), side="right") - 1,
                0, nb - 1)
    L = np.zeros(n, np.int64)
    H = np.zeros(n, np.int64)
    for w in range(int(max_widen) + 1):
        todo = np.flatnonzero(level < 0)
        if todo.size == 0:
            break
        lo = anchor_g[todo] + dbins[np.maximum(k[todo] - w, 0)]
        hi = anchor_g[todo] + dbins[np.minimum(k[todo] + w + 1, nb)]
        if bounds is not None:
            lo = np.maximum(lo, np.asarray(bounds[0])[todo])
            hi = np.minimum(hi, np.asarray(bounds[1])[todo])
        lo = np.ceil(lo).astype(np.int64)
        hi = np.ceil(hi).astype(np.int64)
        lft = np.searchsorted(pool_g, lo, side="left")
        rgt = np.searchsorted(pool_g, hi, side="left")
        own = (np.searchsorted(own_g, hi, side="left")
               - np.searchsorted(own_g, lo, side="left"))
        ok = (rgt - lft - own) >= 1
        hit = todo[ok]
        level[hit] = w
        L[hit], H[hit] = lft[ok], rgt[ok]
    has = np.flatnonzero(level >= 0)
    for dr in range(donor_p.shape[0]):
        alive = has.copy()
        for _ in range(int(max_rounds)):
            if alive.size == 0:
                break
            pick = L[alive] + (rng.random(alive.size)
                               * (H[alive] - L[alive])).astype(np.int64)
            good = pool_cell[pick] != own_cell
            donor_p[dr, alive[good]] = pool_p[pick[good]]
            alive = alive[~good]
        for i in alive:            # mostly this cell's own ends: pick exactly
            cand = np.arange(L[i], H[i])
            cand = cand[pool_cell[cand] != own_cell]
            donor_p[dr, i] = pool_p[rng.choice(cand)]
    return donor_p, level


def read_donor_ends(posteriors_path, min_distance: int,
                    min_sites: int) -> dict:
    """{chrom: (positions, posteriors)} of one cell's donor ends: every end
    with a call (``n >= min_sites``) of a read pair at least
    ``min_distance`` long."""
    import pandas as pd
    df = pd.read_parquet(posteriors_path,
                         columns=["chrom", "pos1", "pos2", "n1", "n2",
                                  "posterior1", "posterior2"])
    chrom = df["chrom"].astype(str).to_numpy()
    pos1 = df["pos1"].to_numpy(np.int64)
    pos2 = df["pos2"].to_numpy(np.int64)
    p1 = df["posterior1"].to_numpy(np.float64, na_value=np.nan)
    p2 = df["posterior2"].to_numpy(np.float64, na_value=np.nan)
    lr = np.abs(pos2 - pos1) >= int(min_distance)
    m1 = lr & (df["n1"].to_numpy() >= min_sites) & np.isfinite(p1)
    m2 = lr & (df["n2"].to_numpy() >= min_sites) & np.isfinite(p2)
    ch = np.concatenate([chrom[m1], chrom[m2]])
    pos = np.concatenate([pos1[m1], pos2[m2]])
    pp = np.concatenate([p1[m1], p2[m2]])
    return {c: (pos[ch == c], pp[ch == c]) for c in np.unique(ch)}


def pools_by_stratum(ends_by_cell: list, strata: list) -> tuple[list, dict]:
    """Per-cell pools from ``ends_by_cell[i]`` ({chrom: (pos, p)}) grouped
    by ``strata[i]``.  Returns (the pool for each cell, the chromosome
    codes used for genome coordinates)."""
    chrom_code = {c: i for i, c in enumerate(sorted({c for e in ends_by_cell
                                                     for c in e}))}
    grouped: dict = {}
    for i, e in enumerate(ends_by_cell):
        if e:
            grouped.setdefault(strata[i], {})[i] = (
                np.concatenate([chrom_code[c] * GENOME_STRIDE + e[c][0]
                                for c in e]),
                np.concatenate([e[c][1] for c in e]))
    pools = {k: build_pool(v) for k, v in grouped.items()}
    empty = build_pool({})
    return [pools.get(strata[i], empty) for i in range(len(ends_by_cell))], \
        chrom_code
