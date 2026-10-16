"""Link-shuffle expected and null: a contact keeps its left end (the anchor)
and its partner is replaced by long-range read ends from OTHER cells of the
same stratum.

A donor is an end of a read pair at least ``min_distance`` long -- the kind
of end the partner it stands in for is -- on the partner's side of the
anchor (to its right), at a distance in the contact's own log-distance bin,
and optionally inside given bounds (step 5: the anchor's bin, where the real
partner is).  Where the donor's own partner sits does not matter, and a
donor from another cell can never re-form a read pair.

THE EXPECTED.  For a contact with anchor posterior a, the expected co-state
is that of the anchor with the AVERAGE donor in its window, m:
co-open a*m, co-closed (1-a)(1-m), mixed the rest.  So the coordination
O - E of a contact is (2a - 1)(p - m): whether the real partner p agrees
with the anchor more than the partners that sit where it sits, at its
distance, in cells of its kind, do.  Ends of the same kind, matched on
place and distance -- unlike the posteriors of short read pairs.

THE NULL is one donor per contact per draw, scored against the same E.

WHICH STATISTIC.  This E holds the contact's own anchor, so only the
agreement O - E (co-open + co-closed) reads coordination.  An open-vs-closed
contrast cancels the anchor: for one contact log(O_open/E_open) -
log(O_closed/E_closed) is logit(p) - logit(m), the partner against m and
nothing about the anchor -- planted coordination vanished from it on
synthetic cells, while C caught it every time.

WHY TWO HALVES, CROSSED.  E and the null must not share donors: a contact
with one candidate would get E = that donor and a null of exactly zero, and
in general a shared donor makes the null agree with E more than chance, so
the null comes out too narrow.  So each stratum's cells are split once into
two halves, and a contact's E averages donors of the OTHER half while its
null draws from its OWN half (never its own cell).  The crossing is what
makes the null a null: a real partner is an end of the contact's half, so
it also sits in the other half's E -- it enters its own cell's O with a
plus and other cells' E with a minus.  A null donor from the contact's own
half does exactly the same.  With one half lending to every E and the other
to every null, real partners of the E half were in E and null donors never
were; on synthetic cells with no coordination that coupling alone made the
null more separable than the real (random forest AUC 0.82-0.86 vs
0.74-0.76).  A contact enters O, E and the null only if both halves have a
donor for it -- so the three are always built on the same contacts.

WHY A WHOLE STRATUM AT ONCE.  A real partner belongs to one cell.  A null
donor handed to several cells in the same draw would make their null values
move together -- noise shared within a cell type, which a classifier or a
cell graph reads as cell-type structure the real data does not have.  So
all of a stratum's contacts are assigned together, and within a draw a
donor goes to one contact only; a contact whose window has no free donor
left borrows a taken one, and how often is reported.  (A precaution: on its
own it did not fix the over-separable null above -- crossing the halves
did.)

WHY OTHER CELLS.  Within one cell, at single-cell depth, a contact has one
candidate or none (on synthetic cells with six contacts per Mb, 42% of those
with any had exactly one), so every draw repeats it and the null is far too
narrow.  Pooled over a stratum's cells the candidates are many.

A contact whose own distance bin lacks a donor in either half may borrow
from up to ``max_widen`` neighbouring bins on each side; how often is
reported too, so a run can say how much of it is matched exactly.
"""

from __future__ import annotations

import zlib

import numpy as np

#: Genome coordinate = chromosome code * stride + position.
GENOME_STRIDE = np.int64(1) << np.int64(33)


def log_dbins(min_d: int, max_d: int, per_decade: int = 4) -> np.ndarray:
    """Log-spaced distance-bin edges across [min_d, max_d]."""
    lo, hi = np.log10(max(min_d, 1)), np.log10(max(max_d, min_d + 1))
    n = max(1, int(np.ceil((hi - lo) * per_decade)))
    return np.logspace(lo, hi, n + 1)


def read_cell(posteriors_path, min_distance: int, min_sites: int,
              max_distance: int = 0, bin_size: int = 0) -> dict:
    """One cell's donor ends and contacts.

    ``ends``: {chrom: (positions, posteriors)} -- every end with a call
    (``n >= min_sites``) of a read pair at least ``min_distance`` long.
    ``contacts``: {chrom: (left ends, lengths, rows)} -- the read pairs with
    both ends called, at least ``min_distance`` long, at most
    ``max_distance`` (if set), with both ends in one ``bin_size`` bin (if
    set).  ``rows`` are their rows in the file: how the caller, reading it
    again, finds each contact's assignment."""
    import pandas as pd
    df = pd.read_parquet(posteriors_path,
                         columns=["chrom", "pos1", "pos2", "n1", "n2",
                                  "posterior1", "posterior2"])
    chrom = df["chrom"].astype(str).to_numpy()
    pos1 = df["pos1"].to_numpy(np.int64)
    pos2 = df["pos2"].to_numpy(np.int64)
    p1 = df["posterior1"].to_numpy(np.float64, na_value=np.nan)
    p2 = df["posterior2"].to_numpy(np.float64, na_value=np.nan)
    sep = np.abs(pos2 - pos1)
    lr = sep >= int(min_distance)
    c1 = (df["n1"].to_numpy() >= min_sites) & np.isfinite(p1)
    c2 = (df["n2"].to_numpy() >= min_sites) & np.isfinite(p2)
    m1, m2 = lr & c1, lr & c2
    ch = np.concatenate([chrom[m1], chrom[m2]])
    pos = np.concatenate([pos1[m1], pos2[m2]])
    pp = np.concatenate([p1[m1], p2[m2]])
    con = lr & c1 & c2
    if max_distance:
        con &= sep <= int(max_distance)
    if bin_size:
        con &= (pos1 // int(bin_size)) == (pos2 // int(bin_size))
    row = np.flatnonzero(con)
    cch, lo, d = chrom[row], np.minimum(pos1, pos2)[row], sep[row]
    return {"ends": {c: (pos[ch == c], pp[ch == c]) for c in np.unique(ch)},
            "contacts": {c: (lo[cch == c], d[cch == c], row[cch == c])
                         for c in np.unique(cch)}}


def build_pool(ends_by_cell: dict):
    """A donor pool from ``{cell_id: (g, p)}``: genome coordinates sorted,
    their posteriors, the cell each came from, and the running sum of the
    posteriors (so a window's mean is two lookups)."""
    if not ends_by_cell:
        return (np.empty(0, np.int64), np.empty(0), np.empty(0, np.int64),
                np.zeros(1))
    g = np.concatenate([np.asarray(e[0], np.int64)
                        for e in ends_by_cell.values()])
    p = np.concatenate([np.asarray(e[1], float) for e in ends_by_cell.values()])
    cell = np.concatenate([np.full(len(e[0]), cid, np.int64)
                           for cid, e in ends_by_cell.items()])
    order = np.argsort(g, kind="mergesort")
    p = p[order]
    return g[order], p, cell[order], np.concatenate([[0.0], np.cumsum(p)])


def split_cells(cells, seed: int = 0) -> dict:
    """{cell_id: 0 or 1}: a stratum's cells split once into two halves --
    0 lends to E, 1 to the null.  Stable for a given seed."""
    cells = sorted(cells)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cells))
    return {cells[j]: int(k >= (len(cells) + 1) // 2)
            for k, j in enumerate(order)}


def _by_cell(pool, shift: int):
    """The pool keyed by (cell, coordinate), sorted, with the running sum of
    its posteriors: a cell's OWN ends in a window are two lookups."""
    g, p, cell, _ = pool
    key = (cell << np.int64(shift)) + g
    o = np.argsort(key, kind="mergesort")
    return key[o], np.concatenate([[0.0], np.cumsum(p[o])])


def assign_stratum(anchor_g, d, anchor_cell, pool_e, pool_n, dbins, rng,
                   n_draws: int = 1, bounds=None, max_widen: int = 2,
                   max_rounds: int = 64):
    """E and null donors for every contact of one stratum.

    ``anchor_g``: the contacts' left ends as genome coordinates; ``d``:
    their lengths; ``anchor_cell``: the cell each is from.  ``pool_e`` and
    ``pool_n``: the E half's and the null half's donor ends
    (:func:`build_pool`).  ``bounds``: optional (lo, hi) per contact a donor
    must lie in.  A cell's own ends are never its donors.

    Per contact, its window is the narrowest -- its own distance bin,
    widened by up to ``max_widen`` bins on each side -- holding a donor in
    both halves; ``m`` is the mean posterior of the E half's donors in it.
    Then, per draw, each contact gets one null-half donor from its window,
    and NO DONOR GOES TO TWO CONTACTS: contacts claim donors in a random
    order, each a free one at random.  One whose window has none left
    borrows a taken one.  (The caller weights by the anchor's own
    posterior: E co-open is a*m.)

    Returns (level, m, donor_p, n_reused): ``level`` the distance bins the
    window was widened by (-1: no donor in one of the halves -- the contact
    is left out of O, E and the null alike), ``donor_p`` n_draws x contacts,
    ``n_reused`` how many null donors were borrowed, summed over draws.
    """
    anchor_g = np.asarray(anchor_g, np.int64)
    anchor_cell = np.asarray(anchor_cell, np.int64)
    n = anchor_g.size
    n_draws = max(1, int(n_draws))
    level = np.full(n, -1, np.int64)
    m = np.full(n, np.nan)
    donor_p = np.full((n_draws, n), np.nan)
    ge, pe, ce, cume = pool_e
    gn, pn, cn, _ = pool_n
    if n == 0 or ge.size == 0 or gn.size == 0:
        return level, m, donor_p, 0
    nb = len(dbins) - 1
    k = np.clip(np.searchsorted(dbins, np.asarray(d), side="right") - 1,
                0, nb - 1)
    # (cell, coordinate) keys for counting a contact's own cell's ends
    top = max(int(ge.max()), int(gn.max()),
              int(anchor_g.max()) + int(np.ceil(dbins[-1]))) + 1
    shift = top.bit_length()
    if (max(int(ce.max()), int(cn.max()), int(anchor_cell.max())) + 1
            >= (1 << (62 - shift))):
        raise ValueError("too many cells for (cell, coordinate) keys")
    same = pool_n is pool_e            # one pool for both: search it once
    own_ke, own_cume = _by_cell(pool_e, shift)
    own_kn = own_ke if same else _by_cell(pool_n, shift)[0]
    base = anchor_cell << np.int64(shift)
    win = np.zeros((n, 2), np.int64)          # the null half's window
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
        hi = np.maximum(np.ceil(hi).astype(np.int64), lo)
        el = np.searchsorted(ge, lo, side="left")
        eh = np.searchsorted(ge, hi, side="left")
        b = base[todo]
        ol = np.searchsorted(own_ke, b + lo, side="left")
        oh = np.searchsorted(own_ke, b + hi, side="left")
        if same:
            nl, nh, own_n = el, eh, oh - ol
        else:
            nl = np.searchsorted(gn, lo, side="left")
            nh = np.searchsorted(gn, hi, side="left")
            own_n = (np.searchsorted(own_kn, b + hi, side="left")
                     - np.searchsorted(own_kn, b + lo, side="left"))
        cnt_e = eh - el - (oh - ol)
        ok = (cnt_e >= 1) & ((nh - nl - own_n) >= 1)
        hit = todo[ok]
        level[hit] = w
        win[hit] = np.stack([nl, nh], 1)[ok]
        m[hit] = ((cume[eh] - cume[el] - (own_cume[oh] - own_cume[ol]))[ok]
                  / cnt_e[ok])
    has = np.flatnonzero(level >= 0)
    L, H = win[:, 0], win[:, 1]
    n_reused = 0
    for dr in range(n_draws):
        used = np.zeros(gn.size, dtype=bool)
        got = np.full(n, -1, np.int64)
        prio = rng.random(n)                  # who wins a contested donor
        todo = has
        for _ in range(int(max_rounds)):
            if todo.size == 0:
                break
            # a free donor at random from each window: the r-th free one
            free_cum = np.concatenate([[0], np.cumsum(~used)])
            n_free = free_cum[H[todo]] - free_cum[L[todo]]
            todo, n_free = todo[n_free > 0], n_free[n_free > 0]
            if todo.size == 0:
                break
            r = np.minimum((rng.random(todo.size) * n_free).astype(np.int64),
                           n_free - 1)
            j = np.searchsorted(free_cum, free_cum[L[todo]] + r + 1,
                                side="left") - 1
            ok = cn[j] != anchor_cell[todo]
            cand, who = j[ok], todo[ok]
            o = np.lexsort((-prio[who], cand))
            cand, who = cand[o], who[o]
            first = np.ones(cand.size, dtype=bool)
            first[1:] = cand[1:] != cand[:-1]
            got[who[first]] = cand[first]
            used[cand[first]] = True
            todo = todo[got[todo] < 0]
        # No free donor left in the window (but perhaps the cell's own):
        # borrow a taken one.
        rest = has[got[has] < 0]
        n_reused += int(rest.size)
        for _ in range(int(max_rounds)):
            if rest.size == 0:
                break
            j = L[rest] + (rng.random(rest.size)
                           * (H[rest] - L[rest])).astype(np.int64)
            ok = cn[j] != anchor_cell[rest]
            got[rest[ok]] = j[ok]
            rest = rest[~ok]
        for i in rest:              # windows mostly of the cell's own ends
            cand = np.arange(L[i], H[i])
            got[i] = rng.choice(cand[cn[cand] != anchor_cell[i]])
        donor_p[dr, has] = pn[got[has]]
    return level, m, donor_p, n_reused


def assign_same_cell(anchor_g, d, anchor_cell, anchor_id, pool, pool_id,
                     dbins, rng, n_draws: int = 1, bounds=None,
                     max_widen: int = 0, max_rounds: int = 16,
                     pool_mate=None, partner_g=None, exclude: int = 0):
    """Null donors from each contact's OWN cell, as
    end_posterior_distance_correlation.py's link-shuffle under --null-scope
    cell: an end of another of the cell's read pairs -- never the contact's
    own -- in its window (its own distance bin, widened by up to
    ``max_widen`` bins on each side, inside ``bounds``), at random and
    independently per draw.  Same nucleus, same place: only the ligation
    differs, so a state the cell holds over a whole region, shared by both
    ends of a contact, cancels.

    A donor may serve several of the cell's contacts: within one cell the
    candidates are few, and reserving them would leave most contacts
    without one.  The same few also make a contact's two draws often the
    same donor, so the control (draw 2 less draw 1) is narrower than the
    null's own noise -- the WHY OTHER CELLS above, which this mode accepts
    for the sake of the shared state.

    ``exclude`` W (with ``pool_mate``, each donor end's read pair's other
    end, and ``partner_g``, each contact's real partner): an end within W
    of the real partner -- likely its own molecule -- or whose read pair's
    other end lies within W of the anchor -- ligated to the anchor's locus:
    a near-copy of the contact (a third of a cortex cell's contacts >= 1 kb
    have one) or another contact of the anchor -- is refused too, and a
    contact left with none drops out.

    ``pool``: (coordinates, posteriors, cells, _) of the donor ends, in any
    order; ``pool_id``: the read pair each belongs to; ``anchor_id``: each
    contact's own.  Returns (level, m, donor_p, n_reused) as
    :func:`assign_stratum`: ``m`` the window's mean posterior, the contact's
    own ends aside (refusals by ``exclude`` not taken out); ``n_reused``
    0."""
    anchor_g = np.asarray(anchor_g, np.int64)
    anchor_cell = np.asarray(anchor_cell, np.int64)
    anchor_id = np.asarray(anchor_id, np.int64)
    n = anchor_g.size
    n_draws = max(1, int(n_draws))
    level = np.full(n, -1, np.int64)
    m = np.full(n, np.nan)
    donor_p = np.full((n_draws, n), np.nan)
    g = np.asarray(pool[0], np.int64)
    if n == 0 or g.size == 0:
        return level, m, donor_p, 0
    cell = np.asarray(pool[2], np.int64)
    nb = len(dbins) - 1
    k = np.clip(np.searchsorted(dbins, np.asarray(d), side="right") - 1,
                0, nb - 1)
    top = max(int(g.max()), int(anchor_g.max()) + int(np.ceil(dbins[-1]))) + 1
    shift = top.bit_length()
    if max(int(cell.max()), int(anchor_cell.max())) + 1 >= (1 << (62 - shift)):
        raise ValueError("too many cells for (cell, coordinate) keys")
    # the pool by (cell, coordinate): a cell's own ends in a window are two
    # lookups
    key = (cell << np.int64(shift)) + g
    o = np.argsort(key, kind="mergesort")
    key = key[o]
    gs = g[o]
    ps = np.asarray(pool[1], float)[o]
    ids = np.asarray(pool_id, np.int64)[o]
    cum = np.concatenate([[0.0], np.cumsum(ps)])
    W = int(exclude or 0)
    if W and pool_mate is not None and partner_g is not None:
        mates = np.asarray(pool_mate, np.int64)[o]
        partner_g = np.asarray(partner_g, np.int64)
    else:
        W = 0

    def valid(j, i):
        ok = ids[j] != anchor_id[i]
        if W:
            ok &= ((np.abs(gs[j] - partner_g[i]) > W)
                   & (np.abs(mates[j] - anchor_g[i]) > W))
        return ok
    # where each contact's own ends sit in that order: two at most, its
    # left and its right
    oi = np.argsort(ids, kind="mergesort")
    s0 = np.searchsorted(ids[oi], anchor_id, side="left")
    s1 = np.searchsorted(ids[oi], anchor_id, side="right")
    last = oi.size - 1
    own = np.stack([np.where(s1 > s0, oi[np.minimum(s0, last)], -1),
                    np.where(s1 > s0 + 1, oi[np.minimum(s0 + 1, last)], -1)])
    base = anchor_cell << np.int64(shift)
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
        hi = np.maximum(np.ceil(hi).astype(np.int64), lo)
        b = base[todo]
        lw = np.searchsorted(key, b + lo, side="left")
        hw = np.searchsorted(key, b + hi, side="left")
        inside = (own[:, todo] >= lw) & (own[:, todo] < hw)
        n_own = inside.sum(0)
        s_own = np.where(inside, ps[np.maximum(own[:, todo], 0)], 0.0).sum(0)
        free = hw - lw - n_own
        ok = free >= 1
        hit = todo[ok]
        level[hit] = w
        L[hit], H[hit] = lw[ok], hw[ok]
        m[hit] = (cum[hw] - cum[lw] - s_own)[ok] / free[ok]
    if W:
        # windows wholly within W of the real partner hold no stand-in
        a_ = np.flatnonzero(level >= 0)
        near = ((gs[L[a_]] >= partner_g[a_] - W)
                & (gs[H[a_] - 1] <= partner_g[a_] + W))
        level[a_[near]] = -1
        m[a_[near]] = np.nan
    has = np.flatnonzero(level >= 0)
    for dr in range(n_draws):
        got = np.full(n, -1, np.int64)
        todo = has
        for _ in range(int(max_rounds)):
            if todo.size == 0:
                break
            j = L[todo] + (rng.random(todo.size)
                           * (H[todo] - L[todo])).astype(np.int64)
            ok = valid(j, todo)
            got[todo[ok]] = j[ok]
            todo = todo[~ok]
        for i in todo:              # windows mostly of refused ends
            cand = np.arange(L[i], H[i])
            cand = cand[valid(cand, np.full(cand.size, i))]
            if cand.size:
                got[i] = rng.choice(cand)
            else:                   # none it may take: out, every draw
                level[i] = -1
                m[i] = np.nan
        has = has[level[has] >= 0]
        donor_p[dr, has] = ps[got[has]]
    return level, m, donor_p, 0


def _genome(parts: dict, code: dict):
    """{chrom: arrays} -> (genome coordinates, the other arrays...)."""
    if not parts:
        return None
    first = [code[c] * GENOME_STRIDE + np.asarray(v[0], np.int64)
             for c, v in parts.items()]
    rest = [np.concatenate([np.asarray(v[j]) for v in parts.values()])
            for j in range(1, len(next(iter(parts.values()))))]
    return (np.concatenate(first), *rest)


def assign_all(cells: list, strata: list, dbins, n_draws: int = 1,
               seed: int = 0, max_widen: int = 2, bin_size: int = 0):
    """E and null donors for every contact of every cell, a stratum at a
    time (:func:`assign_stratum`).

    ``cells[i]``: :func:`read_cell` of cell i; ``strata[i]``: its stratum.
    Each stratum's cells are split once into two halves (:func:`split_cells`);
    a contact's E comes from the other half, its null from its own.
    ``bin_size``: keep each donor in its contact's bin.

    Returns (per cell {"row", "level", "m", "donor_p"} with ``row`` sorted
    and the rest aligned to it, ``donor_p`` n_draws x contacts; a summary
    {"contacts", "own_bin", "widened", "none", "reused", "draws",
    "strata"} -- ``reused`` counts borrowed null donors over all draws).
    """
    n_draws = max(1, int(n_draws))
    chroms = sorted({c for x in cells for part in (x["ends"], x["contacts"])
                     for c in part})
    code = {c: i for i, c in enumerate(chroms)}
    empty = {"row": np.empty(0, np.int64), "level": np.empty(0, np.int64),
             "m": np.empty(0), "donor_p": np.empty((n_draws, 0))}
    out = [empty] * len(cells)
    summary = {"contacts": 0, "own_bin": 0, "widened": 0, "none": 0,
               "reused": 0, "draws": n_draws, "strata": 0}
    groups: dict = {}
    for i, s in enumerate(strata):
        groups.setdefault(s, []).append(i)
    for s, members in groups.items():
        summary["strata"] += 1
        key = zlib.crc32(str(s).encode())
        half = split_cells(members, seed=int(seed) + key)
        ends = {i: _genome(cells[i]["ends"], code) for i in members}
        pools = [build_pool({i: e for i, e in ends.items()
                             if e is not None and half[i] == h})
                 for h in (0, 1)]
        for h in (0, 1):
            con = {i: _genome(cells[i]["contacts"], code) for i in members
                   if half[i] == h}
            con = {i: c for i, c in con.items()
                   if c is not None and c[0].size}
            if not con:
                continue
            ag = np.concatenate([c[0] for c in con.values()])
            ad = np.concatenate([c[1] for c in con.values()])
            rows = np.concatenate([c[2] for c in con.values()])
            ac = np.concatenate([np.full(c[0].size, i, np.int64)
                                 for i, c in con.items()])
            bounds = None
            if bin_size:
                bs = int(bin_size)
                start = ((ag // GENOME_STRIDE) * GENOME_STRIDE
                         + ((ag % GENOME_STRIDE) // bs) * bs)
                bounds = (start, start + bs)
            # E from the other half, the null from this one
            level, m, donor_p, reused = assign_stratum(
                ag, ad, ac, pools[1 - h], pools[h], dbins,
                np.random.default_rng([int(seed), key, h, 31_337]),
                n_draws=n_draws, bounds=bounds, max_widen=max_widen)
            summary["contacts"] += int(ag.size)
            summary["own_bin"] += int((level == 0).sum())
            summary["widened"] += int((level > 0).sum())
            summary["none"] += int((level < 0).sum())
            summary["reused"] += int(reused)
            for i in con:
                sel = np.flatnonzero(ac == i)
                sel = sel[np.argsort(rows[sel], kind="mergesort")]
                out[i] = {"row": rows[sel], "level": level[sel], "m": m[sel],
                          "donor_p": donor_p[:, sel]}
    return out, summary


def lookup(link: dict, rows, n_draws: int):
    """A cell's assignment (:func:`assign_all`) at the given file rows:
    (level, m, donor_p); rows it does not hold get level -1."""
    rows = np.asarray(rows, np.int64)
    n = rows.size
    level = np.full(n, -1, np.int64)
    m = np.full(n, np.nan)
    donor_p = np.full((max(1, int(n_draws)), n), np.nan)
    have = link["row"]
    if have.size and n:
        j = np.minimum(np.searchsorted(have, rows), have.size - 1)
        f = have[j] == rows
        level[f], m[f] = link["level"][j[f]], link["m"][j[f]]
        donor_p[:, f] = link["donor_p"][:, j[f]]
    return level, m, donor_p
