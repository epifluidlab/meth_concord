"""Joint GCH+HCG multi-state long-range coordination — per-cell core (J1/J2).

The *testable* numeric core behind ``scripts/joint_coordination_existence.py``.
Joins a cell's GCH (accessibility) and HCG (methylation) posterior context runs
per the design §7 prerequisite, builds the per-end soft 4-state distribution
(§2, via :mod:`nome_hic_concordance.joint_state`), and forms a distance-matched
**log2(O/E)** concordance for each co-state channel, pooled per (region) within
the E–P band — exactly the leakage-free estimator validated for the
single-context existence test, generalized to the joint multi-state grid.

For a controlled comparison it computes, on the **same** lenient end pool and
the **same** distance-matched null, three feature sets per region:

* **joint** — four channels ``Δ_s = log2(O_s/E_s)`` for
  s ∈ {open, closed, poised, active_methylated}, ``co_s = q_i[s]·q_j[s]`` (§3–4);
* **gch** — the single-context accessibility signed contrast
  ``[log2(O_open/E_open) − log2(O_closed/E_closed)]`` with co-open ``= a_i·a_j``;
* **hcg** — the single-context methylation signed contrast with
  co-methylated ``= m_i·m_j``.

So "does the joint multi-state beat the best single context?" is asked
apples-to-apples (same cells, same ends, same null, same CV folds downstream).
Absolute single-context numbers should still come from
``ep_coordination_existence.py``; the gch/hcg here are *references on the joint
pool*.

Per-region output is a flat 13-vector (see :data:`RVEC`) so assembling the
cells × features matrices downstream is a simple gather.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .joint_state import impute_missing_modality, per_end_joint_state

_MULT = 100_000_000          # region_id = chrom_code * _MULT + bin_index

#: Joint co-state channel order (the four ``Δ_s`` features).
JOINT_CHANNELS = ("open", "closed", "poised", "active_methylated")

# Per-end value layout for the 8 PRODUCT channels (pairwise product → co-value):
#   0 joint open    = q_open       4 gch open    = a
#   1 joint closed  = q_closed     5 gch closed  = 1 − a
#   2 joint poised  = q_poised     6 hcg meth    = m
#   3 joint am      = q_am         7 hcg unmeth  = 1 − m
_N_PROD = 8
# Per-end value layout for the 6 MARGINAL channels (region mean → marginal):
#   0..3 mean q_open/closed/poised/am   4 mean a   5 mean m
_N_MARG = 6

#: Layout of the per-(cell, region) 13-vector returned by :func:`process_cell`.
#:   [0:4]  joint Δ (open, closed, poised, active_methylated)
#:   [4:8]  joint per-channel marginal (mean q_s)
#:   [8]    gch signed Δ        [9]  gch marginal (mean accessibility a)
#:   [10]   hcg signed Δ        [11] hcg marginal (mean methylation m)
#:   [12]   n_o  (contact-pair count in the region — coverage)
RVEC = 13
_J_DELTA, _J_MARG = slice(0, 4), slice(4, 8)
_G_DELTA, _G_MARG, _H_DELTA, _H_MARG, _N_O = 8, 9, 10, 11, 12

_EPS = 1e-6


# --------------------------------------------------------------------------
# Vectorized band end-pairing (shared shape with the single-context test)
# --------------------------------------------------------------------------

def _ranges_concat(lo: np.ndarray, hi: np.ndarray):
    """Vectorized concat of half-open ranges [lo[i], hi[i]) → (left_rep, right)."""
    counts = (hi - lo).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    n = lo.shape[0]
    left_rep = np.repeat(np.arange(n), counts)
    starts = np.cumsum(counts) - counts
    within = np.arange(total) - np.repeat(starts, counts)
    right = np.repeat(lo, counts) + within
    return left_rep, right


def _window_pairs_multi(pos: np.ndarray, vals: np.ndarray, min_d: int, max_d: int,
                        cap: int, rng):
    """All same-chrom end-pairs with separation in [min_d, max_d] (vectorized).

    ``vals`` is ``(n_ends, K)`` per-end channel values; returns
    ``(left_pos, prod, dist)`` where ``prod = vals[li] * vals[ri]`` is
    ``(n_pairs, K)`` (the co-value per channel) and ``left_pos`` = the smaller
    position (for region assignment).  Subsamples ends to ``cap`` first."""
    K = vals.shape[1]
    empty = (np.empty(0, np.int64), np.empty((0, K)), np.empty(0, np.int64))
    n = pos.shape[0]
    if n < 2:
        return empty
    if n > cap:
        idx = np.sort(rng.choice(n, size=cap, replace=False))
        pos, vals = pos[idx], vals[idx]
        n = cap
    order = np.argsort(pos, kind="mergesort")
    pos, vals = pos[order], vals[order]
    lo = np.searchsorted(pos, pos + int(min_d), side="left")
    hi = np.searchsorted(pos, pos + int(max_d), side="right")
    li, ri = _ranges_concat(lo, hi)
    if li.size == 0:
        return empty
    return pos[li], vals[li] * vals[ri], pos[ri] - pos[li]


def _region_ids(codes: np.ndarray, pos: np.ndarray, gran):
    if gran == "chrom":
        return codes.astype(np.int64) * _MULT
    size = int(gran) * 1_000_000
    return codes.astype(np.int64) * _MULT + (pos // size).astype(np.int64)


def _agg_multi(ids: np.ndarray, vals: np.ndarray):
    """(unique_ids, summed (R×K), counts) via per-column bincount — vectorized."""
    K = vals.shape[1]
    if ids.size == 0:
        return np.empty(0, np.int64), np.empty((0, K)), np.empty(0, np.int64)
    uniq, inv = np.unique(ids, return_inverse=True)
    R = uniq.shape[0]
    sums = np.stack([np.bincount(inv, weights=vals[:, k], minlength=R)
                     for k in range(K)], axis=1)
    cnt = np.bincount(inv, minlength=R)
    return uniq, sums, cnt


# --------------------------------------------------------------------------
# Global priors (leak-safe lenient imputation, locked decision #3)
# --------------------------------------------------------------------------

def _cell_obs_sums(path: Path, min_sites: int):
    """(sum_observed_posterior, n_observed) over both ends of one context run."""
    df = pd.read_parquet(path, columns=["n1", "n2", "posterior1", "posterior2"])
    if len(df) == 0:
        return 0.0, 0
    s = c = 0.0
    for nc, pc in (("n1", "posterior1"), ("n2", "posterior2")):
        n = df[nc].to_numpy(np.int32)
        p = df[pc].to_numpy(np.float64, na_value=np.nan)
        ok = (n >= min_sites) & np.isfinite(p)
        s += float(p[ok].sum()); c += int(ok.sum())
    return s, c


def scan_priors(paths: list, min_sites: int, n_threads: int = 1) -> float:
    """Global marginal rate = mean observed posterior over a set of context runs.

    The genome-wide rate ``predict_proba(0, 0)`` approximates, used as the
    leak-safe shared prior for the missing modality (one number across cells →
    prior-imputed ends inject no cell-type structure)."""
    if n_threads > 1 and len(paths) > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_threads)(
            delayed(_cell_obs_sums)(p, min_sites) for p in paths)
    else:
        res = [_cell_obs_sums(p, min_sites) for p in paths]
    s = sum(r[0] for r in res); c = sum(r[1] for r in res)
    return float(s / c) if c > 0 else 0.5


# --------------------------------------------------------------------------
# Per-cell processing
# --------------------------------------------------------------------------

def _load_ctx(path: Path):
    df = pd.read_parquet(path, columns=["chrom", "pos1", "pos2", "n1", "n2",
                                        "posterior1", "posterior2"])
    df["chrom"] = df["chrom"].astype(str)
    return df


def process_cell(gpath, hpath, granularities, min_d, max_d, min_gpc, min_cpg,
                 a_prior, m_prior, cap, seed):
    """Per-(region) joint/gch/hcg concordance for one cell, at each granularity.

    Returns ``(cell, {gran: {region_key: vec13}}, qc)`` where ``vec13`` follows
    :data:`RVEC` and ``qc`` is a dict of pair/end observation fractions (§2.1).
    """
    gpath, hpath = Path(gpath), Path(hpath)
    cell = gpath.name
    for suf in (".posteriors.parquet", ".parquet"):
        if cell.endswith(suf):
            cell = cell[: -len(suf)]
            break
    g = _load_ctx(gpath).rename(columns={"n1": "n1g", "n2": "n2g",
                                         "posterior1": "a1", "posterior2": "a2"})
    h = _load_ctx(hpath).rename(columns={"n1": "n1h", "n2": "n2h",
                                         "posterior1": "m1", "posterior2": "m2"})
    df = g.merge(h, on=["chrom", "pos1", "pos2"], how="outer")
    for c in ("n1g", "n2g", "n1h", "n2h"):
        df[c] = df[c].fillna(0)
    if len(df) == 0:
        return cell, {}, {}

    chrom = df["chrom"].astype(str).values
    pos1 = df["pos1"].to_numpy(np.int64); pos2 = df["pos2"].to_numpy(np.int64)
    n1g = df["n1g"].to_numpy(np.int32); n2g = df["n2g"].to_numpy(np.int32)
    n1h = df["n1h"].to_numpy(np.int32); n2h = df["n2h"].to_numpy(np.int32)
    a1 = df["a1"].to_numpy(np.float64, na_value=np.nan)
    a2 = df["a2"].to_numpy(np.float64, na_value=np.nan)
    m1 = df["m1"].to_numpy(np.float64, na_value=np.nan)
    m2 = df["m2"].to_numpy(np.float64, na_value=np.nan)

    # Lenient missing-modality marginalization (§2.1) per end.
    a1e, m1e, a1o, m1o = impute_missing_modality(a1, n1g, m1, n1h, min_gpc,
                                                 min_cpg, a_prior, m_prior)
    a2e, m2e, a2o, m2o = impute_missing_modality(a2, n2g, m2, n2h, min_gpc,
                                                 min_cpg, a_prior, m_prior)
    info1 = a1o | m1o          # informative end = >=1 modality observed (lenient)
    info2 = a2o | m2o

    q1 = per_end_joint_state(a1e, m1e)
    q2 = per_end_joint_state(a2e, m2e)

    def _prodvec(q, ae, me):           # (n, 8) per-end value for the co-products
        return np.column_stack([q[0], q[1], q[2], q[3],
                                ae, 1.0 - ae, me, 1.0 - me])

    def _margvec(q, ae, me):           # (n, 6) per-end value for region means
        return np.column_stack([q[0], q[1], q[2], q[3], ae, me])

    v1, v2 = _prodvec(q1, a1e, m1e), _prodvec(q2, a2e, m2e)
    mv1, mv2 = _margvec(q1, a1e, m1e), _margvec(q2, a2e, m2e)

    # End pool (null + marginal): every informative end.
    e_chrom = np.concatenate([chrom[info1], chrom[info2]])
    e_pos = np.concatenate([pos1[info1], pos2[info2]])
    e_vals = np.vstack([v1[info1], v2[info2]])      # (n_ends, 8)
    e_mv = np.vstack([mv1[info1], mv2[info2]])      # (n_ends, 6)
    if e_pos.size == 0:
        return cell, {}, {}
    uniq_chroms = np.unique(e_chrom)
    code_of = lambda arr: np.searchsorted(uniq_chroms, arr).astype(np.int64)

    # OBSERVED contacts in band.
    lo = np.minimum(pos1, pos2); hi = np.maximum(pos1, pos2)
    sep = hi - lo
    cm = info1 & info2 & (sep >= min_d) & (sep <= max_d)
    o_codes = code_of(chrom[cm]); o_pos = lo[cm]
    o_prod = v1[cm] * v2[cm]                         # (n_contacts, 8)

    # QC (§2.1): fully- vs partially-observed contact-pair fractions.
    both1 = a1o & m1o; both2 = a2o & m2o
    n_band = int(cm.sum())
    qc = {"n_contacts_band": n_band,
          "frac_pairs_full": float((both1 & both2)[cm].mean()) if n_band else np.nan,
          "frac_ends_both": float(np.mean(np.concatenate([both1[info1], both2[info2]])))
          if e_pos.size else np.nan,
          "frac_ends_gch_only": float(np.mean(np.concatenate(
              [(a1o & ~m1o)[info1], (a2o & ~m2o)[info2]]))) if e_pos.size else np.nan,
          "frac_ends_hcg_only": float(np.mean(np.concatenate(
              [(~a1o & m1o)[info1], (~a2o & m2o)[info2]]))) if e_pos.size else np.nan}

    # Distance-matched NULL end-pairs per chromosome (windowed, vectorized).
    rng = np.random.default_rng(seed)
    ep_pos_l, ep_prod_l, ep_code_l = [], [], []
    for ci, c in enumerate(uniq_chroms):
        sel = e_chrom == c
        lp, pr, _ = _window_pairs_multi(e_pos[sel], e_vals[sel], min_d, max_d,
                                        cap, rng)
        if lp.size:
            ep_pos_l.append(lp); ep_prod_l.append(pr)
            ep_code_l.append(np.full(lp.size, ci, np.int64))
    if ep_pos_l:
        e_lpos = np.concatenate(ep_pos_l)
        e_prod = np.vstack(ep_prod_l)
        e_codes = np.concatenate(ep_code_l)
    else:
        e_lpos = np.empty(0, np.int64); e_prod = np.empty((0, _N_PROD))
        e_codes = np.empty(0, np.int64)
    m_codes = code_of(e_chrom)

    def _l2(x):
        return np.log2(np.clip(x, _EPS, None))

    out: dict = {}
    for gran in granularities:
        gk = str(gran)
        ro = _region_ids(o_codes, o_pos, gran)
        re = _region_ids(e_codes, e_lpos, gran)
        rm = _region_ids(m_codes, e_pos, gran)
        ids_o, o_sum, no = _agg_multi(ro, o_prod)
        ids_e, e_sum, ne = _agg_multi(re, e_prod)
        ids_m, mv_sum, nm = _agg_multi(rm, e_mv)
        all_ids = np.unique(np.concatenate([ids_o, ids_e, ids_m]))
        R = all_ids.shape[0]
        if R == 0:
            out[gk] = {}
            continue

        def place(ids, vals, width):
            arr = np.full((R, width), np.nan)
            if ids.size:
                arr[np.searchsorted(all_ids, ids)] = vals
            return arr

        n_o = np.zeros(R);
        if ids_o.size:
            n_o[np.searchsorted(all_ids, ids_o)] = no
        n_e = np.zeros(R)
        if ids_e.size:
            n_e[np.searchsorted(all_ids, ids_e)] = ne
        n_m = np.zeros(R)
        if ids_m.size:
            n_m[np.searchsorted(all_ids, ids_m)] = nm

        O = place(ids_o, o_sum, _N_PROD)
        E = place(ids_e, e_sum, _N_PROD)
        MV = place(ids_m, mv_sum, _N_MARG)
        with np.errstate(divide="ignore", invalid="ignore"):
            O = O / n_o[:, None]; E = E / n_e[:, None]; MV = MV / n_m[:, None]
            logO, logE = _l2(O), _l2(E)
            joint_delta = logO[:, :4] - logE[:, :4]               # (R, 4)
            gch_delta = (logO[:, 4] - logE[:, 4]) - (logO[:, 5] - logE[:, 5])
            hcg_delta = (logO[:, 6] - logE[:, 6]) - (logO[:, 7] - logE[:, 7])

        valid = (n_o > 0) & (n_e > 0)
        joint_delta = np.where(valid[:, None], joint_delta, np.nan)
        gch_delta = np.where(valid, gch_delta, np.nan)
        hcg_delta = np.where(valid, hcg_delta, np.nan)

        codes = (all_ids // _MULT).astype(int)
        bins = (all_ids % _MULT).astype(int)
        names = uniq_chroms[codes]
        if gran == "chrom":
            keys = names
        else:
            keys = np.char.add(np.char.add(names.astype(str), ":"),
                               bins.astype(str))
        reg: dict = {}
        for i in range(R):
            vec = np.empty(RVEC)
            vec[_J_DELTA] = joint_delta[i]
            vec[_J_MARG] = MV[i, :4]
            vec[_G_DELTA] = gch_delta[i]; vec[_G_MARG] = MV[i, 4]
            vec[_H_DELTA] = hcg_delta[i]; vec[_H_MARG] = MV[i, 5]
            vec[_N_O] = n_o[i]
            reg[str(keys[i])] = vec
        out[gk] = reg
    return cell, out, qc


# --------------------------------------------------------------------------
# Assemble cells × features matrices from per-cell results
# --------------------------------------------------------------------------

def build_matrices(results: list, gran) -> dict:
    """From ``[(cell, {gran: {region: vec13}}, qc), ...]`` build the feature,
    marginal and coverage matrices for the joint / gch / hcg feature sets.

    Returns a dict with ``region_keys`` and, for each of ``joint``/``gch``/``hcg``,
    a ``(D, Mg, Cov)`` triple of ``cells × features`` arrays (joint has 4
    channels per region → ``features = regions × 4``, columns region-major;
    gch/hcg have one feature per region).  ``Cov`` is the per-region contact
    count (broadcast across a region's joint channels)."""
    gk = str(gran)
    region_keys = sorted({r for _, d, _ in results for r in d.get(gk, {})})
    ridx = {r: j for j, r in enumerate(region_keys)}
    nC, R = len(results), len(region_keys)

    Dj = np.full((nC, R * 4), np.nan); Mj = np.full((nC, R * 4), np.nan)
    Cj = np.full((nC, R * 4), np.nan)
    Dg = np.full((nC, R), np.nan); Mgm = np.full((nC, R), np.nan)
    Cg = np.full((nC, R), np.nan)
    Dh = np.full((nC, R), np.nan); Mhm = np.full((nC, R), np.nan)
    for i, (_, d, _) in enumerate(results):
        for r, vec in d.get(gk, {}).items():
            j = ridx[r]
            base = j * 4
            Dj[i, base:base + 4] = vec[_J_DELTA]
            Mj[i, base:base + 4] = vec[_J_MARG]
            Cj[i, base:base + 4] = vec[_N_O]
            Dg[i, j] = vec[_G_DELTA]; Mgm[i, j] = vec[_G_MARG]; Cg[i, j] = vec[_N_O]
            Dh[i, j] = vec[_H_DELTA]; Mhm[i, j] = vec[_H_MARG]
    return {"region_keys": region_keys,
            "joint": (Dj, Mj, Cj),
            "gch": (Dg, Mgm, Cg),
            "hcg": (Dh, Mhm, Cg)}      # gch/hcg share the per-region coverage
