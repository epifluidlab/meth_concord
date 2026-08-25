#!/usr/bin/env python3
"""Segment the coordination plane into states with a chromosome-aware HMM.

WHY AN HMM AND NOT A THRESHOLD.  Coordination domains span many bins, and
classifying each bin on its own throws that away.  On a fixture of domains
with exponential lengths, per-bin sector thresholding recovers 47% of true
states at a noise level comparable to single-cell data; the same model with a
transition prior recovers 94%.  The gain GROWS with noise (+0.22 at sd 1.0,
+0.47 at sd 3.0), which is the regime this data is actually in.  Nothing
about the states is new -- the spatial prior is doing all of the work.

WHAT IT MODELS.  Each (cell, chromosome) is one chain over the standardised
(C, F - F_exp) plane, with full-covariance Gaussian emissions per state.
Transitions are SHARED across every cell and cell type: states fitted
separately per group would not be the same states, so "differential" would
compare labels that mean different things.

THE STATES ARE FITTED, NOT ASSERTED.  K is chosen by BIC unless fixed.  The
taxonomy label on each state comes from where its emission mean LANDS in the
plane, reported so it can be checked -- if two states both sit in the
co-closed sector, that is a result, not a labelling problem to paper over.

THREE THINGS THAT WOULD SILENTLY CORRUPT IT, HANDLED HERE.  Chains break at
chromosome boundaries, or the model smooths across centromeres.  Missing
bins are MASKED, not zero-filled: a bin with no reads is not 'uncoordinated',
and filling it would manufacture domains of absence.  And the fit is pooled
over cells, because a per-cell fit on sparse single-cell tracks does not
converge.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _lse(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))
            ).squeeze(axis)


def robust_sd(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    mad = float(np.median(np.abs(x - np.median(x))))
    return 1.4826 * mad if mad > 0 else (float(np.std(x)) or np.nan)


class PlaneHMM:
    """Gaussian HMM over the 2-D coordination plane, shared transitions."""

    def __init__(self, k, seed=0, init="spread"):
        self.k = int(k)
        self.seed = int(seed)
        self.init = init

    # -- emissions -------------------------------------------------------
    def _log_b(self, X, obs):
        """log N(x | mu_s, Sigma_s) per state; ZERO where the bin is masked.

        Zero, not -inf and not an imputed value: an unobserved bin should
        neither vote nor block, so the chain carries the transition prior
        across it untouched.  Imputing would let a gap become a domain.
        """
        n = X.shape[0]
        out = np.zeros((n, self.k))
        for s in range(self.k):
            d = X[obs] - self.means[s]
            S = self.covs[s]
            det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
            det = max(det, 1e-12)
            inv = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det
            q = np.einsum("ij,jk,ik->i", d, inv, d)
            out[obs, s] = -0.5 * (q + np.log(det)) - np.log(2 * np.pi)
        return out

    # -- one BATCH of equal-length chains --------------------------------
    def _fb(self, logB, lA, lpi):
        """Forward-backward over a batch of chains at once.

        logB is (n_seq, L, k).  The recursion is sequential in t and cannot
        be avoided, but every cell shares the same chromosome lengths, so
        the CELL axis vectorises: one Python step per bin instead of one per
        (cell, bin).  At 920 cells that is the difference between ~21,000
        chains x 500 steps and 23 chromosomes x 500 steps per EM iteration.
        """
        n_seq, L, k = logB.shape
        la = np.empty_like(logB)
        lb = np.zeros_like(logB)
        la[:, 0] = lpi + logB[:, 0]
        for t in range(1, L):
            la[:, t] = logB[:, t] + _lse(la[:, t - 1][:, :, None]
                                         + lA[None], 1)
        for t in range(L - 2, -1, -1):
            lb[:, t] = _lse(lA[None]
                            + (logB[:, t + 1] + lb[:, t + 1])[:, None, :], 2)
        ll = _lse(la[:, -1], 1)
        return la, lb, la + lb - ll[:, None, None], ll

    def _batches(self, X, lengths, obs, n_cells):
        """Group the flat arrays into one batch per chromosome.

        Layout from build_sequences is cell-major: cell0-chr1..chrN, then
        cell1-chr1..chrN.  Every cell has the same chromosome lengths, so
        chromosome j is rows j, N+j, 2N+j, ... and they stack into one
        (n_cells, L_j, .) array that the recursion runs over in a single
        pass.
        """
        n_chrom = len(lengths) // n_cells
        starts = np.r_[0, np.cumsum(lengths)]
        out = []
        for j in range(n_chrom):
            idx = np.concatenate([np.arange(starts[c * n_chrom + j],
                                            starts[c * n_chrom + j + 1])
                                  for c in range(n_cells)])
            L = int(lengths[j])
            out.append((idx, X[idx].reshape(n_cells, L, 2),
                        obs[idx].reshape(n_cells, L)))
        return out

    def _stats(self, Xb, ob, lA, lpi, want_xi=True):
        """Sufficient statistics from one chromosome batch."""
        logB = self._log_b_batch(Xb, ob)
        la, lb, lg, ll = self._fb(logB, lA, lpi)
        g = np.exp(lg)
        gm = np.where(ob[:, :, None], g, 0.0)
        g_sum = gm.sum((0, 1))
        gx = np.einsum("nlk,nld->kd", gm, np.where(ob[:, :, None], Xb, 0.0))
        gxx = np.zeros((self.k, 2, 2))
        for s_ in range(self.k):
            d = np.where(ob[:, :, None], Xb - self.means[s_], 0.0)
            w = gm[:, :, s_]
            flat_d = d.reshape(-1, 2)
            gxx[s_] = (flat_d * w.reshape(-1, 1)).T @ flat_d
        xi = np.zeros((self.k, self.k))
        if want_xi and Xb.shape[1] > 1:
            # chunk over cells: the (n_seq, L-1, k, k) tensor is the one
            # place this can blow memory on a real matrix
            step = max(1, int(4e6 // max(Xb.shape[1] * self.k ** 2, 1)))
            for c0 in range(0, Xb.shape[0], step):
                sl = slice(c0, c0 + step)
                m2 = (la[sl, :-1, :, None] + lA[None, None]
                      + (logB[sl, 1:] + lb[sl, 1:])[:, :, None, :])
                m2 -= _lse(m2.reshape(m2.shape[0], m2.shape[1], -1), 2
                           )[:, :, None, None]
                xi += np.exp(m2).sum((0, 1))
        return g_sum, gx, gxx, xi, g[:, 0].sum(0), float(ll.sum())

    def _log_b_batch(self, Xb, ob):
        """log-emissions for a batch; ZERO where masked, as in _log_b."""
        n, L, _ = Xb.shape
        out = np.zeros((n, L, self.k))
        flat = Xb.reshape(-1, 2)
        m = ob.reshape(-1)
        for s_ in range(self.k):
            d = flat[m] - self.means[s_]
            S = self.covs[s_]
            det = max(S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0], 1e-12)
            inv = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det
            q = np.einsum("ij,jk,ik->i", d, inv, d)
            col = out[:, :, s_].reshape(-1)
            col[m] = -0.5 * (q + np.log(det)) - np.log(2 * np.pi)
        return out

    def fit(self, X, lengths, obs, n_cells, n_iter=40, tol=1e-4,
            threads=1, verbose=False):
        rng = np.random.default_rng(self.seed)
        good = X[obs]
        if len(good) < self.k * 10:
            raise ValueError(f"only {len(good)} observed bins for k={self.k}")
        # k-means start.  Per-axis quantiles were the obvious thing and are
        # wrong: (5th pct of C, 5th pct of dF) through (95th, 95th) all lie
        # on the cloud's MAIN DIAGONAL, so a state at (+C, -dF) never gets
        # a mean anywhere near itself and EM lands in a bad optimum.
        if self.init == "spread":
            sub = density_equalised(good, rng)
        else:
            sub = good
            if len(sub) > 200_000:
                sub = sub[rng.choice(len(sub), 200_000, replace=False)]
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=self.k, n_init=10,
                        random_state=self.seed).fit(sub)
            self.means = km.cluster_centers_.astype(float)
        except Exception:
            self.means = _farthest_point_init(sub, self.k, rng)
        self.means = self.means + rng.normal(0, 1e-6, self.means.shape)
        base = np.cov(good.T) + np.eye(2) * 1e-6
        self.covs = np.stack([base.copy() for _ in range(self.k)])
        A = np.full((self.k, self.k), 0.05 / max(self.k - 1, 1))
        np.fill_diagonal(A, 0.95)
        pi = np.full(self.k, 1.0 / self.k)

        batches = self._batches(X, lengths, obs, n_cells)
        runner = _runner(threads)
        prev = -np.inf
        self.trace = []
        for it in range(n_iter):
            lA, lpi = np.log(A), np.log(pi)
            res = runner(self, batches, lA, lpi)
            g_sum = sum(r[0] for r in res)
            gx = sum(r[1] for r in res)
            gxx = sum(r[2] for r in res)
            xi = sum(r[3] for r in res)
            pi_acc = sum(r[4] for r in res)
            total = sum(r[5] for r in res)
            self.means = gx / np.maximum(g_sum[:, None], 1e-12)
            for s_ in range(self.k):
                self.covs[s_] = (gxx[s_] / max(g_sum[s_], 1e-12)
                                 + np.eye(2) * 1e-6)
            A = xi / np.maximum(xi.sum(1, keepdims=True), 1e-12)
            pi = pi_acc / max(pi_acc.sum(), 1e-12)
            self.trace.append(float(total))
            if verbose:
                print(f"    iter {it:>2}  loglik {total:,.1f}",
                      file=sys.stderr)
            if abs(total - prev) < tol * max(abs(total), 1.0):
                break
            prev = total
        self.A, self.pi, self.loglik = A, pi, total
        self.n_obs = int(obs.sum())
        return self

    # -- decoding --------------------------------------------------------
    def decode(self, X, lengths, obs, n_cells):
        lA, lpi = np.log(self.A), np.log(self.pi)
        path = np.full(X.shape[0], -1, int)
        post = np.full((X.shape[0], self.k), np.nan)
        for idx, Xb, ob in self._batches(X, lengths, obs, n_cells):
            logB = self._log_b_batch(Xb, ob)
            _la, _lb, lg, _ll = self._fb(logB, lA, lpi)
            n, L, _ = logB.shape
            dp = np.empty((n, self.k))
            bp = np.zeros((n, L, self.k), np.int8)
            dp[:] = lpi + logB[:, 0]
            for t in range(1, L):
                mm = dp[:, :, None] + lA[None]
                bp[:, t] = mm.argmax(1)
                dp = mm.max(1) + logB[:, t]
            p = np.empty((n, L), int)
            p[:, -1] = dp.argmax(1)
            for t in range(L - 2, -1, -1):
                p[:, t] = np.take_along_axis(
                    bp[:, t + 1], p[:, t + 1][:, None], 1)[:, 0]
            path[idx] = p.reshape(-1)
            post[idx] = np.exp(lg).reshape(-1, self.k)
        return path, post

    # -- model size ------------------------------------------------------
    def n_params(self):
        return (self.k * 2                      # means
                + self.k * 3                    # 2x2 symmetric covariances
                + self.k * (self.k - 1)         # transitions
                + (self.k - 1))                 # start

    def bic(self):
        """Lower is better.  n is the number of OBSERVED bins, not the
        matrix size: masked bins contribute no likelihood, so counting them
        would reward a model for the data it never saw."""
        return -2.0 * self.loglik + self.n_params() * np.log(max(self.n_obs, 2))


def _runner(threads):
    """Fan the per-chromosome work out, or run it in this process.

    Chromosomes are independent given the current parameters, so the E step
    parallelises exactly.  Threads, not processes: the inner work is numpy,
    which releases the GIL, and the batches are large enough that pickling
    them to workers would cost more than it saves.
    """
    if threads and threads > 1:
        from concurrent.futures import ThreadPoolExecutor

        def run(model, batches, lA, lpi):
            with ThreadPoolExecutor(max_workers=threads) as ex:
                return list(ex.map(
                    lambda b: model._stats(b[1], b[2], lA, lpi), batches))
        return run

    def run(model, batches, lA, lpi):
        return [model._stats(b[1], b[2], lA, lpi) for b in batches]
    return run


def warp_log_radial(X, obs, centre=None):
    """Compress the plane radially about its own centre.

    The real plane is a dense core with arms radiating off it.  Gaussian
    components in Cartesian coordinates are the wrong SHAPE for that: the
    core holds most of the likelihood, so every component settles inside it
    and the arms -- which are the structure -- go unmodelled.  A real fit
    left three states within six units of each other while the cloud spanned
    a hundred.

    Replacing the radius r by log(1 + r) about the centre keeps every angle
    exactly and shrinks distance non-linearly, so the crowded core takes up
    less area and the sparse arms take up more.  Equal area in the warped
    plane is then closer to equal information, which is the assumption
    Gaussian components are making anyway.  On a fixture with a very dense
    core it doubled the separation k-means found between centres, 0.371 to
    0.803 relative to the cloud's own size.

    Angles survive untouched, so the direction labelling means the same
    thing in either space; only radii are comparable differently, and
    ``unwarp_log_radial`` puts the fitted means back for reporting.
    """
    X = np.asarray(X, float)
    if centre is None:
        centre = (np.median(X[obs], axis=0) if obs.any()
                  else np.zeros(X.shape[1]))
    d = X - centre
    r = np.linalg.norm(d, axis=1)
    scale = np.where(r > 1e-12, np.log1p(r) / np.maximum(r, 1e-12), 1.0)
    return d * scale[:, None], np.asarray(centre, float)


def unwarp_log_radial(W, centre):
    """Fitted means back on the original plane, for reporting."""
    W = np.atleast_2d(np.asarray(W, float))
    rw = np.linalg.norm(W, axis=1)
    r = np.expm1(rw)
    scale = np.where(rw > 1e-12, r / np.maximum(rw, 1e-12), 1.0)
    return W * scale[:, None] + centre


def plane_sample(X, sel, warp_centre=None, cap=300_000, seed=0,
                 return_index=False):
    """Selected rows of X back on the ORIGINAL plane, capped for plotting.

    Every diagnostic is quoted in plane units -- the |z| mask, the reach
    warning, panel (a)'s axes -- and ``main`` reports ``model.means`` on the
    plane.  But under ``--space log-radial`` the matrix the model was fitted
    to is NOT the plane, and reading it as though it were fails silently in
    both directions: the reach warning compares warped distances against a
    plane threshold and never fires, and panel (a) draws a log-compressed
    cloud with the means a hundred units away, which looks exactly like a
    fit that has gone wrong rather than a figure that has.

    Unwarping is exact, so this is a change of view and not of data.  It is
    done after subsampling because the full matrix is tens of millions of
    rows and nothing here needs more than the sample.
    """
    idx = np.flatnonzero(np.asarray(sel, bool))
    if len(idx) > cap:
        idx = np.random.default_rng(seed).choice(idx, cap, replace=False)
    Xs = np.asarray(X, float)[idx]
    if warp_centre is not None:
        Xs = unwarp_log_radial(Xs, np.asarray(warp_centre, float))
    return (Xs, idx) if return_index else Xs


def center_cells(C, dF):
    """Subtract each cell's own median, per axis.

    A state is a run of bins INSIDE a cell, so the spread an HMM can model
    is the spread within cells.  Anything that slides a whole cell up or
    down the plane -- coverage, GpC enzyme efficiency, how well that cell
    connected in the RWR graph -- is between-cell variation, and Gaussian
    components fit it as readily as they fit domains, which turns "states"
    into a clustering of cells wearing a domain model's clothes.

    Median rather than mean: a few extreme bins should not decide where a
    cell sits.  Subtracting only a location leaves the shape along the
    genome untouched, which a per-cell RANK (what the pipeline applies
    before its UMAPs) would not.

    It is compositional in principle -- a bin's centred value depends on the
    others through the median -- and that is harmless while genuine
    aberrations are a small share of the genome and wrong when they are not.
    A cell whose coordination really differs everywhere has that difference
    subtracted.
    """
    C, dF = np.asarray(C, float), np.asarray(dF, float)
    with np.errstate(invalid="ignore"):
        mc = np.nanmedian(C, axis=1)
        mf = np.nanmedian(dF, axis=1)
    mc = np.where(np.isfinite(mc), mc, 0.0)
    mf = np.where(np.isfinite(mf), mf, 0.0)
    return C - mc[:, None], dF - mf[:, None], {
        "median_shift_C": float(np.median(np.abs(mc))),
        "median_shift_dF": float(np.median(np.abs(mf))),
        "spread_of_cell_centres_C": float(_robust_sd_1d(mc)),
        "spread_of_cell_centres_dF": float(_robust_sd_1d(mf)),
    }


def _robust_sd_1d(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def cell_summaries(X, obs, n_cells, warp_centre=None, chunk=2_000_000):
    """Per-cell means on the plane, and the share of spread BETWEEN cells.

    The HMM segments the genome: a state is meant to be a run of bins
    inside one cell.  That only makes sense if the spread it is modelling
    is within cells.  Where most of the spread is between cells, the
    Gaussian components fit CELL IDENTITY instead -- the fit becomes a
    clustering of cells wearing a domain model's clothes, and the run
    lengths that come out are an artefact of how the cells happen to be
    ordered along the genome rather than a measurement of anything.

    Structure on the plane that is between cells cannot be segmented away,
    corrected away by a per-bin batch mean, or fixed by a different K.

    One pass in row chunks, so nothing the size of X is duplicated.
    """
    X = np.asarray(X, float)
    n = len(X)
    n_bins = max(n // max(int(n_cells), 1), 1)
    cnt = np.zeros(int(n_cells))
    s1 = np.zeros((int(n_cells), 2))
    s2 = np.zeros(2)
    for a0 in range(0, n, chunk):
        b0 = min(a0 + chunk, n)
        m = np.asarray(obs[a0:b0], bool)
        if not m.any():
            continue
        Z = X[a0:b0][m]
        if warp_centre is not None:
            Z = unwarp_log_radial(Z, np.asarray(warp_centre, float))
        c = (np.arange(a0, b0)[m]) // n_bins
        c = np.minimum(c, int(n_cells) - 1)
        cnt += np.bincount(c, minlength=int(n_cells))
        for j in range(2):
            s1[:, j] += np.bincount(c, weights=Z[:, j],
                                    minlength=int(n_cells))
            s2[j] += float((Z[:, j] ** 2).sum())
    tot = cnt.sum()
    if tot <= 0:
        return np.full((int(n_cells), 2), np.nan), {"C": float("nan"),
                                                    "dF": float("nan")}
    grand = s1.sum(0) / tot
    ok = cnt > 0
    mu = np.full((int(n_cells), 2), np.nan)
    mu[ok] = s1[ok] / cnt[ok, None]
    frac = {}
    for j, name in enumerate(("C", "dF")):
        ss_total = s2[j] - tot * grand[j] ** 2
        ss_between = float((cnt[ok] * (mu[ok, j] - grand[j]) ** 2).sum())
        frac[name] = (ss_between / ss_total if ss_total > 0
                      else float("nan"))
    return mu, frac


def point_colour(tab, column, n_bins, max_levels=12):
    """A per-(cell, bin) colour spec from a per-cell column.

    X is laid out cell-major, so row r belongs to cell r // n_bins whatever
    order the bins were sorted into.
    """
    if column not in tab.columns:
        raise SystemExit(f"--colour-by {column!r} not in the samples table; "
                         f"columns are {list(tab.columns)[:12]}")
    col = tab[column]
    vals = pd.to_numeric(col, errors="coerce")
    numeric = bool(col.notna().any() and vals.notna().sum()
                   == col.notna().sum())
    if numeric and col.nunique(dropna=True) > max_levels:
        return {"kind": "num", "column": column,
                "values": np.repeat(vals.to_numpy(float), n_bins)}
    cat = pd.Categorical(col.astype(str).where(col.notna(), "n/a"))
    if len(cat.categories) > max_levels:
        raise SystemExit(
            f"--colour-by {column!r} has {len(cat.categories)} levels; "
            f"panel (a) can separate at most {max_levels}.  Pick a coarser "
            f"column, or a numeric one for a colour bar.")
    return {"kind": "cat", "column": column,
            "names": [str(c) for c in cat.categories],
            "codes": np.repeat(np.asarray(cat.codes, int), n_bins)}


def load_cell_table(path, cell_column, index):
    """Per-cell annotation aligned to the concordance matrix's rows."""
    df = pd.read_csv(path, sep="\t")
    if cell_column not in df.columns:
        raise SystemExit(f"--cell-column {cell_column!r} not in {path}; "
                         f"columns are {list(df.columns)[:10]}")
    tab = df.set_index(cell_column)
    tab = tab[~tab.index.duplicated(keep="first")]
    return tab.reindex(index)


def batch_series(tab, spec, flag, path):
    """One column, or several as a frame -- what the pipeline's own
    correction accepts, so 'exp_batch,lib_batch' means here what it means
    there."""
    cols = [c.strip() for c in str(spec).split(",") if c.strip()]
    missing = [c for c in cols if c not in tab.columns]
    if missing:
        raise SystemExit(f"{flag} not in {path}: {missing}")
    return tab[cols[0]] if len(cols) == 1 else tab[cols]


def batch_group_confounding(batch, group):
    """For each batch level, the share taken by its commonest group.

    Every per-bin correction here removes each batch's mean.  Where a batch
    is almost entirely one biological group, that mean IS the group, and
    the correction deletes the difference it was meant to protect.  This is
    the ordinary failure of ComBat on a design where the tumours and the
    reference cell types were not prepared together, which is most designs.
    """
    b = pd.Series(np.asarray(batch, object), dtype=object).astype(str)
    g = pd.Series(np.asarray(group, object), dtype=object).astype(str)
    rows = []
    for lev, sub in g.groupby(b.to_numpy()):
        vc = sub.value_counts()
        rows.append({"batch": str(lev), "n_cells": int(len(sub)),
                     "top_group": str(vc.index[0]),
                     "share": float(vc.iloc[0]) / max(len(sub), 1)})
    return pd.DataFrame(rows).sort_values(
        "share", ascending=False).reset_index(drop=True)


def density_equalised(X, rng, n=60_000, grid=120):
    """Subsample so every occupied part of the plane is equally represented.

    Gaussian-mixture EM weights by DENSITY, so when the plane is a dense
    core with low-density arms radiating off it, k-means seeds every centre
    inside the core and EM never leaves.  That is not a tuning problem: the
    core genuinely holds most of the likelihood.  Re-weighting each point by
    the inverse of its local density makes the ARMS as visible to the
    initialiser as the core, which is what lets EM start with a component on
    each and then refine.

    The fit itself still uses every point at its true weight -- only the
    starting positions are chosen this way.
    """
    X = np.asarray(X, float)
    if len(X) == 0:
        return X
    lo = np.nanpercentile(X, 0.5, axis=0)
    hi = np.nanpercentile(X, 99.5, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    ij = np.clip(((X - lo) / span * grid).astype(int), 0, grid - 1)
    key = ij[:, 0] * grid + ij[:, 1]
    counts = np.bincount(key, minlength=grid * grid)
    w = 1.0 / np.maximum(counts[key], 1)
    w = w / w.sum()
    take = min(n, len(X))
    idx = rng.choice(len(X), take, replace=False, p=w)
    return X[idx]


def _farthest_point_init(X, k, rng):
    """k-means++ style seeding, for when sklearn is unavailable."""
    idx = [int(rng.integers(len(X)))]
    for _ in range(k - 1):
        d = np.min(((X[:, None, :] - X[idx][None, :, :]) ** 2).sum(-1), 1)
        tot = d.sum()
        p = d / tot if tot > 0 else None
        idx.append(int(rng.choice(len(X), p=p)))
    return X[idx].astype(float)


def _spans(lengths):
    a = 0
    for L in lengths:
        yield a, a + L
        a += L


def build_sequences(C, dF, bins, scale_c=None, scale_f=None,
                    clip=10.0, space="plane"):
    """Stack cells into one long array of chains, one per (cell, chromosome).

    The chain must break at every chromosome edge; a single chain per cell
    would let the model smooth a domain across a centromere or off the end
    of chr1 into chr2.
    """
    chrom = np.array([str(b).rsplit(":", 1)[0] for b in bins])
    start = np.array([int(str(b).rsplit(":", 1)[1]) for b in bins])
    order = np.lexsort((start, chrom))
    chrom, C, dF = chrom[order], C[:, order], dF[:, order]
    sc = robust_sd(C) if scale_c is None else float(scale_c)
    sf = robust_sd(dF) if scale_f is None else float(scale_f)
    if not (np.isfinite(sc) and np.isfinite(sf)):
        raise ValueError("one axis has no spread; nothing to segment")
    edges = np.flatnonzero(np.r_[True, chrom[1:] != chrom[:-1],
                                 True])
    per_chrom = np.diff(edges)
    X = np.empty((C.shape[0] * C.shape[1], 2))
    X[:, 0] = (C / sc).ravel()
    X[:, 1] = (dF / sf).ravel()
    obs = np.isfinite(X).all(1)
    info = {"scale_c": sc, "scale_f": sf}
    if obs.any():
        _med = np.array([np.median(X[obs, 0]), np.median(X[obs, 1])])
        _dev = np.abs(X[obs] - _med)
        info["max_abs_z"] = float(_dev.max())
        info["frac_beyond_10"] = float(np.mean(_dev > 10))
        info["median_z"] = _med.tolist()
    if isinstance(clip, str) and str(clip).lower() == "auto":
        med0 = np.array([np.median(X[obs, 0]), np.median(X[obs, 1])])
        dev0 = np.abs(X[obs] - med0).max(1)
        clip, ladder = auto_max_z(dev0)
        info["max_z_auto"] = float(clip)
        info["max_z_ladder"] = ladder
    if clip and clip > 0:
        # MASK the extremes, do not clip them to a boundary.  Clipping turns
        # a heavy tail into a spike at +-z, and EM allocates states to that
        # spike just as happily: on a Cauchy-tailed fixture three of five
        # states sat exactly on the clip corners with 1-bin runs.  Masking
        # instead says what is true -- a bin whose estimate is ten robust
        # sds from the median is not measuring coordination -- and it is
        # already how bins with no data are handled, so the chain carries
        # the transition prior across them untouched.
        #
        # A fixed z, not a percentile: with a Cauchy-ish tail the 99.5th
        # percentile of |z| came out 341, which left EM a tail to chase.
        # CENTRED on the median, not on zero.  The values themselves stay
        # uncentred, because C = 0 is a meaningful point and shifting it
        # would destroy that -- but "extreme" has to mean far from the BULK,
        # not far from zero.  On data whose whole cloud sits well away from
        # zero (C came back strictly positive on real data) an uncentred cut
        # marks every bin extreme and masks the entire matrix.
        med = np.array([np.median(X[obs, 0]), np.median(X[obs, 1])])
        extreme = (np.abs(X - med) > clip).any(1) & obs
        info.update({"mask_beyond_z": float(clip),
                     "masked_rows": extreme,
                     "n_masked_extreme": int(extreme.sum()),
                     "frac_masked_extreme": float(
                         extreme.sum() / max(int(obs.sum()), 1))})
        obs = obs & ~extreme
    if space == "log-radial":
        Xw, centre = warp_log_radial(np.where(np.isfinite(X), X, np.nan),
                                     obs)
        X = np.where(np.isfinite(Xw), Xw, 0.0)
        info.update({"space": "log-radial", "warp_centre": centre.tolist()})
    else:
        info["space"] = "plane"
    X = np.where(np.isfinite(X), X, 0.0)
    lengths = list(per_chrom) * C.shape[0]
    return X, np.array(lengths, int), obs, order, info


COORD_CLASSES = ("uncoordinated", "coordinated balanced", "co-open shifted",
                 "co-closed shifted", "anti-coordinated")


def label_states(means, r_cut="auto", sector=30.0, auto_frac=0.35):
    """Name each fitted state by where its emission mean LANDS.

    Post hoc and reported, never assumed.  Two states can legitimately share
    a label -- that says the data wanted more structure inside one sector
    than the taxonomy has names for, which is worth seeing rather than
    hiding by forcing a bijection.

    r_cut is RELATIVE by default, and that matters.  An absolute cut in
    standardised units assumes the scale used to standardise is the NULL
    scale, which only holds when most bins are null.  Where a large share of
    the genome carries signal, the robust sd absorbs it, every state shrinks
    toward the origin, and an absolute cut of 2.0 calls all of them
    uncoordinated -- which is what happened on a fixture whose robust sd
    came out 3.41 against states spanning -3..+3.  A cut at a fraction of
    the largest fitted radius asks the question that is actually meant: is
    this state near the origin RELATIVE TO THE OTHERS the model found.
    Pass a number to override when the null scale is known.
    """
    mu_all = np.asarray(means, float)
    radii = np.hypot(mu_all[:, 0], mu_all[:, 1])
    if isinstance(r_cut, str):
        if r_cut.lower() != "auto":
            raise ValueError(f"--r-cut must be a number or 'auto', "
                             f"got {r_cut!r}")
        r_cut = auto_frac * float(radii.max()) if radii.size else 0.0
    out = []
    for mu in mu_all:
        R = float(np.hypot(mu[0], mu[1]))
        th = float(np.degrees(np.arctan2(mu[1], mu[0])))
        if R < r_cut:
            name = "uncoordinated"
        elif abs(th) <= sector:
            name = "coordinated balanced"
        elif abs(th) >= 180 - sector:
            name = "anti-coordinated"
        elif th > 0:
            name = "co-open shifted"
        else:
            name = "co-closed shifted"
        out.append({"mean_C": mu[0], "mean_dF": mu[1], "R": R, "theta": th,
                    "label": name})
    return out


def auto_max_z(dev, target_gauss=0.999, hard_max=50.0):
    """Choose the mask threshold from the shape of the data.

    A fixed z is a guess about how heavy the tail is.  Pick it instead where
    the empirical tail departs from what a Gaussian core would produce: walk
    a ladder of candidate cuts and take the smallest at which the retained
    fraction stops rising materially -- the knee where the bulk has been
    captured and everything further out is tail.

    Returns (z, table) so the choice is reportable rather than magic.
    """
    dev = np.asarray(dev, float)
    dev = dev[np.isfinite(dev)]
    if dev.size == 0:
        return hard_max, []
    ladder = np.array([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50], float)
    ladder = ladder[ladder <= hard_max]
    keep = np.array([float(np.mean(dev <= z)) for z in ladder])
    rows = [{"z": float(z), "frac_kept": float(k)}
            for z, k in zip(ladder, keep)]
    # the knee: the first cut past which another step buys under 0.5% more
    gain = np.r_[keep[0], np.diff(keep)]
    knee = ladder[-1]
    for z, g, k in zip(ladder, gain, keep):
        if k >= target_gauss:
            knee = z
            break
        if g < 0.005 and k > 0.90:
            knee = z
            break
    return float(knee), rows


def genome_autocorrelation(X, lengths, obs, n_cells, max_lag=20):
    """Autocorrelation along the genome, per cell and for the group mean.

    This is the number that decides whether an HMM can help at all.  The
    transition prior is the only reason to use one, and it has nothing to
    work with unless consecutive bins agree more than chance.

    Both are reported because they answer different questions.  A low
    PER-CELL value with a high POOLED value means the domain structure is
    real but buried under per-cell sampling noise -- in which case the fit
    should be learnt on the pooled track and only the decoding done per
    cell.  Both low means there is no domain structure at this bin size.
    """
    n_chrom = len(lengths) // n_cells
    starts = np.r_[0, np.cumsum(lengths)]
    out = {"lag": list(range(1, max_lag + 1))}
    for axis, name in ((0, "C"), (1, "dF")):
        per_cell = np.zeros(max_lag)
        pooled = np.zeros(max_lag)
        wc = np.zeros(max_lag)
        wp = np.zeros(max_lag)
        for j in range(n_chrom):
            L = int(lengths[j])
            idx = np.concatenate([np.arange(starts[c * n_chrom + j],
                                            starts[c * n_chrom + j + 1])
                                  for c in range(n_cells)])
            V = np.where(obs[idx], X[idx, axis], np.nan).reshape(n_cells, L)
            with np.errstate(invalid="ignore"):
                mean_track = np.nanmean(V, axis=0)
            for lag in range(1, max_lag + 1):
                if L <= lag:
                    continue
                a, b = V[:, :-lag], V[:, lag:]
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() > 10:
                    av, bv = a[ok], b[ok]
                    r = np.corrcoef(av, bv)[0, 1]
                    if np.isfinite(r):
                        per_cell[lag - 1] += r * ok.sum()
                        wc[lag - 1] += ok.sum()
                ma, mb = mean_track[:-lag], mean_track[lag:]
                okm = np.isfinite(ma) & np.isfinite(mb)
                if okm.sum() > 10:
                    r = np.corrcoef(ma[okm], mb[okm])[0, 1]
                    if np.isfinite(r):
                        pooled[lag - 1] += r * okm.sum()
                        wp[lag - 1] += okm.sum()
        out[f"{name}_within_cell"] = list(
            np.where(wc > 0, per_cell / np.maximum(wc, 1), np.nan))
        out[f"{name}_pooled"] = list(
            np.where(wp > 0, pooled / np.maximum(wp, 1), np.nan))
    return out


def qc_figure(path, X, obs, model, acf, mask_z, ladder, loglik_trace,
              labels, title, masked=None, warp_centre=None, colour=None,
              cell_means=None):
    """Six panels that answer whether the fit means anything.

    Ordered by what would invalidate the result soonest: if the plane is one
    blob there are no states to find, and nothing downstream matters.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(14.5, 8.4))
    # Drawn on the PLANE, whatever space the fit ran in, because that is
    # where model.means are reported and the axes are labelled.
    sub, sub_idx = plane_sample(X, obs, warp_centre, seed=0,
                                return_index=True)

    a = ax[0, 0]
    # The masked points are drawn too, because the mask exists to remove a
    # tail and on real data the tail and the STRUCTURE can be the same
    # points.  A mask sitting where the arms are is the single most likely
    # reason a fit collapses into the core, and it should be visible rather
    # than inferred from the run log.
    ms = None
    if masked is not None and masked.any():
        ms = plane_sample(X, masked, warp_centre, cap=40_000, seed=1)
        a.scatter(ms[:, 0], ms[:, 1], s=1.5, c="#c0392b", alpha=0.25,
                  lw=0, zorder=1, label=f"masked ({masked.mean():.1%})")
    if colour is None:
        a.hexbin(sub[:, 0], sub[:, 1], gridsize=70, bins="log", cmap="Blues",
                 mincnt=1, zorder=2)
    elif colour["kind"] == "cat":
        # One colour per level, with that level's median marked.  If an arm
        # is one batch, or one cell type, it says so here and no amount of
        # re-fitting is the answer.
        cm = plt.get_cmap("tab10")
        g = np.asarray(colour["codes"])[sub_idx]
        for i, nm in enumerate(colour["names"]):
            sel = g == i
            if not sel.any():
                continue
            a.scatter(sub[sel, 0], sub[sel, 1], s=1.2, alpha=0.25, lw=0,
                      color=cm(i % 10), zorder=2,
                      label=f"{nm} ({sel.mean():.0%})")
            med = np.median(sub[sel], axis=0)
            a.scatter([med[0]], [med[1]], s=85, marker="D", zorder=4,
                      facecolor=cm(i % 10), edgecolor="k", lw=1.0)
    else:
        v = np.asarray(colour["values"], float)[sub_idx]
        sm = a.scatter(sub[:, 0], sub[:, 1], c=v, s=1.2, alpha=0.35, lw=0,
                       cmap="viridis", zorder=2)
        fig.colorbar(sm, ax=a, fraction=0.046, label=colour["column"])
    if cell_means is not None:
        cm_ = np.asarray(cell_means, float)
        cm_ = cm_[np.isfinite(cm_).all(1)]
        # One point per CELL.  If the arms are here too, they are between
        # cells, and no segmentation of the genome reaches them.
        a.scatter(cm_[:, 0], cm_[:, 1], s=7, c="#111111", alpha=0.55,
                  lw=0, zorder=3, label=f"cell means (n={len(cm_):,})")
    a.scatter(model.means[:, 0], model.means[:, 1], s=110, marker="x",
              c="#1a1a1a", lw=2.4, zorder=5)
    for i, m in enumerate(model.means):
        a.annotate(str(i), m, fontsize=8, color="#c0392b",
                   xytext=(4, 4), textcoords="offset points", zorder=6)
    if a.get_legend_handles_labels()[0]:
        ncol = 1 + (len(colour["names"]) // 7
                    if colour and colour.get("kind") == "cat" else 0)
        a.legend(fontsize=5.5, frameon=False, loc="best", markerscale=6,
                 ncol=ncol)
    a.set_title("a  is the plane multimodal?" if colour is None else
                f"a  the plane, coloured by {colour['column']}",
                fontsize=10, loc="left")
    a.set_xlabel("C (standardised)", fontsize=8)
    a.set_ylabel("F − F_exp (standardised)", fontsize=8)
    # Measured off the DATA, before the means expand the axes -- a mean
    # outside this box is a mean in the wrong coordinates.
    drawn = sub if ms is None else np.vstack([sub, ms])
    extent = (float(drawn[:, 0].min()), float(drawn[:, 0].max()),
              float(drawn[:, 1].min()), float(drawn[:, 1].max()))

    a = ax[0, 1]
    lag = acf["lag"]
    for key, style in (("C_within_cell", "-o"), ("C_pooled", "--s"),
                       ("dF_within_cell", "-^"), ("dF_pooled", "--v")):
        a.plot(lag, acf[key], style, ms=3.2, lw=1.2, label=key)
    a.axhline(0, color="0.6", lw=0.8)
    a.set_title("b  spatial autocorrelation along the genome", fontsize=10,
                loc="left")
    a.set_xlabel("lag (bins)", fontsize=8)
    a.set_ylabel("correlation", fontsize=8)
    a.legend(fontsize=6.5, frameon=False)

    a = ax[0, 2]
    # Masked points included: the panel exists to show where the cut falls
    # relative to the cloud, and the retained set stops at the cut by
    # construction, so on its own it can only ever show the cut at its edge.
    seen = obs if masked is None else (obs | masked)
    dev = np.abs(plane_sample(X, seen, warp_centre, seed=2)
                 - np.median(sub, axis=0)).max(1)
    hi = float(np.percentile(dev, 99.5)) if len(dev) else 60.0
    if np.isfinite(mask_z) and mask_z > 0:
        hi = max(hi, mask_z * 1.3)
    a.hist(np.clip(dev, 0, max(hi, 1e-9)), bins=120, color="#7f8c8d")
    a.axvline(mask_z, color="#c0392b", lw=1.4)
    a.annotate(f"mask at |z| = {mask_z:g}", (mask_z, a.get_ylim()[1] * 0.8),
               xytext=(6, 0), textcoords="offset points", fontsize=8,
               color="#c0392b")
    a.set_yscale("log")
    a.set_title("c  distance from the median, and the cut", fontsize=10,
                loc="left")
    a.set_xlabel("|z| (max over the two axes)", fontsize=8)

    a = ax[1, 0]
    im = a.imshow(model.A, cmap="magma_r", vmin=0, vmax=1)
    for i in range(model.k):
        for j in range(model.k):
            a.text(j, i, f"{model.A[i, j]:.2f}", ha="center", va="center",
                   fontsize=7,
                   color="white" if model.A[i, j] > 0.5 else "0.2")
    a.set_xticks(range(model.k)); a.set_yticks(range(model.k))
    a.set_title("d  transition matrix", fontsize=10, loc="left")
    a.set_xlabel("to state", fontsize=8); a.set_ylabel("from state", fontsize=8)
    fig.colorbar(im, ax=a, fraction=0.046)

    a = ax[1, 1]
    runs = 1.0 / np.maximum(1 - np.diag(model.A), 1e-9)
    cols = ["#2c3e50" if r >= 3 else "#c0392b" for r in runs]
    a.bar(range(model.k), runs, color=cols)
    a.axhline(3, color="#c0392b", ls="--", lw=1.0)
    a.annotate("below 3 bins = no domains", (0, 3), xytext=(2, 4),
               textcoords="offset points", fontsize=7.5, color="#c0392b")
    a.set_xticks(range(model.k))
    a.set_xticklabels([f"{i}\n{labels[i][:12]}" for i in range(model.k)],
                      fontsize=6.5)
    a.set_title("e  expected run length per state", fontsize=10, loc="left")
    a.set_ylabel("bins", fontsize=8)

    a = ax[1, 2]
    if loglik_trace:
        a.plot(range(1, len(loglik_trace) + 1), loglik_trace, "-o", ms=3,
               color="#2c3e50")
        a.set_xlabel("EM iteration", fontsize=8)
        a.set_ylabel("log-likelihood", fontsize=8)
    a.set_title("f  EM convergence", fontsize=10, loc="left")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {"panel_a_extent": extent, "dev_max": float(dev.max())
            if len(dev) else 0.0,
            "means": np.asarray(model.means, float).copy()}


def archetypes(k, anchor):
    """The taxonomy to match against, which depends on K and on the anchor.

    Each K adds one archetype to the one below, so the sets nest and a
    smaller K is a strictly simpler claim rather than a different one:

      3  null, co-open shifted, co-closed shifted
      4  + coordinated balanced
      5  + anti-coordinated

    K = 5 needs C to cross zero, since C = 2*Cov below zero is what
    anti-coordination MEANS.  K = 4 keeps 'coordinated balanced', a state
    with more concordant mass than the null but no directional preference.
    K = 3 drops that too, which is the right model when the elevated-C
    states are all directional -- on a real fit 'coordinated balanced'
    absorbed 79.6% of the genome, which is not a plausible category and is
    what a dominant unstructured blob looks like when a name is forced onto
    it.

    The null archetype is called 'uncoordinated' only when the origin is the
    anchor, because that is when C = 0 is actually reached.  Anchored on the
    lowest-C state it is 'least coordinated', which is what it is: there is
    coordination everywhere and this state has the least of it.
    """
    if k not in (3, 4, 5):
        raise ValueError(
            f"relative labelling handles 3, 4 or 5 states, got {k}.  Three "
            f"when the coordinated states are all directional, four to add "
            f"a balanced state, five when C crosses zero and "
            f"anti-coordination exists.")
    null_name = ("uncoordinated" if anchor == "origin"
                 else "least coordinated")
    out = {null_name: (0.0, 0.0),
           "co-open shifted": (0.7, 0.7),
           "co-closed shifted": (0.7, -0.7)}
    if k >= 4:
        out["coordinated balanced"] = (1.0, 0.0)
    if k >= 5:
        out["anti-coordinated"] = (-1.0, 0.0)
    return out, null_name


ARCHETYPES = {
    "uncoordinated":        (0.0, 0.0),
    "coordinated balanced": (1.0, 0.0),
    "co-open shifted":      (0.7, 0.7),
    "co-closed shifted":    (0.7, -0.7),
    "anti-coordinated":     (-1.0, 0.0),
}


def stationary(A):
    """Long-run occupancy of each state, from the transition matrix."""
    w, v = np.linalg.eig(np.asarray(A, float).T)
    i = int(np.argmin(np.abs(w - 1.0)))
    pi = np.real(v[:, i])
    pi = np.abs(pi)
    return pi / max(pi.sum(), 1e-12)


def label_states_relative(means, A, anchor="auto"):
    """Label the fitted states by their positions relative to a null.

    WHICH NULL depends on the data, and getting this wrong mislabels
    everything.

    ``origin`` takes (0, 0) as the null, which needs no estimating: C = 0 is
    "no more concordant mass than expected" and F - F_exp = 0 is "the mass
    sits where the null puts it".  Both statistics are defined against the
    null, so when the data reaches it this is exact.

    ``min-c`` takes the state with the LOWEST C instead.  On real data C came
    back strictly positive -- 0.000479 at its smallest against a robust sd of
    0.00013, so C = 0 sat 3.7 sds below anything observed across ~11M
    cell-bins.  Neither 'uncoordinated' nor 'anti-coordinated' is reachable
    there, and anchoring on an origin the cloud never approaches labels
    every state by how far the whole cloud sits from it rather than by what
    distinguishes the states.

    ``auto`` compares the nearest state's distance from the origin against
    the spread BETWEEN states.  If the cloud sits further from the origin
    than the states sit from each other, the origin is not a reference the
    data can support and min-c is used.

    Only the SCALE is relative in both cases: offsets are divided by their
    own median size before the five archetypes are matched one-to-one, so
    nothing depends on the units of the plane.  Requires exactly five
    states; with any other K the assignment is not a bijection and the
    labels would be a guess dressed as a result.
    """
    from scipy.optimize import linear_sum_assignment

    mu = np.asarray(means, float)
    occ = stationary(A)
    radii = np.linalg.norm(mu, axis=1)
    pair = [np.linalg.norm(mu[i] - mu[j]) for i in range(len(mu))
            for j in range(i + 1, len(mu))]
    between = float(np.median(pair)) if pair else 0.0
    origin_far = between > 0 and float(radii.min()) > between

    if anchor == "auto":
        anchor = "min-c" if origin_far else "origin"
    if anchor == "origin":
        ref = int(np.argmin(radii))
        base = np.zeros(2)
    elif anchor == "min-c":
        ref = int(np.argmin(mu[:, 0]))
        base = mu[ref]
    else:
        raise ValueError(f"anchor must be auto, origin or min-c, "
                         f"got {anchor!r}")

    arch, null_name = archetypes(mu.shape[0], anchor)
    off = mu - base
    sizes = np.linalg.norm(off, axis=1)
    scale = float(np.median(sizes[sizes > 0])) if (sizes > 0).any() else 0.0
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("fitted states are coincident; nothing to label")
    z = off / scale
    names = list(arch)
    tgt = np.array([arch[n] for n in names], float)
    cost = np.linalg.norm(z[:, None, :] - tgt[None, :, :], axis=2)
    ri, ci = linear_sum_assignment(cost)
    out = [None] * len(names)
    for i, j in zip(ri, ci):
        nm = names[j]
        # With the origin unreachable there is no absolute point to measure
        # "less coordinated than nothing" from, so the two labels that mean
        # exactly that cannot be told apart.  Say so per state rather than
        # let a bijection imply a distinction the data cannot make.
        # With the origin unreachable, 'anti-coordinated' cannot be
        # verified as genuine anti-coordination -- it just means "lowest C
        # present".  The null state does not need the flag: at K = 4 it is
        # NAMED 'least coordinated', which is exactly what it is.
        unreliable = anchor == "min-c" and nm == "anti-coordinated"
        out[i] = {"mean_C": mu[i, 0], "mean_dF": mu[i, 1],
                  "rel_C": z[i, 0], "rel_dF": z[i, 1],
                  "R": float(np.hypot(*z[i])),
                  "theta": float(np.degrees(np.arctan2(z[i, 1], z[i, 0]))),
                  "occupancy": occ[i], "label": nm,
                  "label_identifiable": not unreliable,
                  "match_cost": cost[i, j]}
    return out, {"anchor": anchor, "null_name": null_name,
                 "reference_state": ref, "spread": scale,
                 "between_state_spread": between,
                 "nearest_radius": float(radii.min()),
                 "origin_outside_cloud": bool(origin_far),
                 "null_occupancy": float(occ[ref])}


def select_k(X, lengths, obs, n_cells, k_values, seed=0,
             threads=1, init="spread", verbose=False):
    """Fit each K and keep the lowest BIC."""
    rows, best = [], None
    for k in k_values:
        try:
            m = PlaneHMM(k, seed=seed, init=init).fit(
                X, lengths, obs, n_cells, threads=threads,
                verbose=verbose)
        except (ValueError, np.linalg.LinAlgError) as exc:
            rows.append({"k": k, "loglik": np.nan, "bic": np.nan,
                         "note": str(exc)[:60]})
            continue
        rows.append({"k": k, "loglik": m.loglik, "n_params": m.n_params(),
                     "bic": m.bic(), "note": ""})
        if best is None or m.bic() < best.bic():
            best = m
    return best, pd.DataFrame(rows)


def clr(post, eps=1e-6):
    """Centred log-ratio of the state posteriors.

    Posteriors sum to 1, so one state rising mechanically pushes the others
    down.  Feeding them straight to a test is the compositional trap: a
    change in one state manufactures apparent changes everywhere else.  CLR
    maps the simplex to a space where that is no longer true, at the cost of
    one lost degree of freedom (the rows sum to zero).
    """
    P = np.clip(np.asarray(post, float), eps, None)
    L = np.log(P)
    return L - np.nanmean(L, axis=1, keepdims=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coordination", required=True,
                    help="concordance_coordination.parquet (C)")
    ap.add_argument("--direction-excess", required=True,
                    help="concordance_direction_excess.parquet (F - F_exp)")
    ap.add_argument("--k", default="5",
                    help="Number of states, default 5.  'bic' selects over "
                         "--k-range, but BIC's penalty grows like log(n) "
                         "while the likelihood grows like n: at ~1e7 "
                         "cell-bins the whole penalty is under 2,000 "
                         "against likelihood gains of hundreds of "
                         "thousands, so it keeps buying states to absorb "
                         "tail points and never turns around.  Read its "
                         "table as a diagnostic, not a selector.")
    ap.add_argument("--k-range", default="2,3,4,5,6,7,8",
                    help="Candidate K for BIC selection.")
    ap.add_argument("--fit-cells", type=int, default=0, metavar="N",
                    help="Fit on a random N cells and decode all of them.  "
                         "The transition matrix is shared, so it does not "
                         "need every cell to estimate; decoding is where "
                         "each cell gets its own answer.  0 = use all.")
    ap.add_argument("--label", choices=["relative", "angle"],
                    default="relative",
                    help="'relative' (default) anchors on the most-occupied "
                         "state as the null and matches the other four to "
                         "the archetypes by shape, so nothing depends on "
                         "where the fitted cloud sits or how wide it is.  "
                         "'angle' uses absolute radius and angle, which "
                         "assumes the cloud is centred on the null -- on "
                         "real data it filed all eight states under one "
                         "name.  Needs exactly five states.")
    ap.add_argument("--max-z", default="auto", metavar="Z",
                    dest="clip",
                    help="Treat a bin beyond this |z| on either axis as "
                         "MISSING.  'auto' (default) picks the knee of the "
                         "|z| distribution and reports the ladder it chose "
                         "from; a number pins it; 0 disables.  Gaussian emissions on a "
                         "heavy tail put every state on the tail and the "
                         "transition matrix collapses to runs of one bin.  "
                         "Masked, not clipped: clipping turns the tail into "
                         "a spike at +-z that EM allocates states to just "
                         "as happily -- on a Cauchy-tailed fixture three of "
                         "five states sat exactly on the clip corners.  A "
                         "fixed z, not a percentile: the 99.5th percentile "
                         "of |z| there was still 341.")
    ap.add_argument("--null-anchor", choices=["auto", "origin", "min-c"],
                    default="auto",
                    help="Which point counts as the null when labelling.  "
                         "'origin' is exact when the data reaches C = 0.  "
                         "'min-c' uses the lowest-C state instead, which is "
                         "what real data needed: C came back strictly "
                         "positive, so C = 0 sat 3.7 robust sds below "
                         "anything observed and neither 'uncoordinated' nor "
                         "'anti-coordinated' was reachable.  'auto' picks "
                         "min-c when the cloud sits further from the origin "
                         "than the states sit from each other.")
    ap.add_argument("--threads", type=int, default=0, metavar="N",
                    help="Parallel workers for the E step (0 = one per "
                         "chromosome, capped at the CPU count).  "
                         "Chromosomes are independent given the current "
                         "parameters, so the E step parallelises exactly.  "
                         "Threads rather than processes: the inner work is "
                         "numpy and releases the GIL, and the batches are "
                         "large enough that pickling them to processes "
                         "would cost more than it saves.")
    ap.add_argument("--space", choices=["plane", "log-radial"],
                    default="plane",
                    help="Coordinates the states are fitted in.  "
                         "'log-radial' replaces the radius about the "
                         "cloud's centre by log(1 + r), keeping every angle "
                         "exactly.  Use it when the plane is a dense core "
                         "with arms radiating off it: Gaussian components "
                         "are the wrong shape for that and all settle in "
                         "the core, which is what a real fit did.  Reported "
                         "means are put back on the original plane.")
    ap.add_argument("--init", choices=["spread", "kmeans"],
                    default="spread",
                    help="How the state means are seeded.  'spread' "
                         "(default) re-weights each point by the inverse of "
                         "its local density first, so low-density ARMS of "
                         "the plane are as visible to the initialiser as a "
                         "dense core.  Plain 'kmeans' seeds by density and "
                         "puts every centre in the core when one exists -- "
                         "which is what a real fit did, leaving three "
                         "states within six units of each other while the "
                         "cloud spanned a hundred.  Only the STARTING "
                         "positions differ; the fit uses every point at its "
                         "true weight either way.")
    ap.add_argument("--qc", action="store_true",
                    help="Write hmm_qc.<fmt> and "
                         "hmm_qc_autocorrelation.tsv: is the plane "
                         "multimodal, is there autocorrelation along the "
                         "genome for the transition prior to use, where the "
                         "mask fell, the transition matrix, run lengths, "
                         "and the EM trace.")
    ap.add_argument("--qc-max-lag", type=int, default=20)
    ap.add_argument("--plot-format", default="pdf",
                    choices=["pdf", "png", "svg"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r-cut", default="auto",
                    help="Radius below which a fitted state is called "
                         "uncoordinated.  'auto' (default) uses a fraction "
                         "of the largest fitted radius, because an absolute "
                         "cut assumes the standardising scale is the NULL "
                         "scale -- true only when most bins are null.  Pass "
                         "a number to override.")
    ap.add_argument("--sector", type=float, default=30.0)
    ap.add_argument("--samples", default=None, metavar="TSV",
                    help="Per-cell annotation table, for "
                         "--concordance-batch-correction and --colour-by.")
    ap.add_argument("--cell-column", default="cell",
                    help="Cell-id column in --samples.")
    ap.add_argument("--concordance-batch-column", default=None,
                    metavar="COLS",
                    help="Batch variable(s) in --samples, comma-separated "
                         "(e.g. 'exp_batch,lib_batch').")
    ap.add_argument("--concordance-batch-correction",
                    choices=["none", "center", "zscore", "combat"],
                    default="none",
                    help="Per-bin batch correction of C and F-F_exp before "
                         "fitting, via the pipeline's own "
                         "batch_correction.correct_and_report -- the "
                         "identical transform the component UMAPs get.  It "
                         "is OFF by default and that is not an oversight: "
                         "concordance_coordination.parquet is written "
                         "inside step 5, BEFORE the pipeline applies "
                         "--concordance-batch-correction to the shrunk "
                         "matrix, so no per-bin batch correction has "
                         "touched these files however the pipeline was run "
                         "(--batch-correction is a different stage and "
                         "does reach them, indirectly, by correcting the "
                         "accessibility matrix that seeds the RWR graph).  "
                         "Turning it on here makes "
                         "the HMM consistent with the rest of the run.  It "
                         "is not free: every method removes each batch's "
                         "per-bin MEAN, so wherever a batch is one "
                         "biological group the correction deletes that "
                         "group's biology -- see the confounding table this "
                         "prints.  'combat' additionally rescales per "
                         "(batch, bin) by a shrunken variance ratio, which "
                         "on a non-negative quantity like C can push values "
                         "below zero; the run reports how many.")
    ap.add_argument("--center-cells", choices=["none", "median"],
                    default="none",
                    help="Subtract each cell's own median, per axis, before "
                         "fitting.  A state is a run of bins INSIDE a cell, "
                         "so only within-cell spread is segmentable; "
                         "anything that slides a whole cell across the plane "
                         "-- coverage, GpC enzyme efficiency, RWR "
                         "connectivity -- is between-cell variation that "
                         "Gaussian components fit just as readily as "
                         "domains.  Use it when the QC reports a high "
                         "between-cell fraction.  It is compositional: a "
                         "cell whose coordination genuinely differs "
                         "everywhere has that difference subtracted, so it "
                         "answers 'where are this cell's domains', not 'is "
                         "this cell more coordinated'.")
    ap.add_argument("--colour-by", default=None, metavar="COLUMN",
                    help="Colour QC panel (a) by this --samples column "
                         "instead of point density.  A categorical column "
                         "gets one colour per level with that level's "
                         "median marked; a numeric one gets a colour bar.  "
                         "This is how you find out what an arm of the plane "
                         "IS: if each arm is one batch, or one cell type, "
                         "or one coverage band, no choice of K or emission "
                         "family is the answer.")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    Cdf = pd.read_parquet(args.coordination)
    Fdf = pd.read_parquet(args.direction_excess).reindex(
        index=Cdf.index, columns=Cdf.columns)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tab = (load_cell_table(args.samples, args.cell_column, Cdf.index)
           if args.samples else None)
    if args.colour_by and tab is None:
        sys.exit("--colour-by needs --samples")
    cbc = args.concordance_batch_correction
    if cbc != "none":
        if tab is None or not args.concordance_batch_column:
            sys.exit("--concordance-batch-correction needs --samples and "
                     "--concordance-batch-column")
        bl = batch_series(tab, args.concordance_batch_column,
                          "--concordance-batch-column", args.samples)
        n_lab = int(bl.notna().all(axis=1).sum()
                    if isinstance(bl, pd.DataFrame) else bl.notna().sum())
        print(f"  batch: {cbc} on C and F-F_exp over "
              f"{args.concordance_batch_column} "
              f"({n_lab:,}/{len(Cdf):,} cells labelled)", file=sys.stderr)
        if args.colour_by and args.colour_by in getattr(tab, "columns", []):
            key = (bl if isinstance(bl, pd.Series)
                   else bl.astype(str).agg("|".join, axis=1))
            conf = batch_group_confounding(key, tab[args.colour_by])
            conf.to_csv(out / "hmm_batch_confounding.tsv", sep="\t",
                        index=False)
            worst = conf.iloc[0]
            print(f"  batch vs {args.colour_by}: worst level "
                  f"{worst['batch']!r} is {worst['share']:.0%} "
                  f"{worst['top_group']!r} ({int(worst['n_cells'])} cells)",
                  file=sys.stderr)
            if float(worst["share"]) > 0.9:
                print(f"  WARNING batch and {args.colour_by} are confounded "
                      f"there.  Removing that batch's per-bin mean removes "
                      f"that group's biology with it; the correction cannot "
                      f"tell them apart.  See "
                      f"hmm_batch_confounding.tsv.", file=sys.stderr)
        neg_before = float(np.mean(Cdf.to_numpy(float)[
            np.isfinite(Cdf.to_numpy(float))] < 0))
        from nome_hic_concordance import batch_correction as _bc
        Cdf = _bc.correct_and_report(Cdf, bl, cbc, args.seed, "coordination")
        Fdf = _bc.correct_and_report(Fdf, bl, cbc, args.seed,
                                     "direction_excess")
        cv = Cdf.to_numpy(float)
        neg_after = float(np.mean(cv[np.isfinite(cv)] < 0))
        print(f"  C below zero: {neg_before:.2%} -> {neg_after:.2%}.  C is "
              f"a covariance-scale magnitude; a location-scale correction "
              f"is free to move it across zero, and C = 0 is the point the "
              f"labelling calls the null.", file=sys.stderr)
    else:
        print("  batch: none applied here.  "
              "concordance_coordination.parquet is written inside step 5, "
              "BEFORE the pipeline's --concordance-batch-correction (which "
              "acts on the shrunk matrix and writes "
              "concordance_batchcorrected.parquet), so no per-bin batch "
              "correction has touched these values however the pipeline "
              "was run.  --batch-correction is a different stage and DID "
              "reach them, indirectly: it corrects the accessibility "
              "matrix at step 3.5, which changes the k-NN graph and so "
              "what RWR imputes.", file=sys.stderr)

    C, dF = Cdf.to_numpy(float), Fdf.to_numpy(float)
    if args.center_cells == "median":
        C, dF, cinfo = center_cells(C, dF)
        print(f"  centred each cell on its own median: typical shift "
              f"{cinfo['median_shift_C']:.6f} (C) and "
              f"{cinfo['median_shift_dF']:.6f} (F-F_exp); the cell centres "
              f"themselves were spread over "
              f"{cinfo['spread_of_cell_centres_C']:.6f} and "
              f"{cinfo['spread_of_cell_centres_dF']:.6f} robust sd -- that "
              f"is the between-cell variation now removed from the fit.",
              file=sys.stderr)

    _mz = args.clip
    if isinstance(_mz, str) and _mz.lower() != "auto":
        try:
            _mz = float(_mz)
        except ValueError:
            sys.exit(f"--max-z must be a number or 'auto', got {_mz!r}")
    X, lengths, obs, order, sc = build_sequences(
        C, dF, list(Cdf.columns), clip=_mz, space=args.space)
    import os
    n_chrom_total = len(lengths) // max(C.shape[0], 1)
    if not args.threads:
        args.threads = max(1, min(n_chrom_total, os.cpu_count() or 1))
    bins = list(np.asarray(Cdf.columns)[order])
    n_cells, n_bins = C.shape
    print(f"  {n_cells:,} cells x {n_bins:,} bins; "
          f"{int(obs.sum()):,} observed ({obs.mean():.1%}); "
          f"{len(lengths):,} chains (cell x chromosome); "
          f"{args.threads} thread(s)", file=sys.stderr)
    if "max_abs_z" in sc:
        print(f"  standardised spread about the median: max |z| = "
              f"{sc['max_abs_z']:,.1f}, {sc['frac_beyond_10']:.2%} beyond "
              f"10; median sits at "
              f"({sc['median_z'][0]:+.2f}, {sc['median_z'][1]:+.2f}) "
              f"relative to C = 0", file=sys.stderr)
        if sc["frac_beyond_10"] > 0.01:
            print(f"  NOTE {sc['frac_beyond_10']:.1%} of the plane sits "
                  f"beyond |z| = 10.  Gaussian emissions will chase that "
                  f"tail unless it is clipped.", file=sys.stderr)
    if "max_z_auto" in sc:
        print(f"  --max-z auto chose |z| = {sc['max_z_auto']:g} from the "
              f"knee of the distribution ("
              + ", ".join(f"z={r['z']:g}:{r['frac_kept']:.1%}"
                          for r in sc["max_z_ladder"][:8]) + ")",
              file=sys.stderr)
    if "mask_beyond_z" in sc:
        print(f"  masked {sc['n_masked_extreme']:,} bins beyond |z| = "
              f"{sc['mask_beyond_z']:g} "
              f"({sc['frac_masked_extreme']:.2%} of observed)",
              file=sys.stderr)

    fit_X, fit_len, fit_obs, n_fit_cells = X, lengths, obs, n_cells
    if args.fit_cells and args.fit_cells < n_cells:
        rng = np.random.default_rng(args.seed)
        pick = np.sort(rng.choice(n_cells, args.fit_cells, replace=False))
        per_cell = len(lengths) // n_cells
        keep_rows, keep_len = [], []
        starts = np.r_[0, np.cumsum(lengths)]
        for c in pick:
            for j in range(per_cell):
                i = c * per_cell + j
                keep_rows.append(np.arange(starts[i], starts[i + 1]))
                keep_len.append(lengths[i])
        idx = np.concatenate(keep_rows)
        fit_X, fit_len, fit_obs = X[idx], np.array(keep_len), obs[idx]
        n_fit_cells = args.fit_cells
        print(f"  fitting on {args.fit_cells:,} cells "
              f"({int(fit_obs.sum()):,} observed bins)", file=sys.stderr)

    if str(args.k).lower() == "bic":
        ks = [int(v) for v in args.k_range.split(",") if v.strip()]
        model, table = select_k(fit_X, fit_len, fit_obs,
                                n_fit_cells, ks, args.seed,
                                threads=args.threads,
                                init=args.init)
        table.to_csv(out / "hmm_bic.tsv", sep="\t", index=False)
        if model is None:
            sys.exit("no K converged; try a narrower --k-range")
        print("  BIC: " + "  ".join(
            f"k={int(r.k)}:{r.bic:,.0f}" for r in table.itertuples()
            if np.isfinite(r.bic)), file=sys.stderr)
        print(f"  selected k = {model.k}", file=sys.stderr)
    else:
        model = PlaneHMM(int(args.k), seed=args.seed,
                         init=args.init).fit(
            fit_X, fit_len, fit_obs, n_fit_cells, threads=args.threads)
        print(f"  fixed k = {model.k}  (BIC {model.bic():,.0f})",
              file=sys.stderr)

    if sc.get("space") == "log-radial":
        model.means_warped = model.means.copy()
        model.means = unwarp_log_radial(model.means,
                                        np.array(sc["warp_centre"]))
        print(f"  fitted in log-radial space about "
              f"({sc['warp_centre'][0]:+.2f}, {sc['warp_centre'][1]:+.2f}); "
              f"means reported back on the original plane", file=sys.stderr)
    rel_info = {}
    if args.label == "relative":
        try:
            info, rel_info = label_states_relative(
                model.means, model.A, anchor=args.null_anchor)
        except ValueError as exc:
            sys.exit(f"relative labelling failed: {exc}\n"
                     f"  use --k 3 when the coordinated states are all "
                     f"directional, --k 4 to add a balanced state, --k 5 "
                     f"when C crosses zero; "
                     f"--label angle uses the absolute rule instead, and "
                     f"its labels should be read with suspicion.")
    else:
        _rc = args.r_cut
        if not isinstance(_rc, str) or _rc.lower() != "auto":
            try:
                _rc = float(_rc)
            except ValueError:
                sys.exit(f"--r-cut must be a number or 'auto', got {_rc!r}")
        info = label_states(model.means, _rc, args.sector)
    lab = pd.DataFrame(info)
    lab.insert(0, "state", range(model.k))
    lab["self_transition"] = np.diag(model.A)
    lab["expected_run_bins"] = 1.0 / np.maximum(1 - np.diag(model.A), 1e-9)
    if rel_info:
        if rel_info["anchor"] == "origin":
            print(f"  labelling: against the null at the origin (C = 0, "
                  f"F - F_exp = 0); offsets scaled by "
                  f"{rel_info['spread']:.3g}", file=sys.stderr)
        else:
            print(f"  labelling: against state "
                  f"{rel_info['reference_state']}, the lowest-C state, "
                  f"taken as the null; offsets scaled by "
                  f"{rel_info['spread']:.3g}", file=sys.stderr)
        if rel_info["origin_outside_cloud"] and model.k >= 5:
            print(f"  NOTE the origin is OUTSIDE the fitted cloud: the "
                  f"nearest state sits {rel_info['nearest_radius']:.3g} "
                  f"from C = 0 while the states sit only "
                  f"{rel_info['between_state_spread']:.3g} from each other.  "
                  f"C never reaching zero means neither 'uncoordinated' nor "
                  f"'anti-coordinated' exists in this data.\n"
                  f"  Those two labels are therefore NOT IDENTIFIABLE and "
                  f"are flagged as such in hmm_states.tsv: both mean 'less "
                  f"coordinated than the rest', and separating them needs "
                  f"an absolute zero the data never reaches.  Re-run "
                  f"with --k 4: with C positive everywhere there is no "
                  f"anti-coordinated state to find, and four archetypes "
                  f"(least coordinated, balanced, co-open, co-closed) is "
                  f"the honest model.\n"
                  f"  Worth checking why C never reaches zero: O and E are "
                  f"fitted with SEPARATE Dirichlet priors, and if the "
                  f"linked pool's prior is more concordant than the "
                  f"distance-matched one, every bin inherits that gap as a "
                  f"positive offset.  A contrast between arms at the same "
                  f"bin cancels it; an absolute reading of C does not.",
                  file=sys.stderr)
    lab.to_csv(out / "hmm_states.tsv", sep="\t", index=False)
    print("  fitted states:", file=sys.stderr)
    for r in lab.itertuples():
        rel = (f"  rel=({r.rel_C:+.2f},{r.rel_dF:+.2f})"
               if hasattr(r, "rel_C") else "")
        occ = f"  occ={r.occupancy:.1%}" if hasattr(r, "occupancy") else ""
        print(f"    {r.state}  mean=({r.mean_C:+.3g}, {r.mean_dF:+.3g})"
              f"{rel}{occ}  -> {r.label}  "
              f"(runs ~{r.expected_run_bins:.0f} bins)", file=sys.stderr)
    runs = float(np.mean(lab["expected_run_bins"]))
    if runs < 3.0:
        print(f"  WARNING mean run length is {runs:.1f} bins.  An HMM whose "
              f"states switch almost every bin has found no domains -- the "
              f"spatial prior is the only reason to use one, so this fit is "
              f"not usable.  Usually a heavy tail: lower --clip.",
              file=sys.stderr)
    spread = float(np.max(np.linalg.norm(
        model.means - model.means.mean(0), axis=1)))
    if spread < 1e-6 or (len(model.means) > 1 and spread
                         < 0.05 * float(np.linalg.norm(model.means.mean(0)))):
        print(f"  WARNING the fitted states are nearly coincident (spread "
              f"{spread:.3g} about a centroid at "
              f"{np.linalg.norm(model.means.mean(0)):.3g}).  EM has put "
              f"every state on one cluster; the labels below describe that "
              f"cluster, not five states.", file=sys.stderr)
    dup = lab["label"].duplicated(keep=False)
    if dup.any():
        print(f"  NOTE {int(dup.sum())} states share a taxonomy label.  The "
              f"data wanted more structure inside a sector than the five "
              f"names cover; report the fitted means, not just the labels.",
              file=sys.stderr)

    if rel_info.get("origin_outside_cloud") and model.k == 4:
        print(f"  NOTE the origin is outside the fitted cloud, so the null "
              f"state is named 'least coordinated' rather than "
              f"'uncoordinated': C is positive everywhere, so this state "
              f"has the least coordination present, not none.",
              file=sys.stderr)
    path, post = model.decode(X, lengths, obs, n_cells)
    path = np.where(obs, path, -1).reshape(n_cells, -1)
    pd.DataFrame(path, index=Cdf.index, columns=bins).to_parquet(
        out / "concordance_state.parquet")
    P = post.reshape(n_cells, len(bins), model.k)
    Z = clr(post).reshape(n_cells, len(bins), model.k)
    cols, flat_p, flat_z = [], [], []
    for s in range(model.k):
        cols += [f"{b}|s{s}" for b in bins]
        flat_p.append(P[:, :, s])
        flat_z.append(Z[:, :, s])
    pd.DataFrame(np.hstack(flat_p), index=Cdf.index,
                 columns=cols).to_parquet(
        out / "concordance_state_posterior.parquet")
    pd.DataFrame(np.hstack(flat_z), index=Cdf.index,
                 columns=cols).to_parquet(
        out / "concordance_state_posterior_clr.parquet")
    if args.qc:
        # Taken from the masker, not re-derived: re-deriving it compared
        # a plane-unit threshold against whatever space X is in, so under
        # --space log-radial it silently selected nothing and the red
        # points that show what the cut removed never appeared.
        masked_flag = sc.get("masked_rows",
                             np.zeros(X.shape[0], bool))
        cell_mu, var_frac = cell_summaries(X, obs, n_cells,
                                           sc.get("warp_centre"))
        pd.DataFrame({"axis": ["C", "dF"],
                      "between_cell_fraction": [var_frac["C"],
                                                var_frac["dF"]]}).to_csv(
            out / "hmm_qc_variance.tsv", sep="\t", index=False)
        print(f"  QC: {var_frac['C']:.1%} of C's spread and "
              f"{var_frac['dF']:.1%} of F-F_exp's is BETWEEN cells",
              file=sys.stderr)
        if max(var_frac["C"], var_frac["dF"]) > 0.5:
            print("  QC: most of the plane is between-cell variation, so "
                  "the components are fitting cell identity, not genomic "
                  "domains.  A different K will not change that, and "
                  "neither will a per-bin batch correction -- both act on "
                  "the axis that is not carrying the structure.  Segment "
                  "each cell against its OWN centre (subtract the per-cell "
                  "median before fitting) if the question is domains, or "
                  "cluster the cell means directly if the question is cell "
                  "types.", file=sys.stderr)
        acf = genome_autocorrelation(X, lengths, obs, n_cells,
                                     max_lag=args.qc_max_lag)
        pd.DataFrame(acf).to_csv(out / "hmm_qc_autocorrelation.tsv",
                                 sep="\t", index=False)
        a1 = acf["C_within_cell"][0]
        a1p = acf["C_pooled"][0]
        print(f"  QC: lag-1 autocorrelation of C is {a1:+.3f} within cells "
              f"and {a1p:+.3f} on the pooled track", file=sys.stderr)
        if "mask_beyond_z" in sc and sc.get("frac_masked_extreme", 0) > 0.01:
            pl = plane_sample(X, obs, sc.get("warp_centre"), seed=3)
            good_r = np.linalg.norm(pl - np.median(pl, axis=0), axis=1)
            reach = float(np.percentile(good_r, 99.5))
            if reach > 0.8 * sc["mask_beyond_z"]:
                print(f"  QC: the retained cloud reaches {reach:.0f} from "
                      f"its median and the mask sits at "
                      f"{sc['mask_beyond_z']:.0f} -- the cut is INSIDE the "
                      f"structure, not beyond it.  A fit that only sees the "
                      f"core can only place states in the core.  Re-run "
                      f"with --max-z 0 and compare panel (a): the red "
                      f"points are what the mask removed.", file=sys.stderr)
        if np.isfinite(a1) and abs(a1) < 0.05:
            msg = ("  QC: consecutive bins within a cell barely agree, so "
                   "the transition prior has nothing to work with -- an HMM "
                   "cannot help here whatever K is.")
            if np.isfinite(a1p) and abs(a1p) > 0.2:
                msg += ("\n  QC: the POOLED track is autocorrelated though, "
                        "so the domain structure is real and buried under "
                        "per-cell sampling noise.  Learn the states on the "
                        "pooled track and decode per cell, or aggregate "
                        "bins before segmenting.")
            else:
                msg += ("\n  QC: the pooled track is not autocorrelated "
                        "either, so there is no domain structure at this "
                        "bin size to find.")
            print(msg, file=sys.stderr)
        try:
            qc_path = out / f"hmm_qc.{args.plot_format}"
            qc_figure(qc_path, X, obs, model, acf, sc.get("mask_beyond_z",
                      float("nan")), sc.get("max_z_ladder", []),
                      getattr(model, "trace", []),
                      [d["label"] for d in info],
                      f"HMM QC: k={model.k}, "
                      f"{int(obs.sum()):,} observed bins",
                      masked=masked_flag,
                      warp_centre=sc.get("warp_centre"),
                      colour=(point_colour(tab, args.colour_by, n_bins)
                              if args.colour_by else None),
                      cell_means=cell_mu)
            print(f"  QC figure -> {qc_path.name}", file=sys.stderr)
        except Exception as exc:
            print(f"  QC figure failed: {exc}", file=sys.stderr)
    print(f"  -> concordance_state.parquet, "
          f"concordance_state_posterior{{,_clr}}.parquet, hmm_states.tsv",
          file=sys.stderr)
    print(f"  embed the CLR matrix, not the raw posteriors: they sum to 1, "
          f"so one state rising pushes the rest down and the embedding "
          f"would read that as structure.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
