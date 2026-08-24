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

    def __init__(self, k, seed=0):
        self.k = int(k)
        self.seed = int(seed)

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

    # -- one chain -------------------------------------------------------
    def _fb(self, logB, lA, lpi):
        n = logB.shape[0]
        la = np.empty_like(logB)
        lb = np.zeros_like(logB)
        la[0] = lpi + logB[0]
        for t in range(1, n):
            la[t] = logB[t] + _lse(la[t - 1][:, None] + lA, 0)
        for t in range(n - 2, -1, -1):
            lb[t] = _lse(lA + (logB[t + 1] + lb[t + 1])[None, :], 1)
        ll = _lse(la[-1], 0)
        lg = la + lb - ll
        return la, lb, lg, ll

    def fit(self, X, lengths, obs, n_iter=40, tol=1e-4, verbose=False):
        rng = np.random.default_rng(self.seed)
        good = X[obs]
        if len(good) < self.k * 10:
            raise ValueError(f"only {len(good)} observed bins for k={self.k}")
        # k-means start.  Per-axis quantiles were the obvious thing and are
        # wrong: (5th pct of C, 5th pct of dF) through (95th, 95th) all lie
        # on the cloud's MAIN DIAGONAL, so a state at (+C, -dF) never gets
        # a mean anywhere near itself and EM lands in a bad optimum.
        # k-means initialises in the plane, which is where the states are.
        sub = good
        if len(sub) > 200_000:
            sub = sub[rng.choice(len(sub), 200_000, replace=False)]
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=self.k, n_init=10,
                        random_state=self.seed).fit(sub)
            self.means = km.cluster_centers_.astype(float)
        except Exception:                       # no sklearn: farthest-point
            self.means = _farthest_point_init(sub, self.k, rng)
        self.means = self.means + rng.normal(0, 1e-6, self.means.shape)
        base = np.cov(good.T) + np.eye(2) * 1e-6
        self.covs = np.stack([base.copy() for _ in range(self.k)])
        A = np.full((self.k, self.k), 0.05 / max(self.k - 1, 1))
        np.fill_diagonal(A, 0.95)
        pi = np.full(self.k, 1.0 / self.k)

        prev = -np.inf
        for it in range(n_iter):
            lA, lpi = np.log(A), np.log(pi)
            logB = self._log_b(X, obs)
            g_sum = np.zeros((self.k,))
            gx = np.zeros((self.k, 2))
            gxx = np.zeros((self.k, 2, 2))
            xi = np.zeros((self.k, self.k))
            pi_acc = np.zeros(self.k)
            total = 0.0
            for a, b in _spans(lengths):
                lb_, la_, lg, ll = self._fb(logB[a:b], lA, lpi)
                total += ll
                g = np.exp(lg)
                pi_acc += g[0]
                m = obs[a:b]
                gm = g[m]
                xm = X[a:b][m]
                g_sum += gm.sum(0)
                gx += gm.T @ xm
                for s in range(self.k):
                    d = xm - self.means[s]
                    gxx[s] += (d * gm[:, s:s + 1]).T @ d
                lg2 = (lb_[:-1][:, :, None] + lA[None]
                       + (logB[a:b][1:] + la_[1:])[:, None, :])
                lg2 -= _lse(lg2.reshape(lg2.shape[0], -1), 1)[:, None, None]
                xi += np.exp(lg2).sum(0)
            self.means = gx / np.maximum(g_sum[:, None], 1e-12)
            for s in range(self.k):
                self.covs[s] = (gxx[s] / max(g_sum[s], 1e-12)
                                + np.eye(2) * 1e-6)
            A = xi / np.maximum(xi.sum(1, keepdims=True), 1e-12)
            pi = pi_acc / max(pi_acc.sum(), 1e-12)
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
    def decode(self, X, lengths, obs):
        lA, lpi = np.log(self.A), np.log(self.pi)
        logB = self._log_b(X, obs)
        path = np.full(X.shape[0], -1, int)
        post = np.full((X.shape[0], self.k), np.nan)
        for a, b in _spans(lengths):
            lb_, la_, lg, _ = self._fb(logB[a:b], lA, lpi)
            post[a:b] = np.exp(lg)
            n = b - a
            dp = np.empty((n, self.k))
            bp = np.zeros((n, self.k), int)
            dp[0] = lpi + logB[a]
            for t in range(1, n):
                m = dp[t - 1][:, None] + lA
                bp[t] = m.argmax(0)
                dp[t] = m.max(0) + logB[a + t]
            p = np.empty(n, int)
            p[-1] = dp[-1].argmax()
            for t in range(n - 2, -1, -1):
                p[t] = bp[t + 1][p[t + 1]]
            path[a:b] = p
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


def build_sequences(C, dF, bins, scale_c=None, scale_f=None):
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
    X = np.where(np.isfinite(X), X, 0.0)
    lengths = list(per_chrom) * C.shape[0]
    return X, np.array(lengths, int), obs, order, {"scale_c": sc,
                                                   "scale_f": sf}


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


def select_k(X, lengths, obs, k_values, seed=0, verbose=False):
    """Fit each K and keep the lowest BIC."""
    rows, best = [], None
    for k in k_values:
        try:
            m = PlaneHMM(k, seed=seed).fit(X, lengths, obs, verbose=verbose)
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
    ap.add_argument("--k", default="bic",
                    help="'bic' to select over --k-range (default), or a "
                         "fixed integer such as 5 to pin the taxonomy's "
                         "five states.")
    ap.add_argument("--k-range", default="2,3,4,5,6,7,8",
                    help="Candidate K for BIC selection.")
    ap.add_argument("--fit-cells", type=int, default=0, metavar="N",
                    help="Fit on a random N cells and decode all of them.  "
                         "The transition matrix is shared, so it does not "
                         "need every cell to estimate; decoding is where "
                         "each cell gets its own answer.  0 = use all.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--r-cut", default="auto",
                    help="Radius below which a fitted state is called "
                         "uncoordinated.  'auto' (default) uses a fraction "
                         "of the largest fitted radius, because an absolute "
                         "cut assumes the standardising scale is the NULL "
                         "scale -- true only when most bins are null.  Pass "
                         "a number to override.")
    ap.add_argument("--sector", type=float, default=30.0)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)

    Cdf = pd.read_parquet(args.coordination)
    Fdf = pd.read_parquet(args.direction_excess).reindex(
        index=Cdf.index, columns=Cdf.columns)
    C, dF = Cdf.to_numpy(float), Fdf.to_numpy(float)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X, lengths, obs, order, sc = build_sequences(C, dF, list(Cdf.columns))
    bins = list(np.asarray(Cdf.columns)[order])
    n_cells, n_bins = C.shape
    print(f"  {n_cells:,} cells x {n_bins:,} bins; "
          f"{int(obs.sum()):,} observed ({obs.mean():.1%}); "
          f"{len(lengths):,} chains (cell x chromosome)", file=sys.stderr)

    fit_X, fit_len, fit_obs = X, lengths, obs
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
        print(f"  fitting on {args.fit_cells:,} cells "
              f"({int(fit_obs.sum()):,} observed bins)", file=sys.stderr)

    if str(args.k).lower() == "bic":
        ks = [int(v) for v in args.k_range.split(",") if v.strip()]
        model, table = select_k(fit_X, fit_len, fit_obs, ks, args.seed)
        table.to_csv(out / "hmm_bic.tsv", sep="\t", index=False)
        if model is None:
            sys.exit("no K converged; try a narrower --k-range")
        print("  BIC: " + "  ".join(
            f"k={int(r.k)}:{r.bic:,.0f}" for r in table.itertuples()
            if np.isfinite(r.bic)), file=sys.stderr)
        print(f"  selected k = {model.k}", file=sys.stderr)
    else:
        model = PlaneHMM(int(args.k), seed=args.seed).fit(
            fit_X, fit_len, fit_obs)
        print(f"  fixed k = {model.k}  (BIC {model.bic():,.0f})",
              file=sys.stderr)

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
    _eff = (_rc if not isinstance(_rc, str)
            else 0.35 * float(lab["R"].max()))
    print(f"  labelling radius cut: {_eff:.2f}"
          + ("  (auto: 35% of the largest fitted radius)"
             if isinstance(_rc, str) else "  (given)"), file=sys.stderr)
    lab.to_csv(out / "hmm_states.tsv", sep="\t", index=False)
    print("  fitted states:", file=sys.stderr)
    for r in lab.itertuples():
        print(f"    {r.state}  ({r.mean_C:+.2f}, {r.mean_dF:+.2f})  "
              f"R={r.R:.2f} theta={r.theta:+.0f}  -> {r.label}  "
              f"(runs ~{r.expected_run_bins:.0f} bins)", file=sys.stderr)
    dup = lab["label"].duplicated(keep=False)
    if dup.any():
        print(f"  NOTE {int(dup.sum())} states share a taxonomy label.  The "
              f"data wanted more structure inside a sector than the five "
              f"names cover; report the fitted means, not just the labels.",
              file=sys.stderr)

    path, post = model.decode(X, lengths, obs)
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
    print(f"  -> concordance_state.parquet, "
          f"concordance_state_posterior{{,_clr}}.parquet, hmm_states.tsv",
          file=sys.stderr)
    print(f"  embed the CLR matrix, not the raw posteriors: they sum to 1, "
          f"so one state rising pushes the rest down and the embedding "
          f"would read that as structure.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
