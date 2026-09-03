"""Section B.7 acceptance criteria for the Part B rank-1 test.

These are not optional plots.  Each answers a question that, unanswered, makes
the headline unreadable:

* **B.7.1 coverage flatness** decides ``error_convention``.  A trend in rho
  with coverage means the measurement-error correction has the wrong sign or
  magnitude; the design says pick the convention from THIS, not by argument.
  ``D`` should already be flat because it needs no correction at all -- if it
  is not, the fault is upstream in A.3.4 and no choice here repairs it.
* **B.7.2 distance decay** separates a real second axis from a rescaled shadow
  of the first: rho2 tracking rho1's profile exactly is what an artifact looks
  like.
* **B.7.3 conversion association** checks the A.3.5 threshold.  Incomplete
  conversion inflates both marks at both ends of one molecule, which survives
  a within-cell mate shuffle, so the null cannot see it and only this can.
* **B.7.4 null calibration** is the licence for every rank-2 call.  A
  non-uniform p-value QQ under simulated rank-1 truth invalidates all of them.
* **B.7.5 mate-shuffle sanity** is the floor: rho1 must collapse when mates are
  reassigned, or rho1 is measuring something other than coordination.
* **B.7.6 / B.7.7 ablations** justify the two defaults that are least obvious
  (``--gch-norm location`` and ``--min-hcg-sites 1``) in the paper.

Everything here is pure: it takes arrays or frames and returns frames.  The
figures live in ``scripts/rank1_diagnostics.py``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import rank1_joint as rj

logger = logging.getLogger(__name__)

#: A trend this large across the coverage range is called out.  Not a hard
#: threshold: it is a flag on a panel a human still has to read.
FLATNESS_TOL = 0.10


def _fit(tA, tB, vA, vB, w=None) -> dict:
    """rho1, rho2 and D for one bundle of contacts, or NaNs if degenerate."""
    try:
        m = rj.moments(rj.accumulate(rj.new_acc(), tA, tB, vA, vB, w))
        cn = rj.canonical(rj.whiten(m["S_AA"], m["S_BB"], m["S_AB"],
                                    m["V_A"], m["V_B"])["R"])
        return {"rho1": cn["rho1"], "rho2": cn["rho2"],
                "D": rj.det_statistic(m["S_AB"]), "N_eff": m["N_eff"]}
    except (ValueError, np.linalg.LinAlgError):
        return {"rho1": np.nan, "rho2": np.nan, "D": np.nan, "N_eff": np.nan}


def _arrays(units: pd.DataFrame):
    return (units[["A_gch", "A_hcg"]].to_numpy(),
            units[["B_gch", "B_hcg"]].to_numpy(),
            units[["A_v_gch", "A_v_hcg"]].to_numpy(),
            units[["B_v_gch", "B_v_hcg"]].to_numpy())


def _trend(x, y) -> float:
    """Change in ``y`` across the range of ``x``, from an OLS slope.

    Reported as a total change rather than a slope so the number is comparable
    across statistics on different scales, which is the whole point of putting
    rho1, rho2 and D on one panel.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0:
        return np.nan
    slope = np.polyfit(x[ok], y[ok], 1)[0]
    return float(slope * np.ptp(x[ok]))


# --------------------------------------------------------------- B.7.1 ------

def coverage_flatness(units: pd.DataFrame, n_bins: int = 10,
                      min_contacts: int = 50) -> pd.DataFrame:
    """rho1, rho2 and D recomputed within joint ``(n_gch, n_hcg)`` deciles.

    The deciles are of the contact's TOTAL site count across both ends, which
    is what the B.2 weight responds to.  Stratifying on one end's depth alone
    would leave the other free to vary within a stratum and blur the trend the
    diagnostic exists to see.

    Rising with coverage means under-corrected, falling means over-corrected,
    flat means the convention is right.  Read ``D`` first: it needs no
    correction, so a trend there is an upstream normalization problem (A.3.4)
    that changing ``error_convention`` will not fix.
    """
    if not len(units):
        return pd.DataFrame()
    u = units.copy()
    depth = (u["A_n_gch"] + u["A_n_hcg"] + u["B_n_gch"] + u["B_n_hcg"]).to_numpy()
    u["_depth"] = depth
    try:
        u["decile"] = pd.qcut(depth, n_bins, labels=False, duplicates="drop")
    except ValueError:
        u["decile"] = 0

    rows = []
    for d, sub in u.groupby("decile", sort=True):
        if len(sub) < min_contacts:
            continue
        tA, tB, vA, vB = _arrays(sub)
        r = _fit(tA, tB, vA, vB)
        r.update(decile=int(d), n_contacts=len(sub),
                 mean_depth=float(sub["_depth"].mean()),
                 mean_n_gch=float((sub["A_n_gch"] + sub["B_n_gch"]).mean() / 2),
                 mean_n_hcg=float((sub["A_n_hcg"] + sub["B_n_hcg"]).mean() / 2))
        rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        out.attrs["trend"] = {k: _trend(out["mean_depth"], out[k])
                              for k in ("rho1", "rho2", "D")}
    return out


def flatness_verdict(table: pd.DataFrame, tol: float = FLATNESS_TOL) -> dict:
    """Turn a coverage-flatness table into the B.4 recommendation.

    Deliberately a recommendation and not an automatic switch: the convention
    changes every number downstream, and a diagnostic run on one chromosome
    should not silently re-key the whole analysis.
    """
    tr = table.attrs.get("trend", {}) if len(table) else {}
    out = {"trend_" + k: tr.get(k, np.nan) for k in ("rho1", "rho2", "D")}
    d_tr, r_tr = tr.get("D", np.nan), tr.get("rho1", np.nan)
    if np.isfinite(d_tr) and abs(d_tr) > tol:
        out["verdict"] = "upstream_normalization"
        out["message"] = (
            f"D trends by {d_tr:+.3f} across the coverage range.  D needs no "
            f"measurement-error correction, so this is an A.3.4 normalization "
            f"problem upstream; changing error_convention will not fix it.")
    elif not np.isfinite(r_tr):
        out["verdict"] = "undetermined"
        out["message"] = "too few usable deciles to read a trend"
    elif abs(r_tr) <= tol:
        out["verdict"] = "flat"
        out["message"] = f"rho1 flat to {r_tr:+.3f}; the convention holds"
    elif r_tr > 0:
        out["verdict"] = "under_corrected"
        out["message"] = (f"rho1 rises {r_tr:+.3f} with coverage: "
                          f"under-corrected, try error_convention='raw'")
    else:
        out["verdict"] = "over_corrected"
        out["message"] = (f"rho1 falls {r_tr:+.3f} with coverage: "
                          f"over-corrected, try "
                          f"error_convention='posterior_mean'")
    return out


# --------------------------------------------------------------- B.7.2 ------

def distance_decay(rows: pd.DataFrame, stratum_order) -> pd.DataFrame:
    """Median rho1, rho2 and D per stratum, plus rho2's profile relative to rho1.

    ``rho2_rel`` is rho2/rho1 normalised to its value in the first stratum: if
    the second axis is a rescaled shadow of the first, this is flat at 1 across
    distance.  A departure from 1 is the evidence that it is not.
    """
    out = (rows.groupby(["group", "stratum"], observed=True)
               .agg(rho1=("rho1", "median"), rho2=("rho2", "median"),
                    D=("D", "median"), D_excess=("D_excess", "median"),
                    n_bins=("rho1", "size")).reset_index())
    out["stratum"] = pd.Categorical(out["stratum"], categories=list(stratum_order),
                                    ordered=True)
    out = out.sort_values(["group", "stratum"]).reset_index(drop=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = out["rho2"] / out["rho1"]
    out["rho2_over_rho1"] = ratio
    first = out.groupby("group")["rho2_over_rho1"].transform("first")
    out["rho2_rel"] = ratio / first
    return out


# --------------------------------------------------------------- B.7.3 ------

def conversion_association(units: pd.DataFrame, conversion, n_bins: int = 8,
                           min_contacts: int = 50) -> pd.DataFrame:
    """rho1, rho2 and D against per-molecule conversion rate.

    A residual trend means the A.3.5 threshold is too permissive.  This is the
    only diagnostic that can see it: the artifact is molecule-level, so it
    survives the within-cell mate shuffle and the B.6.2 null alike.
    """
    if not len(units):
        return pd.DataFrame()
    u = units.copy()
    u["_conv"] = np.asarray(conversion, float)
    ok = np.isfinite(u["_conv"])
    u = u[ok]
    if not len(u):
        return pd.DataFrame()
    try:
        u["conv_bin"] = pd.qcut(u["_conv"], n_bins, labels=False,
                                duplicates="drop")
    except ValueError:
        u["conv_bin"] = 0

    rows = []
    for b, sub in u.groupby("conv_bin", sort=True):
        if len(sub) < min_contacts:
            continue
        r = _fit(*_arrays(sub))
        r.update(conv_bin=int(b), n_contacts=len(sub),
                 mean_conversion=float(sub["_conv"].mean()))
        rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        out.attrs["trend"] = {k: _trend(out["mean_conversion"], out[k])
                              for k in ("rho1", "rho2", "D")}
    return out


# --------------------------------------------------------------- B.7.4 ------

def null_calibration(n_units: int = 60, n_contacts: int = 400,
                     n_boot: int = 200, seed: int = 0,
                     rank2: float = 0.0) -> pd.DataFrame:
    """B.6.2 p-values on SIMULATED data of known rank.

    With ``rank2 = 0`` the truth is rank 1 and the p-values must be uniform;
    a non-uniform QQ invalidates every rank-2 call the pipeline makes.  Pass a
    nonzero ``rank2`` to check the same machinery has power against the
    alternative -- a test that is uniform under the null and also uniform under
    the alternative is uniform because it is measuring nothing.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        s = int(rng.integers(1 << 30))
        g = np.random.default_rng(s)
        n = n_contacts
        rho_u = float(g.uniform(0.3, 0.8))
        cov = np.array([[1.0, rho_u], [rho_u, 1.0]])
        u = g.multivariate_normal([0, 0], cov, size=n)
        e = g.multivariate_normal([0, 0], [[1.0, rank2], [rank2, 1.0]], size=n)
        lam_h = -0.8
        tA = np.column_stack([u[:, 0] + 0.5 * g.normal(size=n),
                              lam_h * u[:, 0] + 0.5 * e[:, 0]])
        tB = np.column_stack([u[:, 1] + 0.5 * g.normal(size=n),
                              lam_h * u[:, 1] + 0.5 * e[:, 1]])
        v = np.full((n, 2), 1e-4)
        res = rj.conditional_rank1_null(tA, tB, v, v, n_boot=n_boot, seed=s)
        rows.append({"unit": i, "rho_u": rho_u, "p_rank1": res["p_rank1"],
                     "D": res["D"], "D_excess": res["D_excess"],
                     "rho2_excess": res["rho2_excess"]})
    out = pd.DataFrame(rows)
    out.attrs["rank2"] = rank2
    out.attrs["ks_uniform_p"] = _ks_uniform(out["p_rank1"].to_numpy())
    out.attrs["frac_below_05"] = float((out["p_rank1"] < 0.05).mean())
    return out


def _ks_uniform(p) -> float:
    """One-sample KS against Uniform(0,1), p-value via the asymptotic form.

    scipy is not a dependency of this package, and the asymptotic series is
    accurate well past the n used here.
    """
    p = np.sort(np.asarray(p, float))
    p = p[np.isfinite(p)]
    n = p.size
    if n < 5:
        return float("nan")
    i = np.arange(1, n + 1)
    d = float(max(np.max(i / n - p), np.max(p - (i - 1) / n)))
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
    k = np.arange(1, 101)
    return float(np.clip(2 * np.sum((-1) ** (k - 1)
                                    * np.exp(-2 * k ** 2 * lam ** 2)), 0, 1))


# --------------------------------------------------------------- B.7.5 ------

def mate_shuffle_sanity(rows: pd.DataFrame) -> pd.DataFrame:
    """Per group: where rho1 sits under the B.6.1 mate shuffle.

    It must collapse toward zero.  If it does not, rho1 is picking up something
    that survives reassigning mates -- shared coverage, shared conversion, a
    baseline that still carries cell-type structure -- and is not coordination.
    """
    out = (rows.groupby("group")
               .agg(rho1_observed=("rho1", "median"),
                    rho1_shuffled=("rho1_mate_shuffle_mean", "median"),
                    z_median=("z_mate_shuffle", "median"),
                    n_bins=("rho1", "size")).reset_index())
    with np.errstate(invalid="ignore", divide="ignore"):
        out["shuffled_over_observed"] = (out["rho1_shuffled"]
                                         / out["rho1_observed"])
    out["passes"] = out["shuffled_over_observed"].abs() < 0.5
    return out


# --------------------------------------------- B.7.6 / B.7.7 ablations ------

def compare_ablation(tables: dict[str, pd.DataFrame], key: str,
                     join_on=("group", "chrom", "bin_start", "stratum")
                     ) -> pd.DataFrame:
    """Line up two or more ablation arms on the same (group, bin, stratum).

    Comparing marginal distributions would hide the thing worth seeing: whether
    the SAME bin moves.  A pair of arms with identical medians can still
    disagree bin by bin, and it is the per-bin disagreement that decides
    whether a default is safe.
    """
    keep = list(join_on) + ["D", "D_excess", "rho1", "rho2", "rho2_over_rho1",
                            "N_eff", "flag_pd_clipped"]
    merged = None
    for arm, t in tables.items():
        sub = t[[c for c in keep if c in t.columns]].copy()
        sub = sub.rename(columns={c: f"{c}__{arm}" for c in sub.columns
                                  if c not in join_on})
        merged = sub if merged is None else merged.merge(sub, on=list(join_on),
                                                         how="outer")
    merged.attrs["ablation_key"] = key
    merged.attrs["arms"] = list(tables)
    return merged


def ablation_summary(merged: pd.DataFrame, stat: str = "D") -> pd.DataFrame:
    """Per-arm central tendency plus the per-bin shift from the first arm."""
    arms = merged.attrs.get("arms", [])
    if len(arms) < 2:
        return pd.DataFrame()
    base_col = f"{stat}__{arms[0]}"
    if base_col not in merged:
        return pd.DataFrame()
    base = merged[base_col]
    rows = []
    for arm in arms:
        col = f"{stat}__{arm}"
        if col not in merged:
            continue
        # Pulled as Series, not via merged[[col, base_col]]: for the first arm
        # those are the same name and the two-column selection comes back as a
        # DataFrame, which then fails on the first truth test.
        cur = merged[col]
        d = cur - base
        shared = cur.notna() & base.notna()
        rows.append({
            "arm": arm, "stat": stat,
            "median": float(cur.median()),
            "n_bins": int(cur.notna().sum()),
            "n_shared_with_first": int(shared.sum()),
            "median_shift_vs_first": float(d.median()),
            "max_abs_shift_vs_first": float(d.abs().max())
                if d.notna().any() else np.nan,
            "corr_with_first": float(cur[shared].corr(base[shared]))
                if shared.sum() > 2 else np.nan,
        })
    return pd.DataFrame(rows)
