#!/usr/bin/env python3
"""Does the model's expected value actually capture the null?

Every setting in the sweep -- --baseline, --dm-prior, --dm-dist-bin-bp,
--dm-window-mb -- changes E and leaves O alone, so they are all claims about
where zero is.  Comparing them on the observed statistic cannot settle it:
a setting whose E is too small makes C look larger everywhere, and larger
is indistinguishable from better without something to hold it against.

WHY THERE IS NO PARAMETRIC REFERENCE.  A Q-Q plot needs a null distribution.
For a log odds ratio the textbook one is N(0, 1) after dividing by
sqrt(1/O_open + 1/O_closed), and it does not apply here: the counts are
Dirichlet-Multinomial posterior soft counts rather than counts, RWR has
already pooled evidence across the cell graph so entries are not
independent, and the imputed n is fractional and typically below 1.
Plotting against N(0, 1) would be asserting a sampling theory the pipeline
does not obey, and every setting would look miscalibrated by the same
irrelevant amount.

So the null is DRAWN, by ``--emit-null`` at step 5: exactly n_linked random
non-contact end-pairs per (cell, bin), summed the way the real contacts are,
from ends disjoint from the ones E is built on, and carried through the same
prior, the same imputation and the same normalisation.  It has no
coordination by construction.  Whatever it comes back as IS the null of that
model, including every downstream step, and that is what the observed
statistic is plotted against here.

THE ONE NUMBER.  ``null_bias_z`` is the drawn null's own median in units of
its own robust spread.  A model whose E captures the null returns 0 there --
no coordination in, none out.  Anything else is the model's own offset, and
it is the offset the observed statistic inherits.  ``lambda`` (observed
spread over null spread) says how much signal sits above that null, and is
only worth reading once null_bias_z is small: inflation over a null that is
itself displaced is not evidence of anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STATS = ("coordination", "direction_logor", "direction_excess",
         "directional_corrected", "direction")


def robust_sd(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    mad = float(np.median(np.abs(x - np.median(x))))
    return 1.4826 * mad if mad > 0 else float(np.std(x))


def tail_sd(x):
    """A scale read from the 0.5th-99.5th percentiles instead of the middle.

    (p99.5 - p0.5) / 5.152 is the same quantity as MAD-sd for a Gaussian,
    so the ratio of the two is a shape statistic and not a units mismatch.
    It survives a spike holding up to 99% of the mass, which the middle
    half does not.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    lo, hi = np.percentile(x, [0.5, 99.5])
    span = float(hi - lo)
    return span / 5.1517 if span > 0 else float("nan")


def _mad_sd_raw(x):
    """MAD-sd with NO fallback, so a degenerate middle reads as zero.

    robust_sd falls back to the plain sd when MAD is exactly 0, which keeps
    downstream arithmetic finite -- and hides the very thing the spike
    diagnostic is looking for.  An absence-filled matrix has more than half
    its mass at one value, so its MAD is 0 and the fallback quietly hands
    back the sd of the tails as though it were the width of the middle.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


def load_pair(run_dir, stat, cap=4_000_000, seed=0, min_pairs=0):
    """The observed statistic and its drawn null, flattened and finite.

    Returned as two independent samples rather than paired per (cell, bin):
    the null is a different draw, so pairing them would invite a scatter
    that means nothing.  Only their DISTRIBUTIONS are comparable.

    ``min_pairs`` DROPS ENTRIES THAT MEASURED NOTHING, and on a sparse
    context that is the difference between a readable plot and a
    meaningless one.  A (cell, bin) with no read pairs has k = 0 on both
    sides, so under a shared prior the statistic is exactly 0 -- for the
    observed AND for the drawn null.  Those are absences, not null draws,
    and a matrix full of them is a spike at zero with the real values in
    its tails: MAD-sd then measures the width of the spike and every entry
    carrying data lands thousands of nominal sds out, which is what an HCG
    run did.  Masking on the RAW n_linked (pair_counts_per_bin.parquet,
    written pre-imputation) removes them from both samples together, so
    the comparison stays like-for-like.
    """
    d = Path(run_dir)
    obs_p, null_p = (d / f"concordance_{stat}.parquet",
                     d / f"concordance_{stat}_null.parquet")
    if not obs_p.exists():
        raise SystemExit(f"{obs_p} not found")
    if not null_p.exists():
        raise SystemExit(
            f"{null_p} not found -- re-run step 5 with --emit-null. "
            f"Without a drawn null there is nothing valid to plot against; "
            f"see this script's docstring for why N(0, 1) is not it.")
    rng = np.random.default_rng(seed)
    obs_df, null_df = pd.read_parquet(obs_p), pd.read_parquet(null_p)
    info = {"min_pairs": int(min_pairs), "frac_kept": 1.0}
    if int(min_pairs) > 0:
        cp = Path(run_dir) / "pair_counts_per_bin.parquet"
        if not cp.exists():
            raise SystemExit(
                f"--min-pairs needs {cp}, which step 5 writes beside the "
                f"component matrices")
        counts = pd.read_parquet(cp).reindex(
            index=obs_df.index, columns=obs_df.columns)
        keep = counts.to_numpy(float) >= float(min_pairs)
        info["frac_kept"] = float(np.mean(keep))
        obs_df = obs_df.where(keep)
        null_df = null_df.where(keep)

    def _flat(df):
        a = df.to_numpy(float).ravel()
        a = a[np.isfinite(a)]
        if a.size > cap:
            a = a[rng.choice(a.size, cap, replace=False)]
        return a

    return _flat(obs_df), _flat(null_df), info


def calibration(obs, null):
    """How far the model's own null sits from zero, and what rides on it."""
    obs, null = np.asarray(obs, float), np.asarray(null, float)
    ns = robust_sd(null)
    os_ = robust_sd(obs)
    nm = float(np.median(null)) if null.size else float("nan")
    nb = tail_sd(null)
    # How much wider the TAILS are than the MIDDLE.  Near 1 on a normal
    # sample; large when the null is a spike, which is when the z-axis and
    # everything else divided by the middle stop meaning anything.  Measured
    # against the raw MAD so an exactly-degenerate middle reads as infinite
    # rather than being rescued by robust_sd's fallback.
    _raw = _mad_sd_raw(null)
    ratio = (float(nb / _raw) if np.isfinite(_raw) and _raw > 0
             else float("inf") if np.isfinite(nb) and nb > 0
             else float("nan"))
    out = {
        "n_obs": int(obs.size), "n_null": int(null.size),
        "null_median": nm, "null_madsd": float(ns),
        # THE calibration number: no coordination in, so zero out.
        "null_bias_z": float(nm / ns) if np.isfinite(ns) and ns > 0
        else float("nan"),
        "null_tailsd": float(nb),
        "spike_ratio": ratio,
        "obs_median": float(np.median(obs)) if obs.size else float("nan"),
        "obs_madsd": float(os_),
        "lambda": float(os_ / ns) if np.isfinite(ns) and ns > 0
        else float("nan"),
    }
    if obs.size and null.size:
        hi = float(np.percentile(np.abs(null), 99.0))
        out["tail_excess"] = float(np.mean(np.abs(obs) > hi) - 0.01)
        out["null_p5"] = float(np.percentile(null, 5))
        out["null_p95"] = float(np.percentile(null, 95))
    return out


QQ_LEVELS = np.linspace(0.001, 0.999, 400)


def qq_points(obs, null, n=400):
    """Matched quantiles of the two samples.

    Quantiles rather than sorted values, because the two samples need not be
    the same size and interpolating one onto the other's ranks would put the
    difference in sample size on the axis.
    """
    q = np.linspace(0.001, 0.999, int(n))
    return np.quantile(null, q), np.quantile(obs, q)


def qq_band(obs, null, levels=QQ_LEVELS, alpha=0.05):
    """Where the observed quantiles would fall if it WERE the null.

    Same order-statistic argument as the uniform case, pushed through the
    null's empirical quantile function instead of the identity: if the
    observed were n_obs independent draws from the null, its k-th order
    statistic sits at a Beta(k, n - k + 1) position in the null's CDF.
    """
    n = int(np.asarray(obs).size)
    if n < 2 or np.asarray(null).size < 2:
        return None
    lo, hi = order_statistic_ci(levels, n, alpha)
    return np.quantile(null, lo), np.quantile(null, hi)


def figure(path, table, curves, stat, title, bands=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13.6, 4.3))
    cm = plt.get_cmap("tab10")

    a = ax[0]
    # STANDARDISED ON EACH RUN'S OWN NULL: (q - null_median) / null_madsd.
    #
    # Raw quantiles do not work here.  Under --dm-prior separate the whole
    # distribution is a needle parked on the prior floor -- observed and
    # drawn null both at about -0.19, spread a small fraction of that -- so
    # an axis wide enough to reach zero leaves the data as a sliver and the
    # departure from the diagonal, which is the entire question, invisible.
    # Runs at different offsets could not be shown together at all.
    #
    # Centring and scaling on the NULL, not on the observed, keeps the thing
    # being measured: the observed's excess over its own null survives, and
    # the diagonal means "observed distributed exactly like its null".  The
    # absolute offset is not lost, it moves to panel (b), where it is the
    # headline rather than a scaling nuisance.
    lim = 0.0
    n_thin = 0
    for i, (name, (qx, qy)) in enumerate(curves.items()):
        nm = float(np.median(qx))
        ns = robust_sd(qx)
        if not (np.isfinite(ns) and ns > 0):
            continue
        zx, zy = (qx - nm) / ns, (qy - nm) / ns
        # 95% band, in the same standardised units.  Drawn only when it is
        # WIDE ENOUGH TO SEE: n_obs here runs to millions of cell-bins, and
        # a band a thousandth of an sd wide would read as "every departure
        # is significant" -- which is exactly the claim the independence
        # assumption cannot support after RWR.  Better to omit it and say so
        # than to draw a hairline that flatters the plot.
        bd = (bands or {}).get(name)
        if bd is not None:
            blo, bhi = (bd[0] - nm) / ns, (bd[1] - nm) / ns
            if float(np.max(bhi - blo)) >= 0.02 * max(
                    1e-12, float(np.ptp(np.r_[zx, zy]))):
                a.fill_between(zx, blo, bhi, color=cm(i % 10), alpha=0.15,
                               lw=0, zorder=0)
            else:
                n_thin += 1
        a.plot(zx, zy, "-", lw=1.5, color=cm(i % 10), label=name)
        lim = max(lim, float(np.max(np.abs(np.r_[zx, zy]))))
    lim = lim * 1.05 if lim > 0 else 1.0
    a.plot([-lim, lim], [-lim, lim], "--", color="0.5", lw=1.0, zorder=0)
    a.axhline(0, color="0.85", lw=0.8, zorder=0)
    a.axvline(0, color="0.85", lw=0.8, zorder=0)
    a.set_xlim(-lim, lim)
    a.set_ylim(-lim, lim)
    a.set_xlabel("drawn-null quantiles  (null sd, about the null's median)",
                 fontsize=8)
    a.set_ylabel("observed quantiles  (same units)", fontsize=8)
    a.set_title("a  observed against the model's own null", fontsize=10,
                loc="left")
    a.annotate("on the diagonal = distributed like its own null;\n"
               "where the null itself SITS is panel (b)"
               + ("\n95% band omitted: n is large enough that it is "
                  "narrower\nthan the line, and RWR makes the entries "
                  "non-independent\nanyway" if n_thin else ""),
               (0.03, 0.97), xycoords="axes fraction", fontsize=6.5,
               color="0.35", va="top")
    a.legend(fontsize=6.5, frameon=False, loc="lower right")

    a = ax[1]
    names = list(table["run"])
    z = table["null_bias_z"].to_numpy(float)
    cols = ["#2c3e50" if abs(v) < 1 else "#c0392b" for v in z]
    a.barh(range(len(names)), z, color=cols)
    a.axvline(0, color="0.3", lw=1.0)
    for s in (-1, 1):
        a.axvline(s, color="#c0392b", ls="--", lw=0.9)
    a.set_yticks(range(len(names)))
    a.set_yticklabels(names, fontsize=6.5)
    a.set_xlabel("null_bias_z  (the drawn null's own median / its sd)",
                 fontsize=8)
    a.set_title("b  is the null where the model says it is?", fontsize=10,
                loc="left")
    a.annotate("no coordination in, so zero out", (0, len(names) - 0.4),
               fontsize=7, color="#c0392b", ha="center")

    a = ax[2]
    lam = table["lambda"].to_numpy(float)
    ok = np.abs(z) < 1
    a.scatter(lam[ok], np.arange(len(names))[ok], s=42, color="#2c3e50",
              label="null within 1 sd of zero")
    a.scatter(lam[~ok], np.arange(len(names))[~ok], s=42, color="#c0392b",
              marker="x", label="null displaced -- lambda not readable")
    a.axvline(1.0, color="0.3", lw=1.0)
    a.set_yticks(range(len(names)))
    a.set_yticklabels([""] * len(names))
    a.set_xlabel("lambda  (observed spread / null spread)", fontsize=8)
    a.set_title("c  how much sits above that null", fontsize=10, loc="left")
    a.legend(fontsize=6.5, frameon=False, loc="lower right")

    fig.suptitle(f"{title} — {stat}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def load_rank_calibration(run_dir):
    """The held-out-draw self-check written beside the component matrices.

    Every other check in this script asks whether the model's E sits where
    it claims.  This one asks something the rank makes directly answerable:
    a DRAW from the null is a case whose true answer is known, so holding one
    out and scoring it against its siblings must return uniform p-values.
    Whatever excess shows up below 0.05 is the rank's own false-positive
    rate, and every observed bin scored the same way inherits it.

    Returns None when the run was made without --null-rank, or with
    --null-draws 1 (which leaves no draw to hold out).
    """
    import json
    fp = Path(run_dir) / "null_rank_calibration.json"
    if not fp.exists():
        return None
    with open(fp) as fh:
        return json.load(fh)


def order_statistic_ci(levels, n, alpha=0.05):
    """Pointwise CI for the order statistics of ``n`` draws, on the
    PROBABILITY scale.

    The k-th smallest of n independent Uniform(0, 1) draws is exactly
    Beta(k, n - k + 1), so the band needs no simulation and no asymptotics.
    Returned on the probability scale so the caller can push it through
    whichever reference quantile function it is plotting against: the
    identity for a uniform Q-Q, the null's own empirical quantile function
    for a two-sample one.

    IT ASSUMES THE ENTRIES ARE INDEPENDENT, AND HERE THEY ARE NOT.  RWR
    pools evidence across the cell graph, so neighbouring cells' values in
    the same bin are correlated and the effective sample size is smaller
    than n.  The band is therefore NARROWER than the truth -- read a curve
    that leaves it as "worth explaining", not as a test.
    """
    from scipy.stats import beta
    lv = np.asarray(levels, float)
    k = np.clip(np.ceil(lv * n), 1, n)
    return (beta.ppf(alpha / 2, k, n - k + 1),
            beta.ppf(1 - alpha / 2, k, n - k + 1))


def rank_figure(path, entries, title):
    """Q-Q of the held-out null's p-values against uniform, one line per run.

    Drawn on -log10 axes because the whole question lives in the small-p
    tail: a procedure can be near-perfect over the body and still call 5x
    too many bins at p <= 0.01, and on linear axes that is invisible.  The
    curve flattens at the right where the pool runs out of resolution -- the
    finest p a pool of M draws can express is 2/(M+1) -- which is a property
    of the pool size and not a miscalibration.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.3))
    cm = plt.get_cmap("tab10")
    a = ax[0]
    lim = 0.0

    # 95% pointwise band, drawn ONCE in grey rather than per run.  The
    # reference is Uniform(0, 1) for every run, so the band depends only on
    # how many cell-bins went in; taking the smallest n across runs makes it
    # the widest of them, so a curve outside it is outside every run's.
    _lv = next((np.asarray(c.get("p_quantile_levels", []), float)
                for c in entries.values()
                if len(c.get("p_quantile_levels", [])) > 2), None)
    _n = min((int(c.get("n", 0)) for c in entries.values()), default=0)
    if _lv is not None and _n > 1:
        _ok = _lv > 0
        _lo, _hi = order_statistic_ci(_lv[_ok], _n)
        # -log10 is decreasing, so the band's ends swap.
        a.fill_between(-np.log10(_lv[_ok]), -np.log10(_hi), -np.log10(_lo),
                       color="0.82", lw=0, zorder=0,
                       label=f"95% band (n = {_n:,})")

    for i, (name, cal) in enumerate(entries.items()):
        lv = np.asarray(cal.get("p_quantile_levels", []), float)
        pq = np.asarray(cal.get("p_quantiles", []), float)
        if lv.size < 3 or pq.size != lv.size:
            continue
        ok = (lv > 0) & (pq > 0)
        x, y = -np.log10(lv[ok]), -np.log10(pq[ok])
        a.plot(x, y, "-o", ms=2.5, lw=1.3, color=cm(i % 10), label=name)
        lim = max(lim, float(np.max(np.r_[x, y])))
        fp = cal.get("finest_p")
        if fp and np.isfinite(fp):
            a.axhline(-np.log10(fp), color=cm(i % 10), ls=":", lw=0.8)
    lim = lim * 1.05 if lim > 0 else 1.0
    a.plot([0, lim], [0, lim], "--", color="0.5", lw=1.0, zorder=0)
    a.set_xlim(0, lim)
    a.set_ylim(0, lim)
    a.set_xlabel("expected  -log10(p)   (uniform)", fontsize=8)
    a.set_ylabel("held-out drawn null  -log10(p)", fontsize=8)
    a.set_title("a  does a known-null bin score like a null bin?",
                fontsize=10, loc="left")
    a.annotate("above the diagonal = the rank calls its own null too\n"
               "often; dotted = pool resolution floor.  The band assumes\n"
               "independent entries -- RWR correlates them, so it is\n"
               "narrower than the truth.",
               (0.03, 0.97), xycoords="axes fraction", fontsize=6.5,
               color="0.35", va="top")
    a.legend(fontsize=6.5, frameon=False, loc="lower right")

    a = ax[1]
    names = list(entries)
    keys = (("frac_p05", 0.05, "#2c3e50", "p <= 0.05"),
            ("frac_p01", 0.01, "#8e44ad", "p <= 0.01"),
            ("frac_p05_pos", 0.025, "#c0392b", "p <= 0.05, positive"),
            ("frac_p05_neg", 0.025, "#2980b9", "p <= 0.05, negative"))
    keys = [k for k in keys if any(k[0] in e for e in entries.values())]
    h = 0.8 / max(1, len(keys))
    for j, (key, nominal, col, lab) in enumerate(keys):
        vals = [entries[n].get(key, np.nan) / nominal for n in names]
        a.barh(np.arange(len(names)) + (j - (len(keys) - 1) / 2) * h, vals,
               height=h * 0.92, color=col, label=lab)
    a.axvline(1.0, color="0.3", lw=1.0)
    a.axvline(2.0, color="#c0392b", ls="--", lw=0.9)
    a.set_yticks(range(len(names)))
    a.set_yticklabels(names, fontsize=6.5)
    a.set_xlabel("observed rate / nominal rate   (1 = calibrated)\n"
                 "the two signed bars should MATCH: one much taller means "
                 "the rank is\ncalibrated in magnitude but lopsided in "
                 "direction",
                 fontsize=8, linespacing=1.5)
    a.set_title("b  the rank's own false-positive rate", fontsize=10,
                loc="left")
    a.legend(fontsize=6.5, loc="lower right", framealpha=0.9,
             edgecolor="0.85")

    fig.suptitle(f"{title} — rank self-check", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def named(specs, flag):
    out = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"{flag} wants NAME=PATH, got {spec!r}")
        k, v = spec.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True,
                    metavar="NAME=DIR",
                    help="A 05_concordance directory from a step-5 run made "
                         "with --emit-null.  Repeat to compare settings.")
    ap.add_argument("--statistic", action="append", default=None,
                    choices=list(STATS),
                    help="Default: coordination and direction_logor.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--plot-format", default="pdf",
                    choices=["pdf", "png", "svg"])
    ap.add_argument("--title", default="Null calibration")
    ap.add_argument("--min-pairs", type=int, default=0, metavar="N",
                    help="Drop (cell, bin) entries with fewer than N RAW "
                         "read pairs, from the observed and the drawn null "
                         "TOGETHER.  An entry that measured nothing is "
                         "exactly 0 on both sides under a shared prior, so "
                         "a sparse context fills the matrix with structural "
                         "zeros -- a spike with the real values in its "
                         "tails.  That is an absence, not a null draw, and "
                         "it is why an HCG run came back on a +-4000 axis.  "
                         "Start at 1 and raise it until spike_ratio comes "
                         "down; frac_kept says what it cost.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    runs = named(args.run, "--run")
    stats = args.statistic or ["coordination", "direction_logor"]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for stat in stats:
        curves, bands, sub = {}, {}, []
        for name, d in runs.items():
            obs, null, li = load_pair(d, stat, seed=args.seed,
                                      min_pairs=args.min_pairs)
            m = calibration(obs, null)
            m.update(run=name, statistic=stat, **li)
            sub.append(m)
            rows.append(m)
            curves[name] = qq_points(obs, null)
            bands[name] = qq_band(obs, null)
        tab = pd.DataFrame(sub)
        figure(out / f"null_qq_{stat}.{args.plot_format}", tab, curves,
               stat, args.title, bands=bands)
        print(f"  -> null_qq_{stat}.{args.plot_format}", file=sys.stderr)

    # ---- The rank's own self-check, where a run wrote one ---------------
    rank_rows = []
    for stat in stats:
        entries = {}
        for name, d in runs.items():
            cal = load_rank_calibration(d)
            if cal and stat in cal:
                entries[name] = cal[stat]
                rank_rows.append({"statistic": stat, "run": name,
                                  **{k: v for k, v in cal[stat].items()
                                     if not isinstance(v, list)}})
        if entries:
            rank_figure(out / f"null_rank_qq_{stat}.{args.plot_format}",
                        entries, args.title)
            print(f"  -> null_rank_qq_{stat}.{args.plot_format}",
                  file=sys.stderr)
    if rank_rows:
        rk = pd.DataFrame(rank_rows)
        rk.to_csv(out / "null_rank_calibration.tsv", sep="\t", index=False)
        print("\nrank self-check: one drawn null held out and scored against "
              "its siblings.  The true answer is known there, so these ARE "
              "the rank's false-positive rates.", file=sys.stderr)
        for _, r in rk.iterrows():
            flag = "" if r["frac_p05"] <= 0.10 else "   <- NOT calibrated"
            pos, neg = r.get("frac_p05_pos"), r.get("frac_p05_neg")
            sgn = ""
            if pos is not None and neg is not None and np.isfinite(pos):
                sgn = f"  [+{pos:.3f} / -{neg:.3f}, nominal .025 each]"
                if max(pos, neg) > 3 * max(min(pos, neg), 1e-4):
                    flag += "   <- LOPSIDED in sign"
            print(f"  {r['run']:24s} {r['statistic']:16s} "
                  f"p05={r['frac_p05']:.3f} (nominal 0.05)  "
                  f"p01={r['frac_p01']:.3f} (0.01)  "
                  f"KS={r['ks_uniform']:.3f}  "
                  f"pool={int(r['median_pool'])} -> finest p "
                  f"{r['finest_p']:.2g}{sgn}{flag}", file=sys.stderr)
        print("  A run whose finest expressible p is coarser than the "
              "threshold you intend to FDR at needs more --null-draws, or "
              "wider strata.", file=sys.stderr)

    full = pd.DataFrame(rows)[
        ["statistic", "run", "min_pairs", "frac_kept", "null_median",
         "null_madsd", "null_tailsd", "spike_ratio", "null_bias_z",
         "obs_median", "obs_madsd", "lambda", "tail_excess",
         "n_obs", "n_null"]]
    full.to_csv(out / "null_calibration.tsv", sep="\t", index=False)
    for stat in stats:
        s = full[full["statistic"] == stat].sort_values(
            "null_bias_z", key=lambda c: c.abs())
        print(f"\n{stat}: ranked by |null_bias_z| (0 = the model's null is "
              f"where it says it is)", file=sys.stderr)
        # BY NAME.  'lambda' is a keyword, so itertuples renames it by
        # POSITION -- and reading it back as _8 broke the moment two columns
        # were added ahead of it.
        for _, r in s.iterrows():
            flag = ("" if abs(r["null_bias_z"]) < 1
                    else "   <- null displaced")
            if r["spike_ratio"] > 5.0:
                flag += "   <- null is a spike; see below"
            print(f"  {r['run']:32s} null_bias_z={r['null_bias_z']:+7.2f}  "
                  f"lambda={r['lambda']:6.2f}  "
                  f"tail_excess={r['tail_excess']:+.4f}{flag}",
                  file=sys.stderr)
    print(f"\n  lambda is only readable where the null itself sits near "
          f"zero: inflation measured against a displaced null is not "
          f"evidence of signal.", file=sys.stderr)
    bad = full[full["spike_ratio"] > 5.0]
    for r in bad.itertuples():
        print(f"\n  WARNING {r.run} / {r.statistic}: the drawn null's body "
              f"is {r.spike_ratio:.0f}x wider than its middle, so it is a "
              f"SPIKE with tails rather than a spread.  Its MAD-sd "
              f"({r.null_madsd:.3g}) is measuring the width of the spike, "
              f"not the noise -- which is why the axis runs to thousands of "
              f"nominal sds, and why null_bias_z and lambda are not "
              f"readable for it.  On a sparse context a large share of "
              f"(cell, bin) entries carry no reads and the statistic is "
              f"exactly 0 for all of them, on BOTH sides -- an absence, not "
              f"a null draw.  "
              + (f"Re-run with --min-pairs 1 to drop them from both samples "
                 f"together; frac_kept will say what it cost."
                 if int(r.min_pairs) == 0 else
                 f"Already at --min-pairs {int(r.min_pairs)}, keeping "
                 f"{r.frac_kept:.1%}; raise it further, or the bin size is "
                 f"too small for this context.")
              + f"  Meanwhile compare null_tailsd ({r.null_tailsd:.3g}), "
                f"not null_madsd.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
