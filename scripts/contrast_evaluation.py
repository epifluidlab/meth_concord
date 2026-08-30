#!/usr/bin/env python3
"""Does the differential-bin model actually work on THIS dataset?

Every power and false-positive rate quoted for these tests so far came from
a simulated fixture whose depth, sparsity and dispersion were chosen by
whoever wrote it.  A real run has its own, and they are what decide whether
a p-value means anything.  This measures both on the counts at hand.

FALSE POSITIVES, BY PERMUTING THE UNIT LABELS.  The unit of replication is
the cell type (tumor1 and tumor2 are two tumours; GM12878, IMR90 and
monocyte are three separate sources), so the exchangeable thing under the
null is WHICH units are called the contrast arm.  Every balanced relabelling
other than the real one is run through the identical test, and the p-values
it returns are the null distribution the real run is judged against.

  THIS IS AN UPPER BOUND, NOT AN EXACT RATE, and the direction is knowable.
  Cell types genuinely differ from each other, so a permuted split such as
  {tumor1, GM12878} against {tumor2, IMR90, monocyte} still carries real
  biology.  Whatever it reports as "false" includes some true signal, so the
  measured rate is at least the true one.  A method that looks calibrated
  here IS calibrated; one that looks inflated may be partly reading biology.

POWER, BY SPIKING A KNOWN EFFECT INTO THE REAL COUNTS.  A random sample of
bins has the contrast arm's log odds TILTED by a known delta -- each cell's
own odds multiplied by exp(delta), with its depth held exactly -- so the
real depth, sparsity, dispersion and cell-to-cell structure are all kept and
only the effect is synthetic.  Resampling the counts instead would replace
the dataset's own over-dispersion with binomial noise and overstate power.

WHAT TO READ.  A method whose permuted p-values are uniform is calibrated on
this data; one whose curve bows above the diagonal is not, and its FDR means
nothing.  Power at each delta says the smallest effect worth claiming.  Both
are reported per method, so the choice is made on this dataset rather than
on a fixture.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_MB = Path(__file__).resolve().parent / "celltype_marker_bins.py"


def _marker_bins():
    """The SHIPPED implementation, imported rather than re-written.

    Evaluating a copy would measure the copy.  Every method scored here is
    the same function celltype_marker_bins.py calls.
    """
    spec = importlib.util.spec_from_file_location("_mb", _MB)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_counts(counts_dir, keys, cells=None, bins=None):
    d = Path(counts_dir)
    d = d / "counts" if (d / "counts").is_dir() else d
    out = {}
    for k in keys:
        f = d / f"{k}.parquet"
        if not f.exists():
            sys.exit(f"{f} not found -- re-run step 5 with --emit-counts")
        df = pd.read_parquet(f)
        if cells is not None:
            df = df.reindex(index=cells)
        if bins is not None:
            df = df.reindex(columns=bins)
        out[k] = df.to_numpy(float)
    return out, (list(df.index), list(df.columns))


def balanced_splits(units, n_a, true_a):
    """Every way of calling ``n_a`` of the units the contrast arm, minus the
    real one.

    Balanced -- the same arm sizes as the real contrast -- because power and
    variance both depend on them, and a 1-vs-4 split would not be the test
    that was actually run.
    """
    true_a = frozenset(true_a)
    out = []
    for combo in itertools.combinations(sorted(units), int(n_a)):
        if frozenset(combo) != true_a:
            out.append(frozenset(combo))
    return out


def spike(counts, rows, bins, delta, statistic, models, unit_ids=None,
          unit_sd=0.0, rng=None):
    """Tilt the contrast arm's log odds by ``delta`` in ``bins``.

    Each cell's own odds are multiplied by exp(delta) with its total held
    exactly, so depth, sparsity and the cell's deviation from its arm's mean
    all survive and only the location moves.  Re-drawing the counts instead
    would substitute binomial noise for this dataset's real over-dispersion,
    which is the thing that decides power.
    """
    succ_k, fail_k, _, _ = models[statistic]
    out = {k: v.copy() for k, v in counts.items()}
    idx = np.asarray(bins, int)
    s = np.sum([out[k][np.ix_(rows, idx)] for k in succ_k], axis=0)
    f = np.sum([out[k][np.ix_(rows, idx)] for k in fail_k], axis=0)
    tot = s + f
    ok = (s > 0) & (f > 0)
    # ``unit_sd`` GIVES THE PLANTED EFFECT ITS OWN BETWEEN-UNIT VARIATION,
    # and it is the difference between a fair comparison and a rigged one.
    # At 0 every cell in the arm gets exactly the same shift, so the effect
    # is perfectly consistent across donors -- the best possible case for a
    # test that treats cells as replicates, and not what a real tumour
    # effect looks like.  Drawing a per-(unit, bin) offset around delta
    # models an effect that differs between tumours, which is what the
    # unit-level test is built to survive and the cell-level one is not.
    eff = np.full(s.shape, float(delta))
    if unit_sd and unit_ids is not None:
        rng = rng or np.random.default_rng(0)
        u = np.asarray(unit_ids)[rows]
        levels = list(dict.fromkeys(u.tolist()))
        jitter = rng.normal(0.0, float(unit_sd), (len(levels), idx.size))
        eff = eff + jitter[[levels.index(x) for x in u.tolist()]]
    odds = np.where(ok, s / np.where(f > 0, f, 1.0), np.nan) * np.exp(eff)
    new_s = np.where(ok, tot * odds / (1.0 + odds), s)
    scale_s = np.where(s > 0, new_s / np.where(s > 0, s, 1.0), 1.0)
    scale_f = np.where(f > 0, (tot - new_s) / np.where(f > 0, f, 1.0), 1.0)
    for k in succ_k:
        out[k][np.ix_(rows, idx)] *= scale_s
    for k in fail_k:
        out[k][np.ix_(rows, idx)] *= scale_f
    return out


def run_method(mb, method, counts, in_mask, out_mask, statistic, cells,
               units, min_cells, min_pairs, prior_df, n_jobs):
    """One method's per-bin p-values.  Returns (p, diff)."""
    if method == "glm":
        r = mb.glm_contrast(counts, in_mask, out_mask, statistic, cells,
                            donor=units, min_cells=min_cells,
                            min_pairs=min_pairs, prior_df=prior_df,
                            n_jobs=n_jobs)
        return r["p"], r["diff"]
    if method == "glm-cell":
        # Reported so the run can SHOW why the cell-level p is not used,
        # on this dataset rather than on a fixture.
        import statsmodels.api as sm
        import warnings as _w
        eff = mb.glm_effect(counts, in_mask, out_mask, statistic, cells,
                            min_cells=min_cells, min_pairs=min_pairs,
                            n_jobs=n_jobs)
        succ_k, fail_k, e_s_k, e_f_k = mb.GLM_MODELS[statistic]
        keep = in_mask | out_mask
        rows = np.flatnonzero(keep)
        g = in_mask[rows].astype(float)
        X = np.column_stack([np.ones(rows.size), g])
        p = np.full(counts["n_linked"].shape[1], np.nan)
        for j in range(p.size):
            ok = counts["n_linked"][rows, j] >= min_pairs
            if ok.sum() < 2 * min_cells or g[ok].sum() < min_cells \
                    or (1 - g[ok]).sum() < min_cells:
                continue
            sm_ = lambda ks: np.sum([counts[k][rows[ok], j] for k in ks], axis=0)
            s, f = sm_(succ_k), sm_(fail_k)
            es, ef = sm_(e_s_k), sm_(e_f_k)
            off = np.log(np.clip(es, 1e-9, None)) - np.log(np.clip(ef, 1e-9, None))
            try:
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    p[j] = sm.GLM(np.column_stack([np.clip(s, 0, None),
                                                   np.clip(f, 0, None)]),
                                  X[ok], family=sm.families.Binomial(),
                                  offset=off).fit().pvalues[1]
            except Exception:
                pass
        return p, eff["diff"]
    if method in ("mwu", "welch"):
        # The per-cell statistic the rank tests see, built from the same raw
        # counts so the comparison is between TESTS and not between inputs.
        succ_k, fail_k, e_s_k, e_f_k = mb.GLM_MODELS[statistic]
        g = lambda ks: np.sum([counts[k] for k in ks], axis=0)
        s, f, es, ef = g(succ_k), g(fail_k), g(e_s_k), g(e_f_k)
        with np.errstate(divide="ignore", invalid="ignore"):
            X = (np.log(s) - np.log(f)) - (np.log(es) - np.log(ef))
        X = np.where((s > 0) & (f > 0) & (es > 0) & (ef > 0), X, np.nan)
        cs = mb.contrast_stats(X, in_mask, min_cells, out_mask, method)
        return cs["p"], cs["diff"]
    raise ValueError(method)


def calibration(p_null):
    """What the permuted p-values say about this dataset."""
    p = np.asarray(p_null, float)
    p = p[np.isfinite(p)]
    if p.size < 20:
        return {"n": int(p.size), "fpr_05": np.nan, "fpr_01": np.nan,
                "ks": np.nan, "median": np.nan}
    ps = np.sort(p)
    ecdf = np.arange(1, ps.size + 1) / ps.size
    return {"n": int(p.size),
            "fpr_05": float(np.mean(p <= 0.05)),
            "fpr_01": float(np.mean(p <= 0.01)),
            "ks": float(np.max(np.abs(ecdf - ps))),
            "median": float(np.median(p))}


def figure(path, per_method, powers, deltas, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.3))
    cm = plt.get_cmap("tab10")
    lv = np.concatenate([np.logspace(-4, -1, 16), np.linspace(0.1, 1.0, 25)])
    lv = np.unique(lv)

    a = ax[0]
    lim = 0.0
    for i, (nm, p) in enumerate(per_method.items()):
        p = np.asarray(p, float)
        p = p[np.isfinite(p)]
        if p.size < 20:
            continue
        q = np.quantile(p, lv)
        ok = (lv > 0) & (q > 0)
        x, y = -np.log10(lv[ok]), -np.log10(q[ok])
        a.plot(x, y, "-o", ms=2.5, lw=1.3, color=cm(i % 10), label=nm)
        lim = max(lim, float(np.max(np.r_[x, y])))
        n = p.size
        k = np.clip(np.ceil(lv[ok] * n), 1, n)
        try:
            from scipy.stats import beta
            if i == 0:
                a.fill_between(x, -np.log10(beta.ppf(.975, k, n - k + 1)),
                               -np.log10(beta.ppf(.025, k, n - k + 1)),
                               color="0.85", lw=0, zorder=0,
                               label="95% band")
        except Exception:
            pass
    lim = lim * 1.05 if lim > 0 else 1.0
    a.plot([0, lim], [0, lim], "--", color="0.5", lw=1.0, zorder=1)
    a.set_xlim(0, lim)
    a.set_ylim(0, lim)
    a.set_xlabel("expected  -log10(p)   (uniform)", fontsize=8)
    a.set_ylabel("permuted labels  -log10(p)", fontsize=8)
    a.set_title("a  is the test calibrated on THIS data?", fontsize=10,
                loc="left")
    a.annotate("above the diagonal = anti-conservative;\n"
               "an upper bound, since permuted splits\n"
               "still mix real cell types",
               (0.03, 0.97), xycoords="axes fraction", fontsize=6.5,
               color="0.35", va="top")
    a.legend(fontsize=6.5, frameon=False, loc="lower right")

    a = ax[1]
    names = list(per_method)
    for j, (key, nominal, col) in enumerate(
            (("fpr_05", 0.05, "#2c3e50"), ("fpr_01", 0.01, "#8e44ad"))):
        vals = [calibration(per_method[n]).get(key, np.nan) / nominal
                for n in names]
        a.barh(np.arange(len(names)) + (j - .5) * .38, vals, height=.36,
               color=col, label=f"p <= {nominal:g}")
    a.axvline(1.0, color="0.3", lw=1.0)
    a.axvline(2.0, color="#c0392b", ls="--", lw=0.9)
    a.set_yticks(range(len(names)))
    a.set_yticklabels(names, fontsize=7)
    a.set_xlabel("observed rate / nominal   (1 = calibrated)", fontsize=8)
    a.set_title("b  false positives on permuted labels", fontsize=10,
                loc="left")
    a.legend(fontsize=6.5, frameon=False, loc="lower right")

    a = ax[2]
    for i, nm in enumerate(names):
        y = [powers.get((nm, d), np.nan) for d in deltas]
        a.plot(deltas, y, "-o", ms=4, lw=1.4, color=cm(i % 10), label=nm)
    a.axhline(0.8, color="0.6", ls=":", lw=0.9)
    a.set_ylim(-0.02, 1.02)
    a.set_xlabel("effect spiked into the real counts  (log odds)", fontsize=8)
    a.set_ylabel("fraction recovered at a TRUE 5% rate", fontsize=8)
    a.set_title("c  power against a known effect", fontsize=10, loc="left")
    a.annotate("scored at each method's own threshold for a\n"
               "true 5% rate on this data -- a nominal 0.05\n"
               "would just reward the uncalibrated ones",
               (0.03, 0.02), xycoords="axes fraction", fontsize=6.5,
               color="0.35", va="bottom")
    a.legend(fontsize=6.5, frameon=False, loc="upper left")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-counts", required=True, metavar="DIR",
                    help="A step-5 05_concordance directory (or its counts/ "
                         "subdirectory) written with --emit-counts.")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--cell-type-column", default="cell_type")
    ap.add_argument("--replicate-column", default=None,
                    help="Unit of biological replication; defaults to "
                         "--cell-type-column.")
    ap.add_argument("--contrast", required=True, metavar="NAME=A,B[:C,D]",
                    help="Same syntax as celltype_marker_bins.py --contrast.")
    ap.add_argument("--statistic", default="direction_logor",
                    choices=["coordination", "direction_logor"])
    ap.add_argument("--methods", nargs="+",
                    default=["glm", "mwu", "glm-cell"],
                    choices=["glm", "mwu", "welch", "glm-cell"],
                    help="Default includes glm-cell so the run can show, on "
                         "this dataset, why the cell-level p is not used.")
    ap.add_argument("--spike", nargs="+", type=float,
                    default=[0.2, 0.4, 0.8, 1.6], metavar="DELTA",
                    help="Log-odds effects to inject (default 0.2 0.4 0.8 "
                         "1.6).")
    ap.add_argument("--spike-unit-sd", type=float, default=0.0,
                    metavar="SD",
                    help="Between-unit spread of the spiked effect, in log "
                         "odds.  At 0 (default) every cell in the arm gets "
                         "exactly the same shift, so the planted effect is "
                         "perfectly consistent across donors -- the best "
                         "possible case for any test that treats cells as "
                         "replicates, and not what a real tumour effect "
                         "looks like.  Set it to something like 0.3 to model "
                         "an effect that differs between the two tumours; "
                         "that is the comparison the unit-level test is "
                         "built for, and the ranking can change.  Run both.")
    ap.add_argument("--n-spike-bins", type=int, default=400)
    ap.add_argument("--max-bins", type=int, default=0, metavar="N",
                    help="Subsample bins before evaluating, for speed.  0 "
                         "uses all of them.")
    ap.add_argument("--max-permutations", type=int, default=0, metavar="N",
                    help="Cap the number of label relabellings.  0 uses "
                         "every balanced one.")
    ap.add_argument("--min-cells-per-group", type=int, default=3)
    ap.add_argument("--min-pairs", type=int, default=1)
    ap.add_argument("--glm-prior-df", type=float, default=4.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--plot-format", default="pdf",
                    choices=["pdf", "png", "svg"])
    args = ap.parse_args(argv)

    mb = _marker_bins()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    from nome_hic_concordance.cli import load_samples
    samples = load_samples(args.samples)
    idcol = samples.columns[0]
    ct = samples.set_index(idcol)[args.cell_type_column].astype(str)
    rep_col = args.replicate_column or args.cell_type_column
    rep = samples.set_index(idcol)[rep_col].astype(str)

    label, _, spec = str(args.contrast).partition("=")
    arm_a, _, arm_b = spec.partition(":")
    a_types = [t.strip() for t in arm_a.split(",") if t.strip()]
    b_types = [t.strip() for t in arm_b.split(",") if t.strip()]

    counts, (cells, bins) = load_counts(args.from_counts, mb.GLM_COUNT_KEYS)
    keep_cells = [c for c in cells if c in ct.index]
    if len(keep_cells) < len(cells):
        idx = [cells.index(c) for c in keep_cells]
        counts = {k: v[idx] for k, v in counts.items()}
        cells = keep_cells
    ct_v = ct.reindex(cells).to_numpy()
    rep_v = rep.reindex(cells).to_numpy()

    if args.max_bins and len(bins) > args.max_bins:
        pick = np.sort(rng.choice(len(bins), args.max_bins, replace=False))
        counts = {k: v[:, pick] for k, v in counts.items()}
        bins = [bins[i] for i in pick]
    n_bins = len(bins)

    in_true = np.isin(ct_v, a_types)
    out_true = np.isin(ct_v, b_types) if b_types else ~in_true
    units_a = sorted(set(rep_v[in_true]))
    units_b = sorted(set(rep_v[out_true]))
    units = units_a + units_b
    print(f"  {len(cells):,} cells x {n_bins:,} bins", file=sys.stderr)
    print(f"  replication unit {rep_col!r}: {len(units_a)} in the contrast "
          f"arm ({', '.join(units_a)}) vs {len(units_b)} in the reference "
          f"({', '.join(units_b)})", file=sys.stderr)
    if len(units_a) < 2 or len(units_b) < 2:
        sys.exit(
            f"An arm with fewer than 2 replication units cannot be tested "
            f"or evaluated: no variance between units to estimate.  Point "
            f"--replicate-column at something with more levels, or widen "
            f"the contrast.")

    perms = balanced_splits(units, len(units_a), units_a)
    if args.max_permutations and len(perms) > args.max_permutations:
        sel = rng.choice(len(perms), args.max_permutations, replace=False)
        perms = [perms[i] for i in sorted(sel)]
    print(f"  {len(perms)} balanced relabellings for the null "
          f"({len(perms) * n_bins:,} null p-values per method)",
          file=sys.stderr)

    kw = dict(statistic=args.statistic, cells=cells, units=rep_v,
              min_cells=args.min_cells_per_group, min_pairs=args.min_pairs,
              prior_df=args.glm_prior_df, n_jobs=args.threads)

    rows, per_method_null, powers = [], {}, {}
    for meth in args.methods:
        null_p = []
        for combo in perms:
            m_in = np.isin(rep_v, list(combo))
            m_out = np.isin(rep_v, [u for u in units if u not in combo])
            p, _ = run_method(mb, meth, counts, m_in, m_out, **kw)
            null_p.append(np.asarray(p, float))
        null_p = np.concatenate(null_p) if null_p else np.array([])
        per_method_null[meth] = null_p
        cal = calibration(null_p)
        p_obs, d_obs = run_method(mb, meth, counts, in_true, out_true, **kw)
        n_call = int(np.sum(np.isfinite(p_obs) & (p_obs <= 0.05)))
        print(f"  {meth:9s} permuted FPR: {cal['fpr_05']:.3f} at 0.05, "
              f"{cal['fpr_01']:.3f} at 0.01 (KS {cal['ks']:.3f}); the REAL "
              f"contrast calls {n_call:,} of {n_bins:,} bins at p<=0.05",
              file=sys.stderr)
        row = {"method": meth, "statistic": args.statistic,
               "n_null_p": cal["n"], "fpr_05": cal["fpr_05"],
               "fpr_01": cal["fpr_01"], "ks_uniform": cal["ks"],
               "n_permutations": len(perms), "n_bins": n_bins,
               "real_n_p05": n_call,
               "real_n_p05_expected_by_chance": cal["fpr_05"] * n_bins}
        # POWER AT A NOMINAL 0.05 IS NOT COMPARABLE ACROSS METHODS when they
        # differ in calibration: an inflated test buys power by calling more
        # of everything.  So each is also scored at ITS OWN threshold for a
        # true 5% rate, read off this dataset's permuted null.  That is the
        # comparison worth making.
        fin_null = null_p[np.isfinite(null_p)]
        thr_cal = (float(np.quantile(fin_null, 0.05)) if fin_null.size >= 200
                   else np.nan)
        row["threshold_for_true_5pct"] = thr_cal
        for delta in args.spike:
            hit = np.sort(rng.choice(n_bins, min(args.n_spike_bins, n_bins),
                                     replace=False))
            sc = spike(counts, np.flatnonzero(in_true), hit, float(delta),
                       args.statistic, mb.GLM_MODELS, unit_ids=rep_v,
                       unit_sd=args.spike_unit_sd, rng=rng)
            p_s, _ = run_method(mb, meth, sc, in_true, out_true, **kw)
            v = np.asarray(p_s, float)[hit]
            v = v[np.isfinite(v)]
            pw = float(np.mean(v <= 0.05)) if v.size else np.nan
            pw_c = (float(np.mean(v <= thr_cal))
                    if v.size and np.isfinite(thr_cal) else np.nan)
            powers[(meth, delta)] = pw_c
            row[f"power_delta_{delta:g}"] = pw
            row[f"power_calibrated_delta_{delta:g}"] = pw_c
            print(f"    spiked {delta:+.2f} log-odds into {len(hit)} real "
                  f"bins -> recovered {pw:.2f} at a nominal 0.05, "
                  f"{pw_c:.2f} at this method's own 5% threshold "
                  f"(p<={thr_cal:.2g})", file=sys.stderr)
        rows.append(row)

    tab = pd.DataFrame(rows)
    tab.to_csv(out / f"contrast_evaluation_{args.statistic}.tsv", sep="\t",
               index=False)
    figure(out / f"contrast_evaluation_{args.statistic}.{args.plot_format}",
           per_method_null, powers, list(args.spike),
           f"{label}: {args.statistic} on {n_bins:,} real bins")
    print(f"\n  -> {out}/contrast_evaluation_{args.statistic}.tsv and "
          f".{args.plot_format}", file=sys.stderr)
    print(f"  Read (a) first: a method whose permuted p-values leave the "
          f"band is not calibrated ON THIS DATA and its FDR means nothing, "
          f"whatever it measured on a fixture.  The permuted rate is an "
          f"UPPER bound -- relabelled splits still mix real cell types -- "
          f"so calibrated there is calibrated, while inflated may be partly "
          f"reading biology.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
