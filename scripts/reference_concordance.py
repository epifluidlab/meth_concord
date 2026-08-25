#!/usr/bin/env python3
"""Per-cell reference concordance from a chromosome with no 3D structure.

WHY.  The concordance D reported per bin is a Pearson r between the posteriors
at the two ends of a read pair, taken across pairs.  Under a noise model that
is a RELIABILITY coefficient, not a structural correlation:

    D = S / (S + N),    S = e^2 * var_A,    N = Vbar / n_sites

where ``e`` is the GpC methyltransferase efficiency for that cell, ``var_A``
the between-molecule variance in true accessibility, and ``n_sites`` the GCH
sites per read end.  Taking logits,

    logit(D) = 2 log e + log var_A + log n_sites - log Vbar

so the per-cell enzyme efficiency enters ADDITIVELY.  Subtracting the logit of
a reference D measured in the same cell cancels it.

WHY chrM.  Every nuclear reference candidate -- cLADs, replication-timing
domains, PMDs -- is defined as invariant across a panel of NORMAL cell types,
and chromatin reorganisation is exactly what a tumour does.  Using one assumes
away the thing under test, and because the reference set is small, being wrong
about it is worse than not correcting at all.  chrM does not make that
assumption: it carries no nucleosomes, so it has no chromatin state to
rearrange.  What it costs instead is the assumption that mtDNA packaging
heterogeneity (``var_A`` above) is constant across cells.

NO DISTANCE FLOOR.  The nuclear analysis discards pairs closer than
``--min-distance`` because short-range co-methylation is dominated by
nucleosome-scale structure rather than long-range architecture.  chrM has
neither: no nucleosomes, and no 3D contacts to speak of on a 16.6 kb circle.
A distance floor there discards almost all the data and keeps only spurious
ligations, so the default here is 0.

PCR DUPLICATES.  The reference must be built from a parse that IGNORED the
duplicate flag on chrM (``--keep-duplicates-chrom chrM``).  mtDNA is present
in hundreds to thousands of copies per cell, so two reads sharing
coordinates on 16.6 kb collide by chance rather than by PCR, and marking them
duplicates is simply wrong.  The cost is not depth: deduplication keeps one
read per coordinate, collapsing molecules that differ in methylation into a
single observation, which removes the between-molecule variance that IS the
reference.  This script checks for it -- if no two pairs share coordinates,
it says so rather than returning a confident, wrong number.

FIXED PAIR COUNT.  Pearson r carries a small-sample bias toward zero,
E[r_hat] ~ rho (1 - (1 - rho^2) / 2n).  mtDNA copy number differs severalfold
between cell types, so leaving the pair count free would make the reference
differently biased in each arm -- injecting the artefact it is meant to
remove.  Every cell is therefore subsampled to the SAME number of pairs, and a
cell that cannot reach it is reported unusable rather than quietly estimated
from fewer.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MITO = ("chrM", "M", "MT", "chrMT")


def canon_chrom(c):
    """'chrM', 'M', 'MT' all name the same sequence; a reference that silently
    matches nothing is worse than one that fails loudly."""
    c = str(c)
    c = c[3:] if c.startswith("chr") else c
    return "M" if c in ("M", "MT") else c


def _seed_for(cell, seed):
    """Deterministic per-cell seed, independent between cells.  Deriving it
    from the name rather than a counter keeps a cell's subsample identical
    whether or not its neighbours were processed."""
    h = hashlib.sha1(f"{cell}:{seed}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def reference_pairs(df, chrom="chrM", min_gch_sites=3, min_distance=0):
    """Rows of *df* usable for the reference, before subsampling."""
    need = {"chrom", "n1", "n2", "posterior1", "posterior2"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"posteriors missing columns {sorted(missing)}")
    want = canon_chrom(chrom)
    keep = df["chrom"].map(canon_chrom) == want
    keep &= (df["n1"] >= min_gch_sites) & (df["n2"] >= min_gch_sites)
    keep &= df["posterior1"].notna() & df["posterior2"].notna()
    if min_distance > 0 and "distance" in df.columns:
        keep &= df["distance"] >= min_distance
    return df.loc[keep]


def reference_d(df, cell, n_pairs, chrom="chrM", min_gch_sites=3,
                min_distance=0, seed=0):
    """Reference D for one cell, from exactly *n_pairs* subsampled pairs."""
    sub = reference_pairs(df, chrom, min_gch_sites, min_distance)
    n_avail = int(len(sub))
    out = {"cell": cell, "n_pairs_available": n_avail,
           "n_pairs_used": 0, "reference_d": np.nan,
           "reference_logor": np.nan, "reference_c": np.nan,
           "reference_f": np.nan,
           "median_sites_per_end": np.nan,
           "frac_repeated_coords": np.nan, "note": ""}
    if n_avail:
        out["median_sites_per_end"] = float(
            np.median(np.concatenate([sub["n1"].to_numpy(),
                                      sub["n2"].to_numpy()])))
    if n_avail and {"pos1", "pos2"} <= set(sub.columns):
        # How often two pairs share coordinates.  On a 16.6 kb sequence
        # carried in hundreds to thousands of copies per cell, collisions are
        # expected -- so their ABSENCE is the signature of a duplicate-marked
        # BAM parsed without the chrM exemption.  That is worth detecting
        # rather than trusting: deduplication keeps one read per coordinate,
        # collapsing molecules that differ in methylation, which removes the
        # between-molecule variance this whole number is made of.  The result
        # is not a thinner reference, it is a wrong one, and it looks fine.
        coords = list(zip(sub["pos1"].to_numpy(), sub["pos2"].to_numpy()))
        out["frac_repeated_coords"] = 1.0 - len(set(coords)) / float(n_avail)
    if n_avail < n_pairs:
        out["note"] = f"only {n_avail} pairs, need {n_pairs}"
        return out
    rng = np.random.default_rng(_seed_for(cell, seed))
    idx = rng.choice(n_avail, n_pairs, replace=False)
    p1 = sub["posterior1"].to_numpy(float)[idx]
    p2 = sub["posterior2"].to_numpy(float)[idx]
    out["n_pairs_used"] = int(n_pairs)
    if np.std(p1) < 1e-10 or np.std(p2) < 1e-10:
        # var_A ~ 0: the molecules do not differ, so there is no reliability
        # to measure.  Reported rather than returned as a number, because
        # dividing the genome by a degenerate reference is silently wrong.
        out["note"] = "posteriors have no variance (var_A ~ 0)"
        return out
    out["reference_d"] = float(np.corrcoef(p1, p2)[0, 1])
    lo, c = _logor_and_c(p1, p2)
    out["reference_logor"], out["reference_c"] = lo, c
    out["reference_f"] = _reference_f(p1, p2)
    return out


def _logor_and_c(p1, p2, eps=1e-9):
    """The same chrM pairs read as a log odds ratio and as a covariance.

    A reference is only usable on the scale its statistic lives on, because
    the attenuation it is meant to remove enters differently on each.  Under
    incomplete GpC conversion some open sites read closed, and that
    misclassification pulls any association toward the null MULTIPLICATIVELY:
    to first order L_obs = lambda(e) * L_true.  So a log odds ratio is
    corrected by DIVIDING by a reference log odds ratio, not by subtracting
    one, and not by anything computed on a Pearson r -- corr and log-OR are
    different functions of the same table and one does not stand in for the
    other.

    Same argument for C, which is 2*Cov and carries e^2 var_A: multiplicative
    again, so it divides too.  Only a reliability coefficient D = S/(S+N),
    which is a RATIO rather than an effect size, turns additive under logit.

    Computed from soft counts on the same pool, matching the pipeline's
    definition: F = O_open/(O_open + O_closed) against the marginal-matched
    null built from the two ends' own means.
    """
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    o_open = float(np.mean(p1 * p2))
    o_closed = float(np.mean((1.0 - p1) * (1.0 - p2)))
    m1, m2 = float(np.mean(p1)), float(np.mean(p2))
    e_open, e_closed = m1 * m2, (1.0 - m1) * (1.0 - m2)
    if min(o_open, o_closed, e_open, e_closed) < eps:
        return float("nan"), float("nan")
    logor = (np.log(o_open) - np.log(o_closed)
             - np.log(e_open) + np.log(e_closed))
    return float(logor), float((o_open + o_closed) - (e_open + e_closed))


def _reference_f(p1, p2, eps=1e-9):
    """chrM's own co-open share: F = O_open / (O_open + O_closed).

    A different use of chrM from the attenuation references.  Those divide a
    measured effect size; this is a per-cell NULL POSITION -- where the
    concordant mass sits in DNA that has no nucleosomes and no 3D structure.
    Subtracting its logit from a bin's asks whether that bin's mass sits
    co-OPEN or co-CLOSED relative to structure-free DNA in the same cell,
    which is a question the marginal-matched F_exp does not answer.

    F is a proportion, so logit is defined on it and the correction is an
    honest subtraction rather than a division by something that may be zero.

    WHAT IT DOES NOT CONTROL is the per-bin marginal.  F's null position
    depends on how accessible the bin is -- under independence at marginal p
    the concordant mass already splits p^2/(p^2 + (1-p)^2) -- and one
    per-cell number cannot track that from bin to bin.  Heterochromatin is
    less accessible by definition, so it will read co-closed against a
    chrM baseline whether or not anything is coordinated there.  That the
    answer looks biologically right is not evidence either way: the marginal
    artefact and the biology predict the same sign in exactly these regions.
    """
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    o_open = float(np.mean(p1 * p2))
    o_closed = float(np.mean((1.0 - p1) * (1.0 - p2)))
    if o_open + o_closed < eps:
        return float("nan")
    return o_open / (o_open + o_closed)


def suggest_n_pairs(available, keep_frac=0.9):
    """Largest round pair count that still keeps *keep_frac* of the cells.

    The count is a floor every cell must clear, so it trades cells against
    per-cell precision: raise it and thin cells drop out, lower it and the
    reference gets noisier in ALL of them.  Neither end is free, so report
    the achievable value rather than leaving the user to bisect --n-pairs by
    hand.
    """
    a = np.asarray([x for x in available if np.isfinite(x)], float)
    if a.size == 0:
        return None
    # An order statistic, not an interpolated quantile.  With a handful of
    # cells, interpolation lands ABOVE the smallest value and would quietly
    # exclude a third of them while claiming to keep 90%.  Taking the k-th
    # smallest makes the promise exact: at most k cells fall below it, and
    # rounding down to a round number can only ever keep more.
    k = int(np.floor((1.0 - keep_frac) * a.size))
    target = float(np.sort(a)[min(k, a.size - 1)])
    for step in (5000, 2000, 1000, 500, 200, 100, 50, 20):
        if target >= step:
            return int(step * int(target // step))
    return int(target) if target >= 1 else None


def reference_noise(r, n):
    """Per-cell noise this reference ADDS, on the logit scale it acts on.

    SE(r) ~ (1 - r^2)/sqrt(n - 3), propagated through the logit.  Worth
    printing because the correction is only a gain while this is well under
    the between-cell spread it removes: subtracting a noisy per-cell number
    is itself a per-cell effect, just a random one.
    """
    if not (np.isfinite(r) and 0 < r < 1) or n <= 3:
        return float("nan")
    return ((1.0 - r ** 2) / np.sqrt(n - 3)) / (r * (1.0 - r))


def discover(posterior_dir, suffix=".posteriors.parquet"):
    """cell -> path, from the 04_posteriors layout the pipeline writes."""
    d = Path(posterior_dir)
    return {p.name[: -len(suffix)]: p
            for p in sorted(d.glob(f"*{suffix}"))}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--posteriors", required=True,
                    help="Directory of CELL.posteriors.parquet (the "
                         "pipeline's 04_posteriors), or a single parquet.")
    ap.add_argument("--chrom", default="chrM",
                    help="Reference sequence (default chrM).  chrM/M/MT are "
                         "matched interchangeably.")
    ap.add_argument("--n-pairs", type=int, default=2000,
                    help="Pairs to subsample per cell.  IDENTICAL in every "
                         "cell so the small-sample bias of Pearson r is a "
                         "constant rather than a per-arm nuisance; cells "
                         "below it are reported unusable (default 2000).")
    ap.add_argument("--min-gch-sites", type=int, default=3,
                    help="Minimum GCH sites per read end, matching the "
                         "nuclear run so the posteriors carry the same "
                         "precision (default 3).")
    ap.add_argument("--min-distance", type=int, default=0,
                    help="Distance floor.  Default 0: chrM has no "
                         "nucleosomes and no 3D structure, so the nuclear "
                         "reason for a floor does not apply and a floor "
                         "would keep only spurious ligations.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True, help="Per-cell TSV.")
    args = ap.parse_args(argv)

    src = Path(args.posteriors)
    paths = ({src.stem.replace(".posteriors", ""): src} if src.is_file()
             else discover(src))
    if not paths:
        sys.exit(f"no posteriors parquet found under {src}")

    rows = []
    for cell, path in paths.items():
        # pos1/pos2 are read for the duplicate-marking check, not for the
        # statistic itself
        cols = ["chrom", "pos1", "pos2", "n1", "n2",
                "posterior1", "posterior2"]
        if args.min_distance > 0:
            cols.append("distance")
        try:
            df = pd.read_parquet(path, columns=cols)
        except Exception:                     # older files may lack a column
            df = pd.read_parquet(path)
        rows.append(reference_d(df, cell, args.n_pairs, args.chrom,
                                args.min_gch_sites, args.min_distance,
                                args.seed))

    out = pd.DataFrame(rows).set_index("cell").sort_index()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, sep="\t")

    ok = out["reference_d"].notna()
    print(f"{len(out)} cells; {int(ok.sum())} with a usable reference "
          f"({args.n_pairs} pairs each)", file=sys.stderr)
    if ok.any():
        d = out.loc[ok, "reference_d"]
        print(f"  reference D: median {d.median():.3f}, "
              f"range {d.min():.3f}..{d.max():.3f}", file=sys.stderr)
        if d.median() < 0.05:
            print("  WARNING reference D is near zero, so mtDNA molecules "
                  "barely differ from one another (var_A ~ 0).  There is "
                  "little reliability signal to divide out and the "
                  "correction will mostly add noise.", file=sys.stderr)
    rep = out["frac_repeated_coords"].dropna()
    if len(rep) and rep.median() < 0.001:
        print(f"  NOTE no two read pairs share coordinates on "
              f"{args.chrom}.  That is what a deduplicated parse looks "
              f"like -- and dedup here is harmful, because keeping one read "
              f"per coordinate collapses molecules that differ in "
              f"methylation and removes the between-molecule variance this "
              f"number is made of.  It is ALSO what a library whose reads "
              f"simply never collide looks like, in which case dedup was a "
              f"no-op and the reference is fine; the two cannot be told "
              f"apart from here.  If the parse did not use "
              f"--keep-duplicates-chrom chrM, re-run it.", file=sys.stderr)
    if (~ok).any():
        n_thin = int((out.loc[~ok, "n_pairs_available"]
                      < args.n_pairs).sum())
        print(f"  {int((~ok).sum())} cells unusable "
              f"({n_thin} short of {args.n_pairs} pairs).  They keep their "
              f"uncorrected values downstream.", file=sys.stderr)
        if n_thin:
            a = out["n_pairs_available"].to_numpy(float)
            qs = np.quantile(a, [0.05, 0.25, 0.5, 0.95])
            print(f"  pairs available per cell: 5% {qs[0]:,.0f} | "
                  f"25% {qs[1]:,.0f} | median {qs[2]:,.0f} | "
                  f"95% {qs[3]:,.0f}", file=sys.stderr)
            sug = suggest_n_pairs(a, 0.9)
            if sug:
                r_med = (float(out.loc[ok, "reference_d"].median())
                         if ok.any() else 0.3)
                noise = reference_noise(r_med, sug)
                print(f"  --n-pairs {sug} would keep ~90% of cells.  At that "
                      f"count and r = {r_med:.2f} the reference itself "
                      f"carries about {noise:.2f} of noise on the logit "
                      f"scale it corrects on -- worth it only while that is "
                      f"well under the between-cell spread you are removing, "
                      f"since subtracting a noisy per-cell number is itself "
                      f"a per-cell effect.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
