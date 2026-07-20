#!/usr/bin/env python3
"""Diagnose a spike in a site_distance_correlation curve, from the TSV alone.

A bump in one curve has three candidate explanations, and the TSV already holds
enough to separate them without re-parsing anything:

  1. SAMPLING NOISE -- check ``n``.  At long range these bins carry ~1e5 site
     pairs and ~4e4 effective independent read pairs, where the SE of phi is
     ~0.005; a 0.15 bump is then ~30 SE and is NOT noise.  Small ``n`` is the
     only case where noise is a live option.
  2. MARGINAL SHIFT -- phi is bounded by the methylation rates, Yule's Q is
     not.  A bump present in phi but ABSENT in Q is the rate moving, not the
     sites becoming more coordinated.
  3. LOCUS EFFECT -- the null swaps in a different molecule at a nearby locus,
     so it shares the anchor LOCUS but not the molecule.  A bump that shows up
     in the null too cannot be same-molecule coordination; it is a property of
     which loci populate that distance bin.

Residuals are taken against a rolling median, so "spike" means excess over the
LOCAL trend rather than over the global decay.

Usage
-----
    python scripts/diagnose_spike.py <tsv> [measure] [min_bp] [max_bp]
    python scripts/diagnose_spike.py out/site_distance_corr_all.tsv hh 3000 300000
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def _resid(v):
    """Excess over the local trend (rolling median), not over the global decay."""
    s = pd.Series(v)
    return v - s.rolling(5, center=True, min_periods=2).median().to_numpy()


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    key = sys.argv[2] if len(sys.argv) > 2 else "hh"
    lo = float(sys.argv[3]) if len(sys.argv) > 3 else 3e3
    hi = float(sys.argv[4]) if len(sys.argv) > 4 else 3e5

    df = pd.read_csv(path, sep="\t")
    need = [f"{key}_linked_phi", f"{key}_linked_yule_q", f"{key}_linked_n",
            f"{key}_null_phi"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"{path} lacks {missing} — it predates the multi-metric TSV; "
                 f"regenerate it with --replot.")

    d = df[(df.center_bp >= lo) & (df.center_bp <= hi)]
    if len(d) < 4:
        sys.exit(f"only {len(d)} bins in {lo:,.0f}-{hi:,.0f} bp — widen the range.")
    phi = d[f"{key}_linked_phi"].to_numpy()
    q = d[f"{key}_linked_yule_q"].to_numpy()
    n = d[f"{key}_linked_n"].to_numpy()
    nul = d[f"{key}_null_phi"].to_numpy()

    rp, rq, rn = _resid(phi), _resid(q), _resid(nul)
    sp = np.nanstd(rp) or 1.0

    print(f"{key.upper()}  {lo:,.0f}-{hi:,.0f} bp   "
          f"(spike = |residual| > 2 sigma, sigma={sp:.3f})\n")
    print(f"{'dist':>10}{'n':>13}{'phi':>8}{'null':>8}{'Q':>8}"
          f"{'phi res':>10}{'Q res':>9}   verdict")
    for i in range(len(d)):
        if not np.isfinite(rp[i]):
            continue
        flag = ""
        if abs(rp[i]) > 2 * sp:
            rel = abs(rq[i]) / (abs(rp[i]) + 1e-12)
            flag = ("ASSOCIATION changed" if rel > 0.5
                    else "MARGINALS moved (rate shift, not coordination)")
        print(f"{d.center_bp.iloc[i]:>10,.0f}{int(n[i]):>13,}{phi[i]:>8.3f}"
              f"{nul[i]:>8.3f}{q[i]:>8.3f}{rp[i]:>10.3f}{rq[i]:>9.3f}   {flag}")

    ok = np.isfinite(rp) & np.isfinite(rn)
    if ok.sum() >= 4:
        r = float(np.corrcoef(rp[ok], rn[ok])[0, 1])
        print(f"\ncorr(phi residual, NULL residual) = {r:+.3f}")
        print("  near +1 -> the bump is in BOTH arms: a property of the LOCI in "
              "that bin,\n             not same-molecule coordination."
              "\n  near  0 -> the bump is specific to linked molecules.")

    spikes = np.abs(rp) > 2 * sp
    if spikes.any():
        print(f"\nweight at the spike bins: n = "
              f"{', '.join(f'{int(v):,}' for v in n[spikes])}"
              f"   (10th pct over this range = {np.nanpercentile(n, 10):,.0f})")
        print("  a spike bin with ORDINARY n is not a sparsity artefact.")


if __name__ == "__main__":
    main()
