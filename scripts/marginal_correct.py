#!/usr/bin/env python3
"""Write two- and three-stage marginal-corrected GCH concordance matrices.

Reads a Step-5 ``05_concordance`` directory and writes, per statistic:

    concordance_<stat>.margreg_k<K>.method1_local.parquet        (two-stage)
    concordance_<stat>.margreg_k<K>.method1_local.resid.parquet  (three-stage)

Both are ordinary cells x bins matrices, so every downstream tool consumes
them unchanged -- ``nome_hic_compare --concordance-matrix name=PATH``,
``celltype_marker_bins.py --matrix name=PATH`` and
``scaled_region_profile.py --matrix name=PATH`` all take a path.

WHEN TO USE THIS.  GCH only.  HCG concordance transferred across replicates
at 0.79 without correction and should be left alone; correcting a matrix
that does not need it can only cost signal.  See
:mod:`nome_hic_concordance.marginal_correction` for the measurements behind
the defaults and for what was ruled out before arriving at them.

Usage:
  marginal_correct.py 05_concordance_DIR SAMPLES.tsv --batch-col exp_batch
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nome_hic_concordance.marginal_correction import (      # noqa: E402
    DEFAULT_ANCHOR_K, DEFAULT_K, DEFAULT_PROTECT_PCS, stage3_residual,
    two_stage)

logger = logging.getLogger("marginal_correct")

#: Statistics corrected by default.  Both inherit the marginal through the
#: same identity and were measured to behave alike under the correction.
DEFAULT_STATS = ("coordination", "direction_logor")


def _load(path: Path, what: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{what} not found: {path}")
    return pd.read_parquet(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("concordance_dir",
                    help="Step-5 output directory (05_concordance).")
    ap.add_argument("samples", help="Samples TSV carrying the batch column.")
    ap.add_argument("--batch-col", required=True,
                    help="Replicate/batch column in --samples.")
    ap.add_argument("--stats", default=",".join(DEFAULT_STATS),
                    help=f"Comma list of statistics "
                         f"({','.join(DEFAULT_STATS)}).")
    ap.add_argument("--marginal", default=None,
                    help="Marginal matrix (default: "
                         "concordance_marginal.parquet in the directory).")
    ap.add_argument("--baseline", default=None,
                    help="Accessibility matrix protecting the biology "
                         "(default: accessibility_mean.parquet in the "
                         "directory).  Strongly recommended: unprotected, "
                         "stage 1 inverts.")
    ap.add_argument("--extra-baseline", action="append", default=None,
                    metavar="PATH",
                    help="Repeatable.  Additional matrices regressed out in "
                         "stage 3, e.g. compartment_scores.parquet.")
    ap.add_argument("-k", "--factor-k", type=int, default=DEFAULT_K,
                    help=f"Marginal factors removed in stage 1 "
                         f"({DEFAULT_K}).")
    ap.add_argument("--anchor-k", type=int, default=DEFAULT_ANCHOR_K,
                    help=f"Neighbours per replicate in stage 2 "
                         f"({DEFAULT_ANCHOR_K}).")
    ap.add_argument("--protect-pcs", type=int, default=DEFAULT_PROTECT_PCS)
    ap.add_argument("--transform", choices=["d", "logit", "raw"], default="d",
                    help="'d' = 2*p_bar - 1, the identity's own term; it beat "
                         "'logit' at every k on the real data.")
    ap.add_argument("--no-residual", action="store_true",
                    help="Skip the stage-3 residual output.")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write (default: the input directory).")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    d = Path(a.concordance_dir)
    out_dir = Path(a.output_dir) if a.output_dir else d
    out_dir.mkdir(parents=True, exist_ok=True)
    marg = _load(Path(a.marginal) if a.marginal
                 else d / "concordance_marginal.parquet", "marginal matrix")
    base = _load(Path(a.baseline) if a.baseline
                 else d / "accessibility_mean.parquet", "baseline matrix")
    extra = [_load(Path(p), "extra baseline") for p in (a.extra_baseline or [])]

    s = pd.read_csv(a.samples, sep="\t")
    cell = next((c for c in ("cell_name", "cell", "cell_id")
                 if c in s.columns), s.columns[0])
    if a.batch_col not in s.columns:
        raise SystemExit(f"--batch-col {a.batch_col!r} not in {a.samples} "
                         f"(has {list(s.columns)})")
    batch = s.set_index(cell)[a.batch_col].astype(str)
    n_lev = batch.nunique()
    logger.info(f"{n_lev} replicate level(s) in {a.batch_col!r}: "
                f"{sorted(batch.unique())[:6]}")
    if n_lev < 2:
        logger.warning("Only one replicate level; stage 2 is a no-op.")

    stats = [x.strip() for x in a.stats.split(",") if x.strip()]
    written = []
    for stat in stats:
        src = d / f"concordance_{stat}.parquet"
        if not src.exists():
            logger.warning(f"{src.name} not found; skipping {stat}.")
            continue
        M = pd.read_parquet(src)
        logger.info(f"{stat}: {M.shape[0]} cells x {M.shape[1]} bins")
        two = two_stage(M, marg, batch, baseline=base, k=a.factor_k,
                        anchor_k=a.anchor_k, transform=a.transform,
                        n_pcs=a.protect_pcs)
        stem = f"concordance_{stat}.margreg_k{a.factor_k}.method1_local"
        p2 = out_dir / f"{stem}.parquet"
        two.to_parquet(p2)
        written.append(p2)
        moved = float(np.nanmean(np.abs(two.to_numpy(float)
                                        - M.to_numpy(float))))
        logger.info(f"  two-stage   -> {p2.name}  mean |shift| {moved:.4f}")
        if not a.no_residual:
            three = stage3_residual(two, [base] + extra)
            p3 = out_dir / f"{stem}.resid.parquet"
            three.to_parquet(p3)
            written.append(p3)
            logger.info(f"  three-stage -> {p3.name}  "
                        f"(residual against {1 + len(extra)} baseline(s))")
    if not written:
        raise SystemExit("nothing written; check --stats against the "
                         "concordance_*.parquet files present")
    print("\n".join(str(p) for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
