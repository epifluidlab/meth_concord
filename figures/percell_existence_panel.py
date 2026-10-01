#!/usr/bin/env python3
"""Figure 5c: is long-range concordance detectable inside individual cells?

Panel a pools pairs from every cell into one correlation per distance, which
shows concordance exists on average.  This panel asks the single-cell question:
for EACH cell, are its linked pairs more concordant than its own matched
unlinked pairs?  Same statistic as panel a, computed per cell:

    delta_c = r(linked pairs of cell c) - r(matched unlinked pairs of cell c)
    z_c     = delta_c / se_c

with a within-cell mate-shuffle giving the null.  A correlation subtracts each
end's own mean, so a difference in marginal between linked and unlinked ends
cannot fake a signal -- which is why this panel uses panel a's statistic and
not the marginal-corrected C (that matrix is centred to zero per bin, so the
level this panel tests is gone from it by construction).

Layout, one row per mark:
  (i)   per-cell z, observed vs within-cell null, with how many cells pass
  (ii)  per-cell delta by cell type, in --group-order, with % significant
  (iii) z against linked pairs per cell: how deep a cell must be sequenced

Input is a per-cell table (one row per cell x mark):
    cell_id, group, mark, n_pairs, delta, se, z, z_null[, simulated]

    figures/percell_existence_panel.py --simulate -o figures/percell_existence_panel.simulated
    figures/percell_existence_panel.py --table per_cell_existence.tsv -o fig5c

Anything drawn from simulated input carries a stamp saying so, on the figure
itself, so a mock-up can never be mistaken for a result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The statistics and the figure live in the package, so the real run
# (scripts/end_posterior_distance_correlation.py) and this mock-up draw the
# same panel from the same code.  Re-exported here for the CLI and the tests.
from nome_hic_concordance.percell_existence import (           # noqa: E402,F401
    MARKS, MARK_COLOR, MARK_LABEL, NULL_GREY, REQUIRED, SIM_STAMP, add_q,
    one_sided_p, pass_depth, plot, running_median, z_threshold)

GROUP_ORDER = ["IT-L2/3", "IT-L4", "IT-L5/6", "CT-L6", "Inhib",
               "ODC", "OPC", "ASC", "MGC", "PC/EC"]
#: Placeholders.  The real run takes colours from the manifest's
#: --color-column so this panel matches the UMAP and heatmap.
PLACEHOLDER_COLORS = ["#E69F00", "#D55E00", "#CC79A7", "#882255", "#0072B2",
                      "#009E73", "#56B4E9", "#999933", "#44AA99", "#777777"]


# ----------------------------------------------------------------- simulation

def simulate(seed: int = 0) -> pd.DataFrame:
    """Per-cell table with assumed effect sizes and depths.  For LAYOUT only.

    The assumptions, so it is clear what drives the picture:
      * genome-wide delta per cell: GCH ~0.03-0.05, HCG ~0.09-0.12, set from
        panel a's beyond-1-kb range, slightly higher in neurons;
      * informative linked pairs per cell: log-normal, GCH median 12-18 k,
        HCG 4-6 k (both ends must carry the mark, and CpGs are sparse);
      * se = sqrt(1/n + 1/(3n)): three matched unlinked draws per linked pair.
    The % of significant cells the figure shows follows directly from these
    numbers and says nothing about the real data.
    """
    rng = np.random.default_rng(seed)
    counts = [260, 140, 210, 160, 215, 290, 75, 120, 60, 47]      # sum 1577
    neuron = {"IT-L2/3", "IT-L4", "IT-L5/6", "CT-L6", "Inhib"}
    delta_mu = {
        "GCH": {"IT-L2/3": .046, "IT-L4": .044, "IT-L5/6": .048, "CT-L6": .050,
                "Inhib": .041, "ODC": .034, "OPC": .039, "ASC": .031,
                "MGC": .027, "PC/EC": .030},
        "HCG": {"IT-L2/3": .118, "IT-L4": .112, "IT-L5/6": .121, "CT-L6": .124,
                "Inhib": .108, "ODC": .094, "OPC": .101, "ASC": .090,
                "MGC": .085, "PC/EC": .088},
    }
    delta_sd = {"GCH": 0.012, "HCG": 0.028}
    depth_median = {"GCH": (18_000, 12_000), "HCG": (6_000, 4_000)}

    rows = []
    cell_i = 0
    for g, n_g in zip(GROUP_ORDER, counts):
        for _ in range(n_g):
            cid = f"cell{cell_i:04d}"
            cell_i += 1
            depth_factor = rng.lognormal(0.0, 0.6)   # shared by both marks
            for mark in MARKS:
                med = depth_median[mark][0 if g in neuron else 1]
                n = max(150, int(med * depth_factor * rng.lognormal(0.0, 0.15)))
                se = np.sqrt(1.0 / n + 1.0 / (3 * n))
                true = rng.normal(delta_mu[mark][g], delta_sd[mark])
                delta = true + rng.normal(0.0, se)
                z_null = rng.normal(0.0, 1.0)
                rows.append({
                    "cell_id": cid, "group": g, "mark": mark, "n_pairs": n,
                    "delta": delta, "se": se, "z": delta / se,
                    "z_null": z_null, "delta_null": z_null * se,
                    "simulated": True,
                })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--table", help="per-cell TSV (columns: %s)" % ", ".join(REQUIRED))
    src.add_argument("--simulate", action="store_true",
                     help="draw from simulated input; the figure is stamped")
    ap.add_argument("-o", "--out", required=True, help="output prefix")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    df = simulate(args.seed) if args.simulate else pd.read_csv(args.table, sep="\t")
    df = add_q(df)
    plot(df, group_order=GROUP_ORDER,
         group_colors=dict(zip(GROUP_ORDER, PLACEHOLDER_COLORS)),
         out_prefix=args.out)
    for mark, d in df.groupby("mark"):
        n_sig = int((d["q"] < 0.05).sum())
        print(f"{mark}: {n_sig}/{len(d)} cells q<0.05 "
              f"(null {int((d['q_null'] < 0.05).sum())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
