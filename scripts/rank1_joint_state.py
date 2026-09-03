#!/usr/bin/env python3
"""Part B: the rank-1 test on Sigma_AB, per (group, anchor bin, stratum).

    scripts/rank1_joint_state.py \
        --posteriors output/joint/03j_read_end_posteriors \
        --samples mouse_cortex_1577_cells.samples.with_batches.fixed.tsv \
        --group-order "IT-L2/3,IT-L4,IT-L5/6,CT-L6,Inhib,ODC,OPC,ASC,MGC,PC/EC" \
        -o output/joint

Reads Part A's per-end posteriors, forms the anchor/partner units of Section
B.1, accumulates the Section B.2 moments, and runs the Section B.6 nulls.

What comes out is 04_rank1.parquet plus one BED per class.  **Class II --
coordination the shared open/closed axis does not explain -- ranked by
D_excess is the primary figure object**, and D_excess, not D, is what the
ranking uses: |D| has a strictly positive null expectation at finite n, so
ranking on the raw value would put the thinnest bins on top.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nome_hic_concordance import rank1_joint as rj                # noqa: E402
from nome_hic_concordance.rank1_units import (                    # noqa: E402
    FLAGGED_STRATA, STRATA, STRATUM_NAMES, pair_ends, stratum_of, weights_for)
from nome_hic_concordance.distance_stats import fdr               # noqa: E402
from nome_hic_concordance.samples_manifest import (               # noqa: E402
    add_manifest_args, load_from_args, safe_name)
from nome_hic_concordance.utils import ensure_dir, setup_logging  # noqa: E402

logger = logging.getLogger("rank1_joint_state")

OUT_NAME = "04_rank1.parquet"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_manifest_args(p)
    p.add_argument("--posteriors", required=True,
                   help="Part A's 03j_read_end_posteriors directory.")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--bin-size", type=int, default=250_000)
    p.add_argument("--group-by", default="group",
                   choices=("group", "all"),
                   help="'group' uses the manifest's cell types; 'all' pools "
                        "every cell into one group.")
    p.add_argument("--moment-mode", default="complete",
                   choices=("complete", "pairwise"),
                   help="B.4.  'complete' needs both marks on both ends; "
                        "'pairwise' accumulates each entry over the contacts "
                        "carrying the marks that entry needs, which with "
                        "--min-hcg-sites 1 is substantially more data.  Report "
                        "both in the first analysis.")
    p.add_argument("--error-convention", default="raw",
                   choices=("raw", "posterior_mean", "none"),
                   help="B.4.  Recorded per row; pick it from the B.7.1 "
                        "coverage-flatness diagnostic, not by taste.")
    p.add_argument("--weight-cap-pct", type=float, default=99.0,
                   help="Within-stratum percentile the B.2 weight is capped "
                        "at, so one very deep contact cannot carry a bin.")
    p.add_argument("--min-gch-sites", type=int, default=0,
                   help="Require this many GCH sites on BOTH ends.  0 keeps "
                        "whatever Part A wrote.")
    p.add_argument("--min-hcg-sites", type=int, default=0,
                   help="Require this many HCG sites on BOTH ends.  Applied "
                        "HERE, not by re-extracting: Part A emits "
                        "permissively precisely so this ablation (B.7.7) is a "
                        "filter rather than another pass over the BAMs.")
    p.add_argument("--gch-norm", default=None,
                   choices=("location", "full", "none"),
                   help="Override Part A's centring using the persisted "
                        "prior_sd columns (B.7.6).  Default: use the "
                        "centred_* columns as written.")
    p.add_argument("--min-contacts", type=int, default=30)
    p.add_argument("--min-neff", type=float, default=30.0)
    p.add_argument("--require-conversion-pass", action="store_true",
                   help="Drop ends whose molecule failed Part A's HCH "
                        "conversion floor.  Off by default: Part A marks "
                        "rather than drops so the threshold stays revisable.")
    p.add_argument("--n-boot-null", type=int, default=500)
    p.add_argument("--n-boot-mate", type=int, default=200)
    p.add_argument("--n-boot-ci", type=int, default=500)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--chrom", default=None,
                   help="Restrict to one chromosome (the B.7.6/B.7.7 "
                        "ablations run on one).")
    return p


def recentre(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Re-derive centred_gch under a different --gch-norm (B.7.6).

    Part A persists theta, mu and the grid-cell sd, so the ablation is a column
    division here rather than a second pass over every BAM.  Only GCH moves:
    A.3.4 is about GCH normalization, and rescaling HCG as well would change
    two things at once and make the comparison unreadable.
    """
    out = df.copy()
    if mode == "none":
        out["centred_gch"] = out["theta_gch"]
    elif mode == "location":
        out["centred_gch"] = out["theta_gch"] - out["mu_gch"]
    else:
        sd = out["prior_sd_gch"].to_numpy()
        out["centred_gch"] = ((out["theta_gch"] - out["mu_gch"]).to_numpy()
                              / np.maximum(sd, 1e-6))
    return out


# ------------------------------------------------------------------- driver ---

def _unit_rows(u: pd.DataFrame, args, group: str, chrom: str) -> dict | None:
    """One (group, bin, stratum) row, or None when the unit is too thin."""
    present_A = np.column_stack([u["A_n_gch"] > 0, u["A_n_hcg"] > 0])
    present_B = np.column_stack([u["B_n_gch"] > 0, u["B_n_hcg"] > 0])
    if args.moment_mode == "complete":
        keep = present_A.all(1) & present_B.all(1)
        u = u.loc[keep]
        present_A, present_B = present_A[keep], present_B[keep]
    if len(u) < args.min_contacts:
        return None

    tA = u[["A_gch", "A_hcg"]].to_numpy()
    tB = u[["B_gch", "B_hcg"]].to_numpy()
    vA = u[["A_v_gch", "A_v_hcg"]].to_numpy()
    vB = u[["B_v_gch", "B_v_hcg"]].to_numpy()
    w = weights_for(u, args.weight_cap_pct)

    if args.moment_mode == "pairwise":
        acc40 = rj.accumulate_pairwise(rj.new_acc_pairwise(), tA, tB, vA, vB,
                                       present_A, present_B, w)
        m = rj.moments_pairwise(acc40)
        acc = rj.new_acc()
        for i, f in enumerate(rj.ACC_FIELDS):     # a complete-case shadow, so
            acc[i] = acc40[i]                     # summarize() has one shape
    else:
        acc = rj.accumulate(rj.new_acc(), tA, tB, vA, vB, w)
        m = rj.moments(acc)

    per_cell = [rj.accumulate(rj.new_acc(),
                              tA[idx], tB[idx], vA[idx], vB[idx], w[idx])
                for idx in (u.groupby("cell_id", sort=True).indices.values())]

    row = rj.summarize(
        acc, per_cell_accs=per_cell, theta_A=tA, theta_B=tB, v_A=vA, v_B=vB,
        weights=w, n_boot_null=args.n_boot_null, n_boot_mate=args.n_boot_mate,
        n_boot_ci=args.n_boot_ci, seed=args.seed, alpha_rank1=args.alpha,
        error_convention=args.error_convention, moment_mode=args.moment_mode,
        min_neff=args.min_neff)
    if args.moment_mode == "pairwise":
        # summarize() read the complete-case shadow; overwrite the moment-derived
        # numbers with the pairwise ones so the row says what the flag claims.
        wh = rj.whiten(m["S_AA"], m["S_BB"], m["S_AB"], m["V_A"], m["V_B"])
        cn = rj.canonical(wh["R"])
        row.update({"D": rj.det_statistic(m["S_AB"]),
                    "rho1": cn["rho1"], "rho2": cn["rho2"],
                    "rho2_over_rho1": cn["rho2_over_rho1"],
                    "N_eff": m["N_eff"],
                    "flag_pd_clipped": wh["flag_pd_clipped"]})

    bs = int(args.bin_size)
    anchor = int(u["anchor_bin"].iloc[0])
    row.update({
        "group": group, "chrom": chrom,
        "bin_start": anchor * bs, "bin_end": (anchor + 1) * bs,
        "stratum": u["stratum"].iloc[0],
        "flag_short_range": u["stratum"].iloc[0] in FLAGGED_STRATA,
        "flag_pairwise_incomplete": bool(
            (~(present_A.all(1) & present_B.all(1))).any()),
        "n_both_ends_in_bin": int(u["both_ends_in_bin"].sum()),
    })
    return row


def run(args, manifest) -> pd.DataFrame:
    root = Path(args.posteriors)
    shards = sorted(root.glob("chrom=*/*.parquet"))
    if not shards:
        raise SystemExit(f"no Part A shards under {root}")
    if args.chrom:
        shards = [s for s in shards if s.parent.name == f"chrom={args.chrom}"]
        if not shards:
            raise SystemExit(f"--chrom {args.chrom} matched no shard")

    cell_group = (manifest.table.set_index("name")["group"].to_dict()
                  if args.group_by == "group"
                  else {c: "all" for c in manifest.cells})

    by_chrom: dict[str, list[Path]] = {}
    for s in shards:
        by_chrom.setdefault(s.parent.name.split("=", 1)[1], []).append(s)

    rows = []
    for chrom, paths in sorted(by_chrom.items()):
        df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        if args.require_conversion_pass:
            n0 = len(df)
            df = df[df["conversion_pass"]]
            logger.info("  %s: conversion filter dropped %s/%s ends",
                        chrom, f"{n0 - len(df):,}", f"{n0:,}")
        df["group"] = df["cell_id"].map(cell_group)
        df = df[df["group"].notna()]
        if args.min_gch_sites or args.min_hcg_sites:
            n0 = len(df)
            keep = ((df["n_gch"] >= args.min_gch_sites)
                    & (df["n_hcg"] >= args.min_hcg_sites))
            df = df[keep]
            logger.info("  %s: site minima dropped %s/%s ends",
                        chrom, f"{n0 - len(df):,}", f"{n0:,}")
        if args.gch_norm is not None:
            df = recentre(df, args.gch_norm)
        if not len(df):
            continue

        for group, sub in df.groupby("group", sort=True):
            units = pair_ends(sub, args.bin_size)
            if not len(units):
                continue
            n_units = 0
            for _, u in units.groupby(["anchor_bin", "stratum"], sort=True):
                row = _unit_rows(u, args, str(group), chrom)
                if row is not None:
                    rows.append(row)
                    n_units += 1
            logger.info("  %s / %s: %d unit(s) from %s contact-orientations",
                        chrom, group, n_units, f"{len(units):,}")

    if not rows:
        raise SystemExit("no (group, bin, stratum) unit met --min-contacts")
    out = pd.DataFrame(rows)

    # B.6.4 -- BH within group, over (bin x stratum).  Pooling across groups
    # would let a well-covered cell type set the threshold for a rare one.
    out["q_rank1"] = np.nan
    for g, idx in out.groupby("group").groups.items():
        p = out.loc[idx, "p_rank1"].to_numpy(float)
        ok = np.isfinite(p)
        q = np.full(len(p), np.nan)
        if ok.any():
            q[ok] = fdr(p[ok])
        out.loc[idx, "q_rank1"] = q
    out["significant_rank1"] = out["q_rank1"] < args.alpha
    # The class uses the FDR-adjusted rank test, not the raw p.
    out["bin_class"] = [
        rj.bin_class(bool(r.rho1_significant),
                     bool(np.isfinite(r.q_rank1) and r.q_rank1 < args.alpha),
                     int(r.sign_rho_u))
        for r in out.itertuples()]
    out["bin_class_label"] = out["bin_class"].map(rj.BIN_CLASSES)
    return out


def write_beds(out: pd.DataFrame, out_dir: Path) -> list[str]:
    """One BED per class, for intersection against compartments, PMDs, gene
    bodies, replication timing and CTCF.

    Class II is sorted by D_excess descending -- the EXCESS over the null, not
    the raw |D|, which has a strictly positive null expectation at finite n and
    would rank the thinnest bins first.
    """
    paths = []
    for cls, label in rj.BIN_CLASSES.items():
        sub = out[out["bin_class"] == cls]
        if not len(sub):
            continue
        sub = sub.sort_values("D_excess", ascending=False)
        p = out_dir / f"class{cls}_{label}.bed"
        with open(p, "w") as fh:
            for r in sub.itertuples():
                score = 0 if not np.isfinite(r.D_excess) else \
                    int(np.clip(round(abs(r.D_excess) * 1000), 0, 1000))
                fh.write(f"{r.chrom}\t{r.bin_start}\t{r.bin_end}\t"
                         f"{safe_name(str(r.group))}|{r.stratum}\t{score}\t.\n")
        paths.append(str(p))
    return paths


def qc_tables(out: pd.DataFrame, out_dir: Path) -> dict:
    """The Section B.7 acceptance criteria that are computable from the rows.

    B.7.2 (distance decay), B.7.5 (mate-shuffle sanity) and the class census
    are computable from the rows alone, so they land here.  The rest of
    Section B.7 -- the contact-level panels, the null calibration and the two
    ablations -- is ``scripts/rank1_diagnostics.py``.  Both use the SAME
    functions from ``rank1_diagnostics``, so the summary here cannot drift
    from the one on the figure.
    """
    from nome_hic_concordance import rank1_diagnostics as dx

    qc = ensure_dir(out_dir / "qc")
    dx.distance_decay(out, STRATUM_NAMES).to_csv(
        qc / "B72_distance_decay.tsv", sep="\t", index=False)
    dx.mate_shuffle_sanity(out).to_csv(
        qc / "B75_mate_shuffle_rho1.tsv", sep="\t", index=False)

    census = (out.groupby(["group", "bin_class", "bin_class_label"])
                 .size().rename("n_bins").reset_index())
    census.to_csv(qc / "bin_class_census.tsv", sep="\t", index=False)
    return {"distance_decay": str(qc / "B72_distance_decay.tsv"),
            "mate_shuffle": str(qc / "B75_mate_shuffle_rho1.tsv"),
            "class_census": str(qc / "bin_class_census.tsv")}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    manifest = load_from_args(args, require_bam=False, require_index=False)
    manifest.report()

    out_dir = ensure_dir(args.output)
    out = run(args, manifest)
    out.to_parquet(out_dir / OUT_NAME, index=False)
    beds = write_beds(out, out_dir)
    qc = qc_tables(out, out_dir)

    census = out["bin_class"].value_counts().to_dict()
    logger.info("Bin classes: %s", ", ".join(
        f"{c}={census.get(c, 0)} ({rj.BIN_CLASSES[c]})"
        for c in ("I", "II", "III", "IV")))
    n_ii = int((out["bin_class"] == "II").sum())
    logger.info("Class II (the headline: coordination the shared axis does "
                "not explain): %d bin-strata", n_ii)
    if n_ii and out.loc[out["bin_class"] == "II", "flag_pd_clipped"].mean() > 0.5:
        logger.warning(
            "  more than half of the class II calls had a clipped eigenvalue. "
            "With --min-hcg-sites 1 that is the expected failure mode, and it "
            "means rho2 is not trustworthy there -- D is.")

    (out_dir / "_rank1_metadata.json").write_text(json.dumps({
        "command": " ".join(sys.argv),
        "n_rows": len(out), "bin_class_counts": census,
        "moment_mode": args.moment_mode,
        "error_convention": args.error_convention,
        "bin_size": args.bin_size, "alpha": args.alpha,
        "min_gch_sites": args.min_gch_sites,
        "min_hcg_sites": args.min_hcg_sites,
        "gch_norm_override": args.gch_norm,
        "strata": STRATUM_NAMES, "flagged_strata": sorted(FLAGGED_STRATA),
        "beds": beds, "qc": qc,
        "manifest": manifest.metadata(),
    }, indent=2, default=str))
    logger.info("Done -> %s", out_dir / OUT_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
