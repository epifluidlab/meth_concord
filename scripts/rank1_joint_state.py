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
from nome_hic_concordance.distance_stats import fdr               # noqa: E402
from nome_hic_concordance.samples_manifest import (               # noqa: E402
    add_manifest_args, load_from_args, safe_name)
from nome_hic_concordance.utils import ensure_dir, setup_logging  # noqa: E402

logger = logging.getLogger("rank1_joint_state")

#: Section B.1 strata.  Named and fixed rather than log-spaced: the 1-10 kb
#: stratum is FLAGGED (residual self-ligation lives there) rather than dropped,
#: and trans is its own stratum, never folded into the widest cis one.
STRATA = [
    ("1-10kb", 1e3, 1e4, True),
    ("10-100kb", 1e4, 1e5, False),
    ("100kb-1Mb", 1e5, 1e6, False),
    ("1-10Mb", 1e6, 1e7, False),
    (">10Mb", 1e7, np.inf, False),
    ("trans", np.nan, np.nan, False),
]
STRATUM_NAMES = [s[0] for s in STRATA]
FLAGGED_STRATA = {s[0] for s in STRATA if s[3]}

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


# ------------------------------------------------------------------- units ---

def stratum_of(distance, is_cis) -> np.ndarray:
    """Section B.1 stratum name per contact."""
    d = np.asarray(distance, float)
    cis = np.asarray(is_cis, bool)
    out = np.full(len(d), "", dtype=object)
    out[~cis] = "trans"
    for name, lo, hi, _ in STRATA[:-1]:
        m = cis & (d >= lo) & (d < hi)
        out[m] = name
    return out


def pair_ends(df: pd.DataFrame, bin_size: int) -> pd.DataFrame:
    """One row per (contact, orientation): anchor end A, partner end B.

    Section B.1: every contact appears once per end, so a bin sees every
    contact that touches it with that end as the anchor.  When both ends land
    in the same bin the contact contributes both orientations; they are marked
    so nothing downstream treats them as two independent observations.
    """
    e1 = df[df["end_idx"] == 1].set_index(["cell_id", "pair_idx"])
    e2 = df[df["end_idx"] == 2].set_index(["cell_id", "pair_idx"])
    common = e1.index.intersection(e2.index)
    if not len(common):
        return pd.DataFrame()
    e1, e2 = e1.loc[common], e2.loc[common]

    def side(a, b, tag):
        pos = np.minimum(a["pos_5p"].to_numpy(), a["pos_3p"].to_numpy())
        out = pd.DataFrame({
            "cell_id": a.index.get_level_values(0),
            "pair_idx": a.index.get_level_values(1),
            "anchor_bin": pos // bin_size,
            "orientation": tag,
            "contact_dist": a["contact_dist"].to_numpy(),
            "is_cis": a["is_cis"].to_numpy(),
            "conversion_pass": (a["conversion_pass"].to_numpy()
                                & b["conversion_pass"].to_numpy()),
        })
        for side_tag, frame in (("A", a), ("B", b)):
            for mark in ("gch", "hcg"):
                out[f"{side_tag}_{mark}"] = frame[f"centred_{mark}"].to_numpy()
                out[f"{side_tag}_v_{mark}"] = frame[f"var_{mark}"].to_numpy()
                out[f"{side_tag}_n_{mark}"] = frame[f"n_{mark}"].to_numpy()
        return out

    both = side(e1, e2, 1)
    flip = side(e2, e1, 2)
    out = pd.concat([both, flip], ignore_index=True)
    same = out.groupby(["cell_id", "pair_idx"])["anchor_bin"].transform("nunique")
    out["both_ends_in_bin"] = same == 1
    out["stratum"] = stratum_of(out["contact_dist"], out["is_cis"])
    return out[out["stratum"] != ""].reset_index(drop=True)


def weights_for(u: pd.DataFrame, cap_pct: float) -> np.ndarray:
    """Section B.2 inverse total-variance weight, capped within stratum.

    Deliberately NOT a function of |theta - 0.5|: a posterior near 0.5 may be
    uninformative or a genuine intermediate state, and one weight cannot tell
    the two apart.
    """
    v = (u["A_v_gch"] + u["A_v_hcg"] + u["B_v_gch"] + u["B_v_hcg"]).to_numpy()
    w = 1.0 / (v + 1e-12)
    for s in pd.unique(u["stratum"]):
        m = (u["stratum"] == s).to_numpy()
        if m.sum() > 1:
            w[m] = np.minimum(w[m], np.percentile(w[m], cap_pct))
    return w


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

    B.7.1 coverage flatness and B.7.3 conversion association need
    contact-level stratification and are emitted by --moment-mode/--chrom
    re-runs; what is here is B.7.2 (distance decay), B.7.5 (mate-shuffle
    sanity) and the class census.
    """
    qc = ensure_dir(out_dir / "qc")
    decay = (out.groupby(["group", "stratum"])
                .agg(rho1=("rho1", "median"), rho2=("rho2", "median"),
                     D=("D", "median"), D_excess=("D_excess", "median"),
                     n=("rho1", "size")).reset_index())
    decay["stratum"] = pd.Categorical(decay["stratum"],
                                      categories=STRATUM_NAMES, ordered=True)
    decay = decay.sort_values(["group", "stratum"])
    decay.to_csv(qc / "B72_distance_decay.tsv", sep="\t", index=False)

    mate = (out.groupby("group")["rho1_mate_shuffle_mean"]
               .agg(["median", "mean", "std", "size"]).reset_index())
    mate.to_csv(qc / "B75_mate_shuffle_rho1.tsv", sep="\t", index=False)

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
        "strata": STRATUM_NAMES, "flagged_strata": sorted(FLAGGED_STRATA),
        "beds": beds, "qc": qc,
        "manifest": manifest.metadata(),
    }, indent=2, default=str))
    logger.info("Done -> %s", out_dir / OUT_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
