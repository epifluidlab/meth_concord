#!/usr/bin/env python3
"""Section B.7: the acceptance criteria for the Part B rank-1 test.

    scripts/rank1_diagnostics.py \
        --posteriors output/joint/03j_read_end_posteriors \
        --rank1 output/joint/04_rank1.parquet \
        --samples mouse_cortex_1577_cells.samples.with_batches.fixed.tsv \
        --group-order "IT-L2/3,IT-L4,..." \
        -o output/joint/qc

Seven panels, each a pass/fail on the analysis rather than a decoration:

  B.7.1 coverage flatness      -> which error_convention to use
  B.7.2 distance decay         -> is rho2 a real second axis or rho1's shadow
  B.7.3 conversion association -> is the A.3.5 threshold permissive
  B.7.4 null calibration       -> the licence for every rank-2 call
  B.7.5 mate-shuffle sanity    -> rho1 must collapse when mates are reassigned
  B.7.6 normalization ablation -> justifies --gch-norm location
  B.7.7 min-hcg-sites ablation -> justifies --min-hcg-sites 1

The two ablations run as FILTERS over Part A's output, not as re-extractions:
A.1 emits permissively and A.3.4's grid-cell sd is persisted, which is exactly
what makes both a column operation here.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nome_hic_concordance import figure_style                     # noqa: E402
from nome_hic_concordance import rank1_diagnostics as dx          # noqa: E402
from nome_hic_concordance.rank1_units import (                    # noqa: E402
    STRATUM_NAMES, pair_ends)
from nome_hic_concordance.samples_manifest import (               # noqa: E402
    add_manifest_args, load_from_args)
from nome_hic_concordance.utils import ensure_dir, setup_logging  # noqa: E402

logger = logging.getLogger("rank1_diagnostics")

_R1 = Path(__file__).resolve().parent / "rank1_joint_state.py"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_manifest_args(p)
    p.add_argument("--posteriors", required=True)
    p.add_argument("--rank1", required=True, help="04_rank1.parquet")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--bin-size", type=int, default=250_000)
    p.add_argument("--chrom", default=None,
                   help="Chromosome the contact-level panels and the ablations "
                        "use.  Defaults to the first shard; B.7.6 says one "
                        "chromosome is enough for the normalization ablation.")
    p.add_argument("--plot-format", default="pdf",
                   choices=("pdf", "png", "svg"))
    p.add_argument("--null-units", type=int, default=60)
    p.add_argument("--null-boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-ablations", action="store_true",
                   help="B.7.6 and B.7.7 re-run Part B several times; skip "
                        "them for a quick pass over the other five.")
    p.add_argument("--ablation-args", default="",
                   help="Extra flags passed through to every ablation arm, "
                        "space separated.")
    return p


# ------------------------------------------------------------------ figures ---

def _save(fig, out_dir: Path, name: str, fmt: str) -> str:
    path = out_dir / f"{name}.{fmt}"
    fig.savefig(path, bbox_inches="tight", dpi=600 if fmt == "png" else None)
    return str(path)


def _trend_panel(ax, t, xcol, xlabel, title):
    import matplotlib.pyplot as plt  # noqa: F401
    for i, k in enumerate(("rho1", "rho2", "D")):
        ax.plot(t[xcol], t[k], "o-", ms=3, lw=1,
                color=figure_style.PALETTE[i], label=k)
    tr = t.attrs.get("trend", {})
    ax.set_xlabel(xlabel)
    ax.set_ylabel("statistic")
    ax.set_title(title + "\n" + "  ".join(
        f"Δ{k}={tr.get(k, np.nan):+.3f}" for k in ("rho1", "rho2", "D")))
    ax.axhline(0, color="0.7", lw=0.5, zorder=0)
    ax.legend(loc="best")


def figure_coverage_and_conversion(cov, conv, out_dir, fmt) -> list[str]:
    """B.7.1 and B.7.3 side by side -- both are 'is this flat' panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context(figure_style.RC):
        fig, axes = plt.subplots(1, 2, figsize=(figure_style.FIG_W_IN, 2.6))
        if len(cov):
            _trend_panel(axes[0], cov, "mean_depth", "sites per contact",
                         "B.7.1 coverage flatness")
        else:
            axes[0].set_axis_off()
            axes[0].text(0.5, 0.5, "B.7.1: too few contacts", ha="center")
        if len(conv):
            _trend_panel(axes[1], conv, "mean_conversion",
                         "per-molecule conversion",
                         "B.7.3 conversion association")
        else:
            axes[1].set_axis_off()
            axes[1].text(0.5, 0.5, "B.7.3: no conversion spread", ha="center")
        fig.tight_layout()
        out = _save(fig, out_dir, "B71_B73_flatness", fmt)
        plt.close(fig)
    return [out]


def figure_decay_and_shuffle(decay, shuffle, out_dir, fmt) -> list[str]:
    """B.7.2 and B.7.5."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = [s for s in STRATUM_NAMES if s in set(decay["stratum"].astype(str))]
    x = np.arange(len(order))
    with plt.rc_context(figure_style.RC):
        fig, axes = plt.subplots(1, 3, figsize=(figure_style.FIG_W_IN, 2.6))
        for gi, (g, sub) in enumerate(decay.groupby("group", observed=True)):
            sub = sub.set_index(sub["stratum"].astype(str)).reindex(order)
            col = figure_style.PALETTE[gi % len(figure_style.PALETTE)]
            axes[0].plot(x, sub["rho1"], "o-", ms=3, lw=1, color=col, label=str(g))
            axes[1].plot(x, sub["rho2"], "o-", ms=3, lw=1, color=col)
            axes[2].plot(x, sub["rho2_rel"], "o-", ms=3, lw=1, color=col)
        for a, lab in zip(axes, ("rho1", "rho2", "rho2/rho1, relative")):
            a.set_xticks(x)
            a.set_xticklabels(order, rotation=45, ha="right")
            a.set_ylabel(lab)
        axes[0].set_title("B.7.2 distance decay")
        # Flat at 1 is what rho2 being a rescaled shadow of rho1 looks like.
        axes[2].axhline(1.0, color="0.6", lw=0.6, ls="--")
        axes[2].set_title("flat = rho2 tracks rho1")
        if decay["group"].nunique() <= 8:
            axes[0].legend(loc="best")
        fig.tight_layout()
        out = [_save(fig, out_dir, "B72_distance_decay", fmt)]
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(figure_style.FIG_W_IN / 2.5, 2.4))
        y = np.arange(len(shuffle))
        ax.barh(y - 0.2, shuffle["rho1_observed"], height=0.38,
                color=figure_style.PALETTE[0], label="observed")
        ax.barh(y + 0.2, shuffle["rho1_shuffled"], height=0.38,
                color=figure_style.PALETTE[1], label="mate-shuffled")
        ax.set_yticks(y)
        ax.set_yticklabels(shuffle["group"].astype(str))
        ax.set_xlabel("median rho1")
        ax.set_title("B.7.5 mate shuffle must collapse rho1")
        ax.legend(loc="best")
        fig.tight_layout()
        out.append(_save(fig, out_dir, "B75_mate_shuffle", fmt))
        plt.close(fig)
    return out


def figure_null_calibration(null_tables, out_dir, fmt) -> list[str]:
    """B.7.4 -- the QQ that licenses every rank-2 call."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context(figure_style.RC):
        fig, ax = plt.subplots(figsize=(figure_style.FIG_W_IN / 2.5, 2.6))
        for i, (name, t) in enumerate(null_tables.items()):
            p = np.sort(t["p_rank1"].to_numpy())
            exp = (np.arange(1, len(p) + 1) - 0.5) / len(p)
            ax.plot(exp, p, "o", ms=2.5,
                    color=figure_style.PALETTE[i % len(figure_style.PALETTE)],
                    label=f"{name} (KS p={t.attrs['ks_uniform_p']:.2f})")
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.6, ls="--")
        ax.set_xlabel("expected uniform quantile")
        ax.set_ylabel("observed p (B.6.2)")
        ax.set_title("B.7.4 null calibration")
        ax.legend(loc="upper left")
        fig.tight_layout()
        out = _save(fig, out_dir, "B74_null_calibration", fmt)
        plt.close(fig)
    return [out]


def figure_ablations(summaries, merged, out_dir, fmt) -> list[str]:
    """B.7.6 and B.7.7, per bin rather than per distribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    with plt.rc_context(figure_style.RC):
        for key, s in summaries.items():
            m = merged[key]
            arms = m.attrs["arms"]
            fig, axes = plt.subplots(1, len(arms) - 1 + 1,
                                     figsize=(figure_style.FIG_W_IN, 2.6),
                                     squeeze=False)
            axes = axes[0]
            base = arms[0]
            for j, arm in enumerate(arms[1:]):
                a = axes[j]
                a.scatter(m[f"D__{base}"], m[f"D__{arm}"], s=6,
                          color=figure_style.PALETTE[0], alpha=0.6)
                lim = np.nanpercentile(
                    np.concatenate([m[f"D__{base}"].to_numpy(),
                                    m[f"D__{arm}"].to_numpy()]), [1, 99])
                a.plot(lim, lim, color="0.6", lw=0.6, ls="--")
                a.set_xlabel(f"D  ({base})")
                a.set_ylabel(f"D  ({arm})")
                a.set_title(f"{key}: {base} vs {arm}")
            a = axes[-1]
            a.bar(np.arange(len(s)), s["median_shift_vs_first"],
                  color=figure_style.PALETTE[1])
            a.set_xticks(np.arange(len(s)))
            a.set_xticklabels(s["arm"], rotation=45, ha="right")
            a.set_ylabel("median per-bin shift in D")
            a.axhline(0, color="0.6", lw=0.6)
            fig.tight_layout()
            paths.append(_save(fig, out_dir, f"{key}_ablation", fmt))
            plt.close(fig)
    return paths


# ---------------------------------------------------------------- ablations ---

def _run_arm(args, extra, out_dir: Path) -> pd.DataFrame:
    cmd = [sys.executable, str(_R1),
           "--posteriors", args.posteriors, "--samples", args.samples,
           "-o", str(out_dir), "--bin-size", str(args.bin_size)]
    if args.group_order:
        cmd += ["--group-order", args.group_order]
    if args.chrom:
        cmd += ["--chrom", args.chrom]
    cmd += extra + [t for t in args.ablation_args.split() if t]
    logger.info("  arm: %s", " ".join(extra))
    r = subprocess.run(cmd, capture_output=True, text=True)
    parquet = out_dir / "04_rank1.parquet"
    if r.returncode != 0:
        # An arm that filters everything away is a RESULT, not a crash: the
        # whole point of B.7.7 is to find where the floor stops leaving data.
        # Killing the run there would also destroy the arms that succeeded.
        if "no (group, bin, stratum) unit met" in (r.stdout + r.stderr):
            logger.warning("  arm %s kept no unit at all -- recorded as empty",
                           " ".join(extra))
            return pd.DataFrame(columns=["group", "chrom", "bin_start",
                                         "stratum", "D", "D_excess", "rho1",
                                         "rho2", "rho2_over_rho1", "N_eff",
                                         "flag_pd_clipped"])
        raise SystemExit(f"ablation arm {' '.join(extra)} failed:\n"
                         f"{r.stderr[-2000:]}")
    return pd.read_parquet(parquet)


def run_ablations(args, work: Path) -> tuple[dict, dict]:
    """B.7.6 and B.7.7 as filters over Part A's output.

    Neither needs a re-extraction, and that is not a shortcut -- it is what A.1
    emitting permissively and A.3.4 persisting the grid-cell sd were for.  A
    re-extraction would also change the reads that survive, confounding the
    ablation with a different contact set.
    """
    merged, summaries = {}, {}

    arms = {}
    for norm in ("location", "full", "none"):
        arms[norm] = _run_arm(args, ["--gch-norm", norm], work / f"norm_{norm}")
    m = dx.compare_ablation(arms, "B76_gch_norm")
    merged["B76_gch_norm"] = m
    summaries["B76_gch_norm"] = dx.ablation_summary(m, "D")

    arms = {}
    for k in (1, 2, 3):
        arms[f"hcg{k}"] = _run_arm(args, ["--min-hcg-sites", str(k)],
                                   work / f"hcg_{k}")
    m = dx.compare_ablation(arms, "B77_min_hcg_sites")
    merged["B77_min_hcg_sites"] = m
    summaries["B77_min_hcg_sites"] = dx.ablation_summary(m, "D")
    return merged, summaries


# --------------------------------------------------------------------- main ---

def load_units(args, manifest) -> tuple[pd.DataFrame, pd.Series, str]:
    root = Path(args.posteriors)
    shards = sorted(root.glob("chrom=*/*.parquet"))
    if not shards:
        raise SystemExit(f"no Part A shards under {root}")
    chrom = args.chrom or shards[0].parent.name.split("=", 1)[1]
    paths = [s for s in shards if s.parent.name == f"chrom={chrom}"]
    if not paths:
        raise SystemExit(f"--chrom {chrom} matched no shard")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    keep = set(manifest.cells)
    df = df[df["cell_id"].isin(keep)]
    units = pair_ends(df, args.bin_size)
    conv = df.set_index(["cell_id", "pair_idx"])["conversion_rate"]
    conv = conv[~conv.index.duplicated()]
    aligned = pd.MultiIndex.from_arrays(
        [units["cell_id"], units["pair_idx"]]).map(conv)
    return units, pd.Series(np.asarray(aligned, float)), chrom


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    manifest = load_from_args(args, require_bam=False, require_index=False)
    out_dir = ensure_dir(args.output)
    rows = pd.read_parquet(args.rank1)

    units, conv, chrom = load_units(args, manifest)
    logger.info("Contact-level panels on %s: %s orientations",
                chrom, f"{len(units):,}")

    cov = dx.coverage_flatness(units)
    verdict = dx.flatness_verdict(cov)
    conv_tab = dx.conversion_association(units, conv)
    decay = dx.distance_decay(rows, STRATUM_NAMES)
    shuffle = dx.mate_shuffle_sanity(rows)

    nulls = {
        "rank-1 truth": dx.null_calibration(args.null_units, n_boot=args.null_boot,
                                            seed=args.seed, rank2=0.0),
        "rank-2 injected": dx.null_calibration(args.null_units,
                                               n_boot=args.null_boot,
                                               seed=args.seed + 1, rank2=0.9),
    }

    figs = []
    figs += figure_coverage_and_conversion(cov, conv_tab, out_dir, args.plot_format)
    figs += figure_decay_and_shuffle(decay, shuffle, out_dir, args.plot_format)
    figs += figure_null_calibration(nulls, out_dir, args.plot_format)

    for name, t in (("B71_coverage_flatness", cov),
                    ("B72_distance_decay", decay),
                    ("B73_conversion_association", conv_tab),
                    ("B75_mate_shuffle", shuffle)):
        t.to_csv(out_dir / f"{name}.tsv", sep="\t", index=False)
    for name, t in nulls.items():
        t.to_csv(out_dir / f"B74_null_{name.split()[0]}.tsv", sep="\t",
                 index=False)

    summaries = {}
    if not args.skip_ablations:
        work = ensure_dir(out_dir / "_ablations")
        merged, summaries = run_ablations(args, work)
        for k, m in merged.items():
            m.to_parquet(out_dir / f"{k}_per_bin.parquet", index=False)
        for k, s in summaries.items():
            s.to_csv(out_dir / f"{k}_summary.tsv", sep="\t", index=False)
        figs += figure_ablations(summaries, merged, out_dir, args.plot_format)

    # ---- the verdicts, said out loud --------------------------------------
    logger.info("B.7.1 %s -- %s", verdict["verdict"].upper(), verdict["message"])
    ct = conv_tab.attrs.get("trend", {}) if len(conv_tab) else {}
    if np.isfinite(ct.get("rho1", np.nan)) and abs(ct["rho1"]) > dx.FLATNESS_TOL:
        logger.warning(
            "B.7.3 FAIL -- rho1 moves %+.3f across the conversion range; the "
            "A.3.5 threshold is too permissive.  This artifact is "
            "molecule-level, so no null in B.6 can see it.", ct["rho1"])
    else:
        logger.info("B.7.3 pass -- no residual conversion trend")
    ks = nulls["rank-1 truth"].attrs["ks_uniform_p"]
    frac = nulls["rank-1 truth"].attrs["frac_below_05"]
    power = nulls["rank-2 injected"].attrs["frac_below_05"]
    if ks < 0.01:
        logger.warning(
            "B.7.4 FAIL -- p-values under simulated rank-1 truth are not "
            "uniform (KS p=%.4g, %.1f%% below 0.05).  Every rank-2 call in "
            "04_rank1.parquet is invalid until this is fixed.", ks, 100 * frac)
    else:
        logger.info("B.7.4 pass -- KS p=%.2f, %.1f%% below 0.05 at nominal 5%%; "
                    "power against injected rank-2 %.0f%%", ks, 100 * frac,
                    100 * power)
    bad = shuffle[~shuffle["passes"]]
    if len(bad):
        logger.warning(
            "B.7.5 FAIL for %d group(s): %s.  rho1 survives reassigning mates, "
            "so it is not measuring coordination.",
            len(bad), ", ".join(map(str, bad["group"])))
    else:
        logger.info("B.7.5 pass -- rho1 collapses under the mate shuffle")

    (out_dir / "_b7_verdicts.json").write_text(json.dumps({
        "chrom": chrom,
        "B71_coverage_flatness": verdict,
        "B73_conversion_trend": {k: float(v) for k, v in ct.items()},
        "B74_null_calibration": {
            "ks_uniform_p": ks, "frac_below_05": frac,
            "power_vs_rank2": power},
        "B75_mate_shuffle_pass": bool(len(bad) == 0),
        "B76_B77_ablations": {k: s.to_dict("records")
                              for k, s in summaries.items()},
        "figures": figs,
    }, indent=2, default=str))
    logger.info("Done -> %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
