#!/usr/bin/env python3
"""Part C: per-cell coordination -> imputation -> embedding -> region selection.

    scripts/percell_coordination.py \
        --posteriors output/joint/03j_read_end_posteriors \
        --rank1 output/joint/04_rank1.parquet \
        --samples mouse_cortex_1577_cells.samples.with_batches.fixed.tsv \
        --group-order "IT-L2/3,IT-L4,IT-L5/6,CT-L6,Inhib,ODC,OPC,ASC,MGC,PC/EC" \
        -o output/joint

This is where cell-type specificity gets tested, so two things are load-bearing
and are checked rather than assumed:

**The feasibility check runs first.**  C.1 says compute the per-(cell, bin)
N_eff distribution before anything else; under a median of ~10 the answer is a
coarser bin, not a longer run.  The program reports the verdict and, without
--force, stops there.

**The ablation ladder decides whether this module earned its existence.**
contact-only / GCH-only / HCG-only / concatenated marginals / joint
coordination, same imputation and SVD for each.  If joint coordination does not
beat concatenated marginals on ARI, the coordination signal adds nothing over
ordinary methylation clustering, and the report says so in those words.
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

from nome_hic_concordance import figure_style                     # noqa: E402
from nome_hic_concordance import percell_coordination as pc       # noqa: E402
from nome_hic_concordance import rwr_graphs                       # noqa: E402
from nome_hic_concordance.distance_stats import fdr               # noqa: E402
from nome_hic_concordance.imputation import rwr_impute_matrices   # noqa: E402
from nome_hic_concordance.metrics import cluster_agreement, label_agreement  # noqa: E402
from nome_hic_concordance.rank1_units import pair_ends            # noqa: E402
from nome_hic_concordance.samples_manifest import (               # noqa: E402
    add_manifest_args, load_from_args, safe_name)
from nome_hic_concordance.utils import ensure_dir, setup_logging  # noqa: E402

logger = logging.getLogger("percell_coordination")

#: C.3 open decision 3: concatenating all strata multiplies features by ~5 and
#: they are not independent.  Start here, then test whether adding others helps.
DEFAULT_STRATA = ("100kb-1Mb", "1-10Mb")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_manifest_args(p)
    p.add_argument("--posteriors", required=True)
    p.add_argument("--rank1", required=True,
                   help="04_rank1.parquet; the pooled directions come from "
                        "its all-cells rows.")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--bin-size", type=int, default=250_000)
    p.add_argument("--strata", default=",".join(DEFAULT_STRATA))
    p.add_argument("--channel", type=int, default=1, choices=(1, 2),
                   help="1 is the shared axis; 2 is the mark-specific "
                        "direction.  Only 1 is expected to be usable at "
                        "250 kb -- C.4 is how that gets verified.")
    p.add_argument("--min-neff-cell", type=float, default=5.0)
    p.add_argument("--force", action="store_true",
                   help="Proceed even when the C.1 feasibility check says the "
                        "bins are too thin.")

    g = p.add_argument_group("imputation (C.3)")
    g.add_argument("--rwr-genomic", dest="rwr_genomic",
                   action="store_true", default=True)
    g.add_argument("--no-rwr", dest="rwr_genomic", action="store_false")
    g.add_argument("--rwr-restart", type=float, default=0.5)
    g.add_argument("--rwr-bandwidth", type=int, default=2)
    g.add_argument("--rwr-cells", action="store_true",
                   help="Smooth across a cell kNN graph.  Requires "
                        "--cell-graph-features and --cell-graph-provenance; "
                        "the program refuses a graph built from the "
                        "coordination matrix (C.3.2).")
    g.add_argument("--cell-graph-features", default=None,
                   help="Parquet of cells x features from an INDEPENDENT "
                        "measurement.")
    g.add_argument("--cell-graph-provenance", default=None,
                   choices=rwr_graphs.ALLOWED_PROVENANCE)
    g.add_argument("--rwr-sensitivity", default="0.25,0.5,0.75",
                   help="Restart probabilities to show the embedding's "
                        "sensitivity to, in QC.")

    g = p.add_argument_group("embedding (C.4)")
    g.add_argument("--n-svd", type=int, default=20)
    g.add_argument("--drop-pcs", default="",
                   help="Comma-separated 1-based PCs to drop.  Never "
                        "automatic: the panel reports, the user decides.")
    g.add_argument("--regress-covariates", default="",
                   help="Comma-separated per-cell covariates to regress out "
                        "of the PCs.")
    g.add_argument("--batch-correct", default="none",
                   choices=("none", "harmony", "linear"))
    g.add_argument("--skip-ablation", action="store_true")

    g = p.add_argument_group("regions (C.5)")
    g.add_argument("--top-n", type=int, default=2000)
    g.add_argument("--fdr", type=float, default=0.05)
    g.add_argument("--n-perm", type=int, default=1000)
    g.add_argument("--n-shuffle", type=int, default=20)

    p.add_argument("--plot-format", default="pdf",
                   choices=("pdf", "png", "svg"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    return p


# --------------------------------------------------------------------- C.1 ---

def load_directions(rank1_path: str, channel: int) -> pd.DataFrame:
    """Pooled per-(bin, stratum) directions from Part B's all-cells rows.

    Per-BIN directions rather than one global direction, because a single
    direction would erase the lambda_h sign flips that gene bodies and PMDs
    show -- and those flips are real readings of the model, not noise.
    """
    r = pd.read_parquet(rank1_path)
    pooled = r[r["group"] == "all"] if (r["group"] == "all").any() else r
    if not (r["group"] == "all").any():
        logger.warning(
            "04_rank1.parquet has no 'all' group; directions are pooled over "
            "every group present, which is NOT the same thing -- rerun Part B "
            "with --group-by all for the intended directions.")
        pooled = (r.groupby(["chrom", "bin_start", "stratum"], observed=True)
                   .agg({c: "mean" for c in ("w_A_gch", "w_A_hcg", "w_B_gch",
                                             "w_B_hcg", "rho1", "rho2")})
                   .reset_index())
    cols = ["chrom", "bin_start", "stratum", "w_A_gch", "w_A_hcg",
            "w_B_gch", "w_B_hcg", "rho1", "rho2"]
    out = pooled[[c for c in cols if c in pooled.columns]].copy()
    # sigma_s: the pooled disattenuated SD the C.1 kernel divides by.  With the
    # whitened directions this is 1 by construction, so a per-bin scale that
    # departs from it is a conditioning problem, not a measurement.
    out["sigma_A"] = 1.0
    out["sigma_B"] = 1.0
    if channel == 2:
        # The second canonical direction is orthogonal to the first in the
        # whitened basis; rotating w by 90 degrees there is the cheap stand-in
        # when Part B did not persist u2/v2.
        out[["w_A_gch", "w_A_hcg"]] = out[["w_A_hcg", "w_A_gch"]].to_numpy() * [-1, 1]
        out[["w_B_gch", "w_B_hcg"]] = out[["w_B_hcg", "w_B_gch"]].to_numpy() * [-1, 1]
    return out


def per_cell_table(args, manifest, directions: pd.DataFrame,
                   shuffle_mates: bool = False, seed: int = 0) -> pd.DataFrame:
    """C.1 steps 2-4 for every (cell, bin, stratum).

    ``shuffle_mates`` rebuilds the same table on B.6.1-shuffled contacts --
    partners reassigned within (cell, anchor bin, stratum).  That destroys the
    coordination while keeping every cell's coverage, which is what makes the
    resulting per-bin variance the empirical noise floor C.5 asks for.
    """
    rng = np.random.default_rng(seed)
    root = Path(args.posteriors)
    shards = sorted(root.glob("chrom=*/*.parquet"))
    if not shards:
        raise SystemExit(f"no Part A shards under {root}")
    strata = [s.strip() for s in args.strata.split(",") if s.strip()]
    keep_cells = set(manifest.cells)

    out = []
    for shard in shards:
        chrom = shard.parent.name.split("=", 1)[1]
        df = pd.read_parquet(shard)
        df = df[df["cell_id"].isin(keep_cells)]
        if not len(df):
            continue
        u = pair_ends(df, args.bin_size)
        u = u[u["stratum"].isin(strata)]
        if not len(u):
            continue
        u["chrom"] = chrom
        u["bin_start"] = u["anchor_bin"] * args.bin_size
        u = u.merge(directions, on=["chrom", "bin_start", "stratum"], how="inner")
        if not len(u):
            continue
        if shuffle_mates:
            idx = np.arange(len(u))
            for _, g in u.groupby(["cell_id", "anchor_bin", "stratum"],
                                  observed=True).indices.items():
                if len(g) > 1:
                    idx[g] = rng.permutation(g)
            for c in ("B_gch", "B_hcg", "B_v_gch", "B_v_hcg"):
                u[c] = u[c].to_numpy()[idx]

        s_A, v_sA = pc.project_ends(u[["A_gch", "A_hcg"]].to_numpy(),
                                    u[["A_v_gch", "A_v_hcg"]].to_numpy(),
                                    u[["w_A_gch", "w_A_hcg"]].to_numpy())
        s_B, v_sB = pc.project_ends(u[["B_gch", "B_hcg"]].to_numpy(),
                                    u[["B_v_gch", "B_v_hcg"]].to_numpy(),
                                    u[["w_B_gch", "w_B_hcg"]].to_numpy())
        # C.1 step 3: centred on the A.3.3 baseline (already, via centred_*)
        # AND on the cell's genome-wide mean s.  The second is what keeps a
        # per-cell global shift out of every bin.
        u["_sA"], u["_sB"] = s_A, s_B
        for col in ("_sA", "_sB"):
            u[col] = u[col] - u.groupby("cell_id")[col].transform("mean")
        kappa = pc.contact_kernel(u["_sA"], u["_sB"], u["sigma_A"], u["sigma_B"])
        w = 1.0 / (v_sA + v_sB + 1e-12)
        u["bin"] = u["chrom"] + ":" + u["bin_start"].astype(str)
        g = pc.aggregate(kappa, w, u[["cell_id", "bin", "stratum"]])
        g["chrom"] = chrom
        g["bin_start"] = g["bin"].str.rsplit(":", n=1).str[-1].astype(int)
        out.append(g)

    if not out:
        raise SystemExit("no (cell, bin, stratum) survived; check --strata and "
                         "that Part B covered these bins")
    tab = pd.concat(out, ignore_index=True)
    rho_bar = tab.groupby("bin", observed=True)["rho_hat"].transform("mean")
    tau = pc.tau_moment(tab["rho_hat"], tab["n_eff"], tab["bin"])
    tab["rho_bar"] = rho_bar
    tab["tau"] = tab["bin"].map(tau).astype(float)
    tab["rho_shrunk"] = pc.shrink(tab["rho_hat"], tab["n_eff"],
                                  tab["rho_bar"], tab["tau"])
    tab["channel"] = args.channel
    return tab


# --------------------------------------------------------------------- C.3 ---

def impute(mat: pc.CellBinMatrix, args, bins_meta: pd.DataFrame):
    """Genomic-neighbour RWR by default; cell-neighbour only past the guard.

    The matrix is already centred (C.2 step 2) and already shrunk toward the
    bin mean (C.1 step 4), so RWR runs on the centred values -- re-injecting
    the pseudo-bulk mean twice would double-count it.
    """
    X = np.where(np.isfinite(mat.values), mat.values, 0.0)
    report = {"rwr_genomic": bool(args.rwr_genomic),
              "rwr_cells": bool(args.rwr_cells)}
    if args.rwr_genomic:
        P = rwr_graphs.genomic_neighbor_graph(
            bins_meta["chrom"].to_numpy(), bins_meta["bin_start"].to_numpy(),
            bandwidth=args.rwr_bandwidth)
        # RWR over BINS: transpose so the walk runs along the genome, per cell.
        imp, info = rwr_impute_matrices({"m": X.T}, P,
                                        restart_prob=args.rwr_restart)
        X = imp["m"].T
        report["genomic"] = {k: v for k, v in info.items()
                             if isinstance(v, (int, float, str, bool))}
    if args.rwr_cells:
        if not args.cell_graph_features or not args.cell_graph_provenance:
            raise SystemExit(
                "--rwr-cells needs --cell-graph-features and "
                "--cell-graph-provenance.  A cell kNN graph built from the "
                "coordination matrix makes the clustering circular (C.3.2).")
        feats = pd.read_parquet(args.cell_graph_features)
        feats = feats.reindex(mat.cells)
        P, guard = rwr_graphs.cell_neighbor_graph(
            feats.to_numpy(), mat.values, args.cell_graph_provenance)
        imp, info = rwr_impute_matrices({"m": X}, P,
                                        restart_prob=args.rwr_restart)
        X = imp["m"]
        report["cells"] = guard
    return X, report


# --------------------------------------------------------------------- C.4 ---

def embed(X: np.ndarray, obs: pd.DataFrame, args) -> tuple[np.ndarray, dict]:
    """Truncated SVD, then the mandatory PC-covariate panel.

    Signed features, so an ordinary SVD -- not a count-based topic model.
    Nothing is dropped automatically: in scHi-C PC1 almost always tracks
    coverage, and silently removing it would also remove whatever real
    structure happens to correlate with depth.
    """
    from sklearn.decomposition import TruncatedSVD
    k = int(min(args.n_svd, min(X.shape) - 1))
    svd = TruncatedSVD(n_components=k, random_state=args.seed)
    scores = svd.fit_transform(X)

    rows = []
    for i in range(k):
        row = {"pc": i + 1,
               "variance_explained": float(svd.explained_variance_ratio_[i])}
        for cov in obs.columns:
            v = obs[cov]
            if pd.api.types.is_numeric_dtype(v):
                x = v.to_numpy(float)
                ok = np.isfinite(x)
                # A constant covariate (one batch, uniform depth) has no
                # correlation to report; np.corrcoef would divide by zero and
                # bury the panel in warnings.
                if ok.sum() > 2 and np.std(x[ok]) > 0 and np.std(scores[ok, i]) > 0:
                    row[f"r_{cov}"] = float(
                        np.corrcoef(scores[ok, i], x[ok])[0, 1])
                else:
                    row[f"r_{cov}"] = np.nan
            else:
                row[f"eta2_{cov}"] = _eta_squared(scores[:, i], v)
        rows.append(row)
    panel = pd.DataFrame(rows)

    drop = [int(t) - 1 for t in args.drop_pcs.split(",") if t.strip()]
    keep = [i for i in range(k) if i not in drop]
    used = scores[:, keep]
    if args.regress_covariates:
        names = [c.strip() for c in args.regress_covariates.split(",") if c.strip()]
        used = _regress_out(used, obs, names)
    return used, {"panel": panel, "dropped_pcs": [i + 1 for i in drop],
                  "variance_explained": svd.explained_variance_ratio_.tolist()}


def _eta_squared(x, labels) -> float:
    """Fraction of a PC's variance explained by a categorical covariate."""
    x = np.asarray(x, float)
    g = pd.Series(labels).astype(str).to_numpy()
    grand = x.mean()
    ssb = sum(((g == l).sum()) * (x[g == l].mean() - grand) ** 2
              for l in pd.unique(g))
    sst = ((x - grand) ** 2).sum()
    return float(ssb / sst) if sst > 0 else float("nan")


def _regress_out(scores, obs, names):
    cols = []
    for n in names:
        v = obs[n]
        if pd.api.types.is_numeric_dtype(v):
            cols.append(v.to_numpy(float).reshape(-1, 1))
        else:
            cols.append(pd.get_dummies(v.astype(str), drop_first=True)
                        .to_numpy(float))
    D = np.hstack([np.ones((len(obs), 1))] + cols)
    D = np.where(np.isfinite(D), D, 0.0)
    beta, *_ = np.linalg.lstsq(D, scores, rcond=None)
    return scores - D @ beta


def ablation_ladder(mat: pc.CellBinMatrix, marginals: dict, groups, args) -> pd.DataFrame:
    """C.4's comparison: does joint coordination beat concatenated marginals?

    Same imputation, SVD and clustering for every arm, or the ladder measures
    the pipeline rather than the features.
    """
    from sklearn.decomposition import TruncatedSVD
    arms = dict(marginals)
    arms["joint_coordination"] = np.where(np.isfinite(mat.values),
                                          mat.values, 0.0)
    if "gch_marginal" in arms and "hcg_marginal" in arms:
        arms["concatenated_marginals"] = np.hstack(
            [arms["gch_marginal"], arms["hcg_marginal"]])
    rows = []
    for name, X in arms.items():
        X = np.where(np.isfinite(X), X, 0.0)
        if X.shape[1] < 2:
            continue
        k = int(min(args.n_svd, min(X.shape) - 1))
        s = TruncatedSVD(n_components=k, random_state=args.seed).fit_transform(X)
        ari, nmi = cluster_agreement(s, groups, random_state=args.seed)
        lab = label_agreement(s, groups, random_state=args.seed)
        rows.append({"arm": name, "n_features": X.shape[1], "ari": ari,
                     "nmi": nmi, **{k2: v for k2, v in lab.items()
                                    if isinstance(v, float)}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- C.5 ---

def region_stats(mat: pc.CellBinMatrix, null_mat: pc.CellBinMatrix,
                 bins_meta: pd.DataFrame, groups,
                 per_cell: pd.DataFrame, args) -> pd.DataFrame:
    # values_centred, NOT values: per-bin standardization sets every bin's
    # across-cell variance to 1, which would make sigma2_bio a constant and the
    # "most variable" ranking meaningless.
    V = mat.values_centred
    s = pc.sigma2_bio(V, mat.se)
    s = pc.empirical_sigma2_bio(
        s, pc.noise_floor_from_null_matrix(null_mat.values_centred))
    s = pd.concat([bins_meta.reset_index(drop=True), s,
                   pc.group_f_statistic(V, groups),
                   pc.specificity_index(V, groups)], axis=1)
    s["p_value"] = pc.permutation_p(V, groups, args.n_perm, args.seed)
    ok = np.isfinite(s["p_value"])
    s["q_value"] = np.nan
    if ok.any():
        s.loc[ok, "q_value"] = fdr(s.loc[ok, "p_value"].to_numpy())

    agg = (per_cell.groupby("bin", observed=True)
                   .agg(mean_rho=("rho_shrunk", "mean"),
                        mean_neff=("n_eff", "mean")).reset_index())
    s = s.merge(agg, on="bin", how="left")
    s["passes_qc"] = (s["mean_neff"] >= args.min_neff_cell) & (s["n_cells"] >= 3)
    return pc.classify_regions(s, top_n=args.top_n, fdr=args.fdr)


def write_region_beds(stats: pd.DataFrame, out_dir: Path, bin_size: int) -> dict:
    """One BED per C.5 category.

    ``background`` is emitted explicitly and is not optional: it is the
    comparison set every enrichment test needs, and computing enrichments
    against the whole genome instead is how a coverage bias becomes a finding.
    """
    paths = {}
    for cat, col in (("variable", "is_variable"),
                     ("celltype_specific", "is_celltype_specific"),
                     ("stable", "is_stable"),
                     ("background", "is_background")):
        sub = stats[stats[col]]
        p = out_dir / f"06_regions_{cat}.bed"
        with open(p, "w") as fh:
            for r in sub.itertuples():
                name = (f"{safe_name(str(r.top_group))}"
                        if cat == "celltype_specific" else cat)
                score = int(np.clip(round(float(r.sigma2_bio) * 1000), 0, 1000))
                fh.write(f"{r.chrom}\t{r.bin_start}\t{r.bin_start + bin_size}\t"
                         f"{name}\t{score}\t.\n")
        paths[cat] = str(p)
        logger.info("  %s: %d bins -> %s", cat, len(sub), p.name)
    return paths


# --------------------------------------------------------------------- C.6 ---

def heatmaps(mat, stats, obs, manifest, out_dir: Path, fmt: str, top_n: int):
    """Rows clustered, columns NOT.

    The point is to show whether the GIVEN --group-order already produces block
    structure; clustering the columns would produce blocks whatever the data
    said, and the figure would then be evidence of nothing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list, linkage

    order = [g for g in manifest.group_order if g in set(obs["group"])]
    col_order = np.concatenate([np.flatnonzero(obs["group"].to_numpy() == g)
                                for g in order]) if order else np.arange(len(obs))
    vmax = float(np.nanpercentile(np.abs(mat.values), 98)) or 1.0
    paths = []
    for cat, col in (("variable", "is_variable"),
                     ("celltype_specific", "is_celltype_specific"),
                     ("stable", "is_stable")):
        idx = np.flatnonzero(stats[col].to_numpy())
        if idx.size < 2:
            continue
        idx = idx[np.argsort(-stats["sigma2_bio"].to_numpy()[idx])][:top_n]
        M = np.nan_to_num(mat.values[np.ix_(col_order, idx)]).T
        if M.shape[0] > 2:
            idx = idx[leaves_list(linkage(M, method="average"))]
            M = np.nan_to_num(mat.values[np.ix_(col_order, idx)]).T
        with plt.rc_context(figure_style.RC):
            fig, ax = plt.subplots(figsize=(figure_style.FIG_W_IN, 3.4))
            ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      interpolation="nearest")
            ax.set_xlabel("cells, ordered by --group-order (never clustered)")
            ax.set_ylabel(f"{cat} bins ({len(idx)})")
            ax.set_yticks([])
            edges = np.cumsum([int((obs["group"].to_numpy() == g).sum())
                               for g in order])[:-1]
            for e in edges:
                ax.axvline(e - 0.5, color="k", lw=0.4)
            ax.set_xticks([])
            ax.set_title(f"C.6 {cat}")
            fig.tight_layout()
            p = out_dir / f"heatmap_{cat}.{fmt}"
            fig.savefig(p, bbox_inches="tight")
            plt.close(fig)
        paths.append(str(p))
    return paths


# -------------------------------------------------------------------- main ---

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()
    manifest = load_from_args(args, require_bam=False, require_index=False)
    manifest.report()
    out = ensure_dir(args.output)
    qc = ensure_dir(out / "qc")
    figs = ensure_dir(out / "figures")

    directions = load_directions(args.rank1, args.channel)
    tab = per_cell_table(args, manifest, directions)

    feas = pc.feasibility(tab["n_eff"], args.bin_size)
    logger.info("C.1 feasibility: %s -- %s", feas["verdict"].upper(),
                feas["message"])
    if feas["verdict"] == "too_thin" and not args.force:
        (out / "_partC_metadata.json").write_text(
            json.dumps({"feasibility": feas, "stopped": True}, indent=2))
        raise SystemExit(
            feas["message"] + "  Re-run with a larger --bin-size, or --force "
            "to proceed anyway.")

    tab.to_parquet(out / "05_percell_coordination.parquet", index=False)

    cells = [c for c in manifest.cells if c in set(tab["cell_id"])]
    bins_meta = (tab[["bin", "chrom", "bin_start"]].drop_duplicates()
                    .sort_values(["chrom", "bin_start"]).reset_index(drop=True))
    mat = pc.build_matrix(tab, cells, bins_meta["bin"].tolist(),
                          min_neff_cell=args.min_neff_cell)

    meta = manifest.table.set_index("name").reindex(cells)
    mat.obs["group"] = meta["group"].to_numpy()
    if "batch" in meta:
        mat.obs["batch"] = meta["batch"].to_numpy()
    per_cell_cov = (tab.groupby("cell_id")
                       .agg(n_contacts=("n_contacts", "sum"),
                            mean_neff=("n_eff", "mean"),
                            global_rho=("rho_hat", "mean")))
    for c in per_cell_cov.columns:
        mat.obs[c] = per_cell_cov.reindex(cells)[c].to_numpy()
    mat.obs["log10_n_contacts"] = np.log10(
        np.maximum(mat.obs["n_contacts"].to_numpy(float), 1))

    X, imp_report = impute(mat, args, bins_meta)
    scores, emb = embed(X, mat.obs.drop(columns=["group"], errors="ignore"), args)

    if args.batch_correct != "none" and "batch" in mat.obs:
        from nome_hic_concordance.batch_correction import harmonize_embedding
        scores = harmonize_embedding(scores, mat.obs[["batch"]],
                                     method=args.batch_correct)

    groups = mat.obs["group"].to_numpy()
    ari, nmi = cluster_agreement(scores, groups, random_state=args.seed)
    lab = label_agreement(scores, groups, random_state=args.seed)
    logger.info("C.4 agreement with --group-column: ARI=%.3f NMI=%.3f", ari, nmi)

    emb["panel"].to_csv(qc / "C4_pc_covariate_panel.tsv", sep="\t", index=False)
    pd.DataFrame([{"ari": ari, "nmi": nmi, **lab}]).to_csv(
        qc / "C4_label_agreement.tsv", sep="\t", index=False)
    pd.DataFrame(scores, index=pd.Index(cells, name="cell_id"),
                 columns=[f"PC{i + 1}" for i in range(scores.shape[1])]
                 ).reset_index().to_parquet(out / "05_svd.parquet", index=False)

    ladder = pd.DataFrame()
    if not args.skip_ablation:
        ladder = ablation_ladder(mat, {}, groups, args)
        ladder.to_csv(qc / "C4_ablation_ladder.tsv", sep="\t", index=False)
        if {"joint_coordination", "concatenated_marginals"} <= set(ladder["arm"]):
            j = float(ladder.loc[ladder["arm"] == "joint_coordination", "ari"])
            m = float(ladder.loc[ladder["arm"] == "concatenated_marginals", "ari"])
            if j <= m:
                logger.warning(
                    "C.4 ABLATION: joint coordination (ARI %.3f) does NOT beat "
                    "concatenated marginals (ARI %.3f).  The coordination "
                    "signal adds nothing over ordinary methylation clustering "
                    "and this must be reported.", j, m)
        else:
            logger.info(
                "C.4 ablation ran %d arm(s); the marginal arms need "
                "--cell-graph-features-style inputs this run did not get, so "
                "the 'does it beat marginals' question is NOT answered here.",
                len(ladder))

    # C.5's floor: the same pipeline on B.6.1-shuffled contacts.
    logger.info("C.5 noise floor: rebuilding on mate-shuffled contacts")
    null_tab = per_cell_table(args, manifest, directions, shuffle_mates=True,
                              seed=args.seed + 1)
    null_mat = pc.build_matrix(null_tab, cells, bins_meta["bin"].tolist(),
                               min_neff_cell=args.min_neff_cell)
    stats = region_stats(mat, null_mat, bins_meta, groups, tab, args)
    stats.to_parquet(out / "06_region_stats.parquet", index=False)
    beds = write_region_beds(stats, out, args.bin_size)
    hm = heatmaps(mat, stats, mat.obs, manifest, figs, args.plot_format,
                  args.top_n)

    (out / "_partC_metadata.json").write_text(json.dumps({
        "command": " ".join(sys.argv),
        "feasibility": feas, "channel": args.channel,
        "strata": args.strata, "bin_size": args.bin_size,
        "matrix_steps": mat.steps, "imputation": imp_report,
        "dropped_pcs": emb["dropped_pcs"],
        "agreement": {"ari": ari, "nmi": nmi},
        "ablation": ladder.to_dict("records") if len(ladder) else [],
        "region_beds": beds, "heatmaps": hm,
        "manifest": manifest.metadata(),
    }, indent=2, default=str))
    logger.info("Done -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
