#!/usr/bin/env python3
"""J3 — per-channel embedding of the joint GCH+HCG multi-state concordance.

Design §6.  Builds the per-cell joint concordance tensor (cells × regions × 4
co-state channels) with the leakage-free distance-matched O/E core
(``joint_coordination``), then embeds it the §6 way:

  per-channel normalize → per-channel SVD → concat scores → (joint SVD/skip-top
  →) single UMAP

and reports the **per-channel component AUC** — WHICH co-state channel
(open / closed / poised / active_methylated) carries cell-type identity.

This is the interpretive companion to the existence test
(``joint_coordination_existence.py``): the existence test answers *whether* the
joint signal beats the single contexts; this answers *what structures the
cells* and draws the map.  SVD/UMAP of the Δ tensor is unsupervised
dimensionality reduction — NOT the RWR cross-cell smoothing — so it stays
leakage-free.

The embedding-score table and the channel-AUC diagnostic are always written.
The UMAP scatter (needs ``umap-learn``) and the optional tensor decomposition
(needs ``tensorly``) are graceful add-ons: skipped with a message if the
dependency is absent.

Example
-------
    python scripts/joint_multistate_embedding.py \\
        --gch-dir output/.../context_GCH/04_posteriors \\
        --hcg-dir output/.../context_HCG/04_posteriors \\
        --samples samples.tsv --granularity 1 \\
        --min-distance 20000 --max-distance 200000 \\
        --feature resid_full --min-region-frac 0.5 --tensor \\
        --n-threads 8 --output-prefix joint_embed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nome_hic_concordance import coordination_eval as ce
from nome_hic_concordance import joint_embedding as je
from nome_hic_concordance.joint_coordination import (
    JOINT_CHANNELS, build_matrices, process_cell, scan_priors,
)


def _labels(path: str) -> pd.Series:
    df = pd.read_csv(path, sep="\t", comment="#", dtype=str)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "cell_name" not in df.columns:
        df = df.rename(columns={"cell": "cell_name"} if "cell" in df.columns
                       else {df.columns[0]: "cell_name"})
    if "group" not in df.columns:
        raise SystemExit(f"{path} has no 'group' column.")
    return df.set_index("cell_name")["group"]


def _cells(d: Path) -> dict:
    return {p.name[: -len(".posteriors.parquet")]: p
            for p in d.glob("*.posteriors.parquet")}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gch-dir", required=True)
    ap.add_argument("--hcg-dir", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--min-distance", type=int, default=20_000)
    ap.add_argument("--max-distance", type=int, default=200_000)
    ap.add_argument("--min-gpc-sites", type=int, default=3)
    ap.add_argument("--min-cpg-sites", type=int, default=1)
    ap.add_argument("--granularity", default="1",
                    help="Single resolution for the embedding: 'chrom' or N (Mb "
                         "window). Default 1 (1 Mb).")
    ap.add_argument("--max-ends-per-chrom", type=int, default=4000)
    ap.add_argument("--min-region-frac", type=float, default=0.5,
                    help="Keep regions observed in >= this fraction of cells "
                         "(default 0.5 — well-observed regions for a stable "
                         "embedding).")
    ap.add_argument("--feature", choices=["delta", "resid_full"],
                    default="delta",
                    help="Embed the raw concordance Δ (default) or the marginal+"
                         "coverage-residualized Δ|m+cov (the marginal-"
                         "independent view — recommended when the marginal "
                         "methylation level alone separates the cells).")
    ap.add_argument("--no-rank", action="store_true",
                    help="Skip the per-cell-rank step of the per-channel "
                         "normalization (keep magnitudes).")
    ap.add_argument("--k-per-channel", type=int, default=10,
                    help="SVD components kept per channel before concatenation.")
    ap.add_argument("--n-svd", type=int, default=30,
                    help="Joint SVD dim on the concatenated scores before UMAP.")
    ap.add_argument("--skip-top-svd", type=int, default=0,
                    help="Drop the first N joint-SVD components before UMAP "
                         "(when the top components carry depth/batch).")
    ap.add_argument("--classifier", choices=["svd-lr", "lr", "rf"], default="lr")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--umap-neighbors", type=int, default=15)
    ap.add_argument("--umap-min-dist", type=float, default=0.1)
    ap.add_argument("--tensor", action="store_true",
                    help="Also run a Tucker tensor decomposition (needs "
                         "tensorly) → a channel factor showing how the 4 "
                         "co-states combine.")
    ap.add_argument("--tensor-rank", default="10,10,4",
                    help="Tucker ranks 'cell,region,channel' (default 10,10,4).")
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--output-prefix", required=True)
    args = ap.parse_args()

    gran = args.granularity if args.granularity == "chrom" else int(args.granularity)
    gch_cells = _cells(Path(args.gch_dir))
    hcg_cells = _cells(Path(args.hcg_dir))
    shared = sorted(set(gch_cells) & set(hcg_cells))
    if not shared:
        sys.exit("No cells shared between --gch-dir and --hcg-dir.")
    labels = _labels(args.samples)

    pri = shared[: min(len(shared), 40)]
    a_prior = scan_priors([gch_cells[c] for c in pri], args.min_gpc_sites,
                          args.n_threads)
    m_prior = scan_priors([hcg_cells[c] for c in pri], args.min_cpg_sites,
                          args.n_threads)
    print(f"[joint-embed] {len(shared)} shared cells | gran={gran} | band "
          f"{args.min_distance:,}–{args.max_distance:,} bp | feature="
          f"{args.feature} | priors a={a_prior:.3f} m={m_prior:.3f}")

    if args.n_threads > 1 and len(shared) > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.n_threads, verbose=5)(
            delayed(process_cell)(
                gch_cells[c], hcg_cells[c], [gran], args.min_distance,
                args.max_distance, args.min_gpc_sites, args.min_cpg_sites,
                a_prior, m_prior, args.max_ends_per_chrom, args.random_seed + i)
            for i, c in enumerate(shared))
    else:
        results = [process_cell(
            gch_cells[c], hcg_cells[c], [gran], args.min_distance,
            args.max_distance, args.min_gpc_sites, args.min_cpg_sites,
            a_prior, m_prior, args.max_ends_per_chrom, args.random_seed + i)
            for i, c in enumerate(shared)]
    results = [(c, d, qc) for c, d, qc in results if d]
    cells = [c for c, _, _ in results]
    y_all = labels.reindex(cells)

    mats = build_matrices(results, gran)
    region_keys = mats["region_keys"]
    if not region_keys:
        sys.exit("No regions at this granularity.")
    Dj, Mj, Cj = mats["joint"]          # cells × (R·4), region-major

    # Split into the 4 co-state channels (cells × R each), optionally
    # residualizing each against its own marginal + marginal² + coverage.
    channels = []
    for s in range(4):
        D = Dj[:, s::4]; M = Mj[:, s::4]; C = Cj[:, s::4]
        if args.feature == "resid_full":
            D = ce.residualize_multi(D, [M, M * M, np.log1p(np.nan_to_num(C))])
        # region-observation filter (well-observed regions only)
        if args.min_region_frac > 0:
            of = np.isfinite(Dj[:, s::4]).mean(axis=0)
            keepc = of >= args.min_region_frac
            if int(keepc.sum()) >= 2:
                D = D[:, keepc]
        channels.append(D)
    print(f"[joint-embed] channels: " +
          ", ".join(f"{n}={c.shape[1]}reg" for n, c in zip(JOINT_CHANNELS, channels)))

    scores, blocks, normed = je.build_embedding(
        channels, list(JOINT_CHANNELS), k_per_channel=args.k_per_channel,
        do_rank=not args.no_rank, seed=args.random_seed)

    # ---- Per-channel component AUC (the §6 "which co-state" diagnostic) ----
    labeled = y_all.notna().to_numpy()
    auc_rows = []
    if labeled.sum() >= 10 and y_all[labeled].nunique() >= 2:
        yl = y_all[labeled].to_numpy()
        ch_auc = je.channel_auc([b[labeled] for b in blocks],
                                list(JOINT_CHANNELS), yl,
                                method=args.classifier, n_folds=args.n_folds,
                                seed=args.random_seed)
        joint_auc = ce.score(scores[labeled], yl, args.classifier,
                             args.n_folds, args.random_seed)
        print(f"\n[joint-embed] per-channel component AUC ({args.classifier}) "
              f"— which co-state carries cell-type identity:")
        for n in JOINT_CHANNELS:
            m, sd = ch_auc[n]
            print(f"    {n:<18} AUC={m:.3f} ± {sd:.3f}")
            auc_rows.append({"channel": n, "auc": m, "auc_std": sd})
        print(f"    {'JOINT (all 4)':<18} AUC={joint_auc[0]:.3f} ± {joint_auc[1]:.3f}")
        auc_rows.append({"channel": "joint_all", "auc": joint_auc[0],
                         "auc_std": joint_auc[1]})
    else:
        print("[joint-embed] too few labeled cells / classes for channel AUC.")

    # ---- Write embedding scores + channel AUC (always) ----
    sdf = pd.DataFrame(scores, index=cells,
                       columns=[f"{ch}_svd{j}" for ch in JOINT_CHANNELS
                                for j in range(scores.shape[1] // 4)])
    sdf.insert(0, "group", y_all.values)
    sdf.to_csv(f"{args.output_prefix}.embedding_scores.csv")
    print(f"\n[joint-embed] wrote {args.output_prefix}.embedding_scores.csv")
    if auc_rows:
        pd.DataFrame(auc_rows).to_csv(f"{args.output_prefix}.channel_auc.csv",
                                      index=False)
        print(f"[joint-embed] wrote {args.output_prefix}.channel_auc.csv")

    # ---- UMAP (joint SVD/skip-top → 2-D), graceful if umap absent ----
    try:
        from nome_hic_concordance.visualization import run_umap_and_plot
        out_dir = str(Path(args.output_prefix).parent or ".")
        prefix = Path(args.output_prefix).name + ".umap"
        run_umap_and_plot(
            pd.DataFrame(scores, index=cells),
            group_labels=y_all,
            title=f"Joint multi-state concordance ({args.feature}, gran={gran})",
            output_prefix=prefix, output_dir=out_dir,
            n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist,
            n_svd=args.n_svd, skip_top_svd=args.skip_top_svd,
            n_threads=args.n_threads, random_state=args.random_seed)
        print(f"[joint-embed] wrote {out_dir}/{prefix}.* (UMAP)")
    except Exception as e:
        print(f"[joint-embed] UMAP skipped ({type(e).__name__}: {e})")

    # ---- Optional tensor decomposition (Tucker) → channel factor ----
    if args.tensor:
        try:
            _tensor_decompose(normed, list(JOINT_CHANNELS), args)
        except Exception as e:
            print(f"[joint-embed] tensor decomposition skipped "
                  f"({type(e).__name__}: {e})")


def _tensor_decompose(normed: list, names: list, args) -> None:
    """Tucker decomposition of the cells × regions × channel tensor.

    Yields a cell factor (→ could be embedded), a region factor, and a CHANNEL
    factor (how the 4 co-states combine into each latent component) — the §6
    optional view.  Requires that all channels share a region axis, so it uses
    the intersection of well-observed regions (here: the per-channel normalized
    matrices are already aligned to their own filtered region sets, so the
    tensor is built on the common min width)."""
    import tensorly as tl
    from tensorly.decomposition import tucker

    w = min(N.shape[1] for N in normed)
    T = np.stack([N[:, :w] for N in normed], axis=2)        # cells × w × 4
    ranks = [int(x) for x in args.tensor_rank.split(",")]
    ranks = [min(ranks[0], T.shape[0]), min(ranks[1], T.shape[1]),
             min(ranks[2], T.shape[2])]
    core, factors = tucker(tl.tensor(T), rank=ranks, random_state=args.random_seed)
    chan_factor = np.asarray(factors[2])                    # 4 × rank_channel
    print(f"\n[joint-embed] Tucker channel factor (4 co-states × "
          f"{chan_factor.shape[1]} components) — loadings:")
    for n, row in zip(names, chan_factor):
        print("    " + f"{n:<18} " + " ".join(f"{v:+.2f}" for v in row))
    cf = pd.DataFrame(chan_factor, index=names,
                      columns=[f"comp{j}" for j in range(chan_factor.shape[1])])
    cf.to_csv(f"{args.output_prefix}.tensor_channel_factor.csv")
    # cell factor → embedding table
    cell_factor = np.asarray(factors[0])
    pd.DataFrame(cell_factor,
                 columns=[f"cellfac{j}" for j in range(cell_factor.shape[1])]) \
        .to_csv(f"{args.output_prefix}.tensor_cell_factor.csv", index=False)
    print(f"[joint-embed] wrote {args.output_prefix}.tensor_channel_factor.csv "
          f"+ .tensor_cell_factor.csv")


if __name__ == "__main__":
    main()
