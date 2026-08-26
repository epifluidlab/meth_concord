"""Command-line entry point for the NOMe-HiC concordance pipeline.

Invoked via the ``nome_hic_concordance`` console script (see pyproject.toml)
or with ``python -m nome_hic_concordance``.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import (
    bam_parser,
    concordance as concordance_mod,
    distance_analysis,
    matrix_ops,
    mixture_model,
    posterior,
    reference,
    shrinkage,
    visualization,
)
from .utils import (add_exclude_chrom_arg, add_include_chrom_arg,
                    chrom_excluded, chrom_key, ensure_dir,
                    parse_exclude_chroms, parse_include_chroms,
                    report_excluded, setup_logging)

logger = logging.getLogger("nome_hic_concordance")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="nome_hic_concordance",
        description="NOMe-HiC Long-Range Chromatin Accessibility Concordance Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--samples", required=True,
                        help="Path to sample manifest TSV (cell_name, bam_path, [group]).")
    parser.add_argument("--reference", default=None,
                        help="Path to .2bit reference genome. Required when "
                             "extracting contexts from the reference (no "
                             "--context-bed). Optional when --context-bed is "
                             "provided (chromosome sizes are read from the "
                             "BAM headers instead).")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory (will be created).")

    # Filtering
    parser.add_argument("--min-distance", type=int, default=1_000,
                        help="Min genomic distance for qualifying pairs.")
    parser.add_argument("--max-distance", type=int, default=0,
                        help="Max genomic distance for the long-range (linked) "
                             "pool (signed method).  Default 0 = no upper bound. "
                             "Set e.g. 200000 to focus the 20–200 kb "
                             "enhancer–promoter regulatory band; pairs beyond it "
                             "are excluded from the linked pool (the short-range "
                             "baseline pool, < min-distance, is unaffected).")
    parser.add_argument("--min-gch-sites", type=int, default=2,
                        help="Min informative GCH sites per read end "
                             "(only applied when processing context GCH; "
                             "design-doc §6 #14 — empirically the sweet spot "
                             "for NOMe-HiC: 1.48× more pairs than a value of 3 "
                             "with no measurable loss of per-pair sharpness).")
    parser.add_argument("--min-hcg-sites", type=int, default=2,
                        help="Min informative HCG sites per read end "
                             "(only applied when processing context HCG). "
                             "HCG (CpG) sites are typically sparser than GCH, "
                             "so 2 is a good floor.")
    add_exclude_chrom_arg(parser)
    parser.add_argument("--bin-size", type=int, default=250_000,
                        help="Genomic bin size in bp.")
    parser.add_argument("--min-pairs-per-bin", type=int, default=None,
                        help="Min pairs per bin to compute Δ. "
                             "Default depends on --concordance-method: "
                             "1 for 'signed' or 'betabinom' (DM/BB EB "
                             "shrinkage works with n=1), 5 for 'pearson' "
                             "(correlation needs ≥5 pairs).")
    parser.add_argument("--min-cell-coverage", type=float, default=0.8,
                        help="Fraction of cells non-NA to retain a bin.")
    parser.add_argument(
        "--min-cell-pairs", type=int, default=0,
        help="Signed method only: pre-imputation cell QC.  Drop cells whose "
             "total informative pairs (linked + unlinked, summed over bins) "
             "is below this, BEFORE building the RWR graph.  Sparse cells get "
             "imputation-dominated Δ and can form spurious UMAP islands / "
             "inject noise into the lower SVD components.  Default 0 (off); "
             "use scripts/umap_residual_mixing_diagnostic.py to pick a "
             "threshold from the coverage of mis-separated cells.")
    parser.add_argument(
        "--min-bin-coverage", type=float, default=0.0,
        help="Signed method only: pre-imputation bin QC.  Drop bins where the "
             "fraction of cells with any raw pair is below this, BEFORE RWR. "
             "Default 0.0 (off).  (Distinct from --min-cell-coverage, which "
             "filters the post-imputation Δ matrix in Step 7.)")
    parser.add_argument(
        "--blacklist-bed", type=str, default=None,
        help="Optional path to a BED file of blacklist regions (e.g. ENCODE). "
             "Applied at Step 2 (BAM parsing): any read pair whose either "
             "end (pos1 or pos2) falls inside a blacklist interval on its "
             "chromosome is dropped before the per-cell parquet is written, "
             "so the mixture model / posteriors / concordance never see "
             "blacklisted reads.  Accepts plain or gzipped BED; only the "
             "first three columns are read.  Note: to take effect, Step 2 "
             "must run with this flag — adding it later (e.g. for --steps 7 "
             "only) is a no-op.",
    )
    parser.add_argument(
        "--concordance-method", choices=["pearson", "betabinom", "signed"],
        default="signed",
        help="Per-bin concordance feature. 'signed' (new): "
             "Dirichlet-Multinomial signed Δ = (E[π_l_open]−E[π_l_closed]) − "
             "(E[π_u_open]−E[π_u_closed]) per (cell, bin) using U-3 soft "
             "counts and a short-range pair baseline. Captures direction "
             "(active vs silenced). 'betabinom': one-sided Δ_open = "
             "E[θ_L|data] − p̂² (the v0.2 behaviour; aliased to "
             "--concordance-statistic open). 'pearson': legacy Pearson r.",
    )
    parser.add_argument(
        "--concordance-statistic", choices=["signed", "open", "both"],
        default=None,
        help="Alias / refinement of --concordance-method for the new "
             "DM-based pipeline. 'signed' = directional Δ (when "
             "--concordance-method=signed). 'open' = legacy one-sided Δ. "
             "'both' = compute and save both matrices in one run.",
    )
    parser.add_argument(
        "--matrix-norm", choices=["auto", "zscore", "quantile"], default="quantile",
        help="Per-bin normalization across cells in Step 7.  'zscore' "
             "for the classical pipeline; 'quantile' for rank-based QN, matching "
             "the bulk-style preprocessing that's robust to outliers in sparse "
             "single-cell data.  'auto' resolves to 'quantile' for "
             "--concordance-method signed, 'zscore' otherwise.",
    )
    parser.add_argument(
        "--concordance-formula", choices=["auto", "dm", "mle"], default="auto",
        help="Per-(cell, bin) rate estimator for --concordance-method signed. "
             "'dm' = Dirichlet-Multinomial posterior mean (k+α)/(n+α_total): "
             "calibrated, but on post-RWR counts it double-shrinks (RWR already "
             "pooled the cell graph) and the MoM concentration is inflated by "
             "RWR's variance reduction, crushing cell-type signal onto a thin "
             "PC.  'mle' = plug-in k/n: preserves RWR's cell-type-aware "
             "smoothing.  NOTE: a full-set comparison (kNN cell-type accuracy "
             "on all cells) found DM beats MLE (DM+skip1 0.937 vs MLE best "
             "0.930; MLE collapses without harmony); the earlier MLE win was "
             "an artifact of validating on the high-coverage labeled subset.  "
             "'auto' (default): 'dm' for signed, 'dm' otherwise.",
    )
    parser.add_argument(
        "--baseline",
        choices=["short-range", "unlinked", "distance-matched",
                 "distance-pooled"],
        default="distance-pooled",
        help="Baseline E that the long-range observed concordance O is "
             "compared against (signed method).  'short-range' = the "
             "short-distance (<min-distance) same-bin pool (legacy; these are "
             "real short-range *contacts*, not non-contacts — so it does NOT "
             "control distance.  'unlinked' is a deprecated alias).  "
             "'distance-matched' = the expected co-state of random "
             "NON-contact end-pairs at the SAME distance within the bin (an "
             "O/E null from the bin's ~hundreds of read ends), controlling "
             "distance + bin-marginal.  'distance-pooled' = the same "
             "non-contact end-pair null but pooling ALL pairs with sep "
             ">= min-distance (excluding the observed contacts), no distance "
             "matching — many more pairs → a less-noisy expected, at the "
             "cost of distance control.",
    )
    parser.add_argument(
        "--concordance-combine",
        choices=["auto", "signed-diff", "log2-ratio", "linked-only",
                 "concordant-diff", "concordant-signed", "direction-logor"],
        default="concordant-signed",
        help="How the long-range observed (O) and baseline (E) co-state rates "
             "are combined.  'signed-diff' = "
             "(O_open−O_closed)−(E_open−E_closed) (legacy; NOTE with an O/E null "
             "the open−closed contrast cancels the coordination — it measures "
             "only the marginal).  'log2-ratio' = "
             "log2(O_open/E_open)−log2(O_closed/E_closed) (symmetric, "
             "scale-free, Hi-C O/E convention; retains coordination but with a "
             "marginal-dependent scaling that vanishes near the 0.5 level).  "
             "'direction-logor' = log(O_open/O_closed) − log(E_open/E_closed), "
             "the log odds ratio of the same 2×2 table: additive, with a "
             "variance free of F, so bins of different depth and different "
             "marginal are comparable — which a difference of proportions "
             "is not.  It is the SAME statistic as 'log2-ratio' up to the "
             "base (log2-ratio = L / ln 2), with one difference that matters "
             "on sparse data: where a bin has no observed concordant mass, "
             "log2-ratio still returns log2(E_closed/E_open) out of its "
             "clip — a confident direction read off the null in a bin with "
             "no data — while this returns 0.  Prefer it over 'log2-ratio'.  "
             "It is DIRECTION only: scaling the concordant mass up at a "
             "fixed co-open:co-closed ratio leaves it unchanged and raises "
             "C, so a claim that coordination INCREASED needs "
             "'concordant-diff'.  "
             "'concordant-diff' = (O_open+O_closed)−(E_open+E_closed), the "
             "excess concordant mass / mixed deficit = 2·Cov: the marginal-free "
             "coordination magnitude (theoretically the cleanest against an O/E "
             "null; sign = concordant vs anti-concordant), but DIRECTION-BLIND "
             "(Cov is unchanged by open<->closed).  'concordant-signed' = "
             "D = C·(2F−1) where C is the 'concordant-diff' magnitude and "
             "F = O_open/(O_open+O_closed) is the realized direction: the "
             "single co-CLOSED (−) -> mixed (0) -> co-OPEN (+) scale.  The "
             "direction cannot come from the excess itself — "
             "(O_open−E_open) == (O_closed−E_closed) identically — so it is "
             "taken from where the concordant mass sits.  NOTE D is also ~0 "
             "for a coordinated-but-BALANCED bin (F~0.5); read it next to the "
             "emitted coordination matrix C to tell that apart from mixed "
             "(mixed -> C~0; balanced -> C large, D~0).  'linked-only' = "
             "(O_open−O_closed) with NO baseline subtracted (--baseline is "
             "ignored) — a diagnostic to check whether the concordance UMAP is "
             "driven by the short-range baseline; note this is the raw "
             "marginal-laden score and typically correlates MORE with 1D "
             "accessibility, not less.  'auto' resolves to "
             "'log2-ratio' for the distance-matched / distance-pooled O/E "
             "baselines, 'signed-diff' otherwise (the legacy behaviour); it "
             "is now an explicit opt-in rather than the default.  "
             "DEFAULT: 'concordant-signed' — it is the only mode that is both "
             "coordination-bearing and directional, and it also turns on "
             "--component-umaps auto, so C / F / D / p̄ are all embedded.  "
             "(The library default in compute_signed_concordance_matrix "
             "remains 'signed-diff' for direct API callers.)",
    )
    parser.add_argument(
        "--step5-batch-sweep", default=None, metavar="METHODS",
        help="Comma-separated batch-correction methods to sweep at STEP 5 "
             "(the mean-accessibility matrix that seeds the RWR graph), "
             "crossed with every non-empty subset of --batch-column.  Unlike "
             "the Step-7 axes this one cannot be swept from a finished Δ "
             "matrix: correcting the RWR basis changes the k-NN graph, hence "
             "what RWR imputes, hence the concordance values themselves.  "
             "Steps 1-3 are computed once and replayed, so each setting costs "
             "only RWR + Layer 2.  Writes step5_batch_sweep.tsv ranked by kNN "
             "purity.  Requires --batch-column and a run that includes step 5; "
             "applying a winner means re-running steps 5-7, not step 7 alone.",
    )
    parser.add_argument(
        "--concordance-batch-column", default=None,
        help="Batch variables for --concordance-batch-correction, when they "
             "should differ from --batch-column.  The two corrections act on "
             "different matrices at different stages, so a sweep can find "
             "that (say) the plate effect shows up per bin while the "
             "experiment effect shows up in the embedding.  Defaults to "
             "--batch-column, so the common case stays one flag.",
    )
    parser.add_argument(
        "--concordance-batch-correction",
        choices=["none", "center", "zscore", "combat", "harmony"],
        default="none",
        help="Batch-correct the CONCORDANCE matrix itself (and the C / F / D / "
             "p̄ components) per bin, in Step 7 before normalization.  This is "
             "a different stage from --batch-correction, which corrects the "
             "mean-accessibility matrix seeding the RWR graph in Step 5 and "
             "the SVD scores just before UMAP in Step 7 — neither of which "
             "touches the concordance values.  Batch enters those values in "
             "between (per-cell coverage drives the DM shrinkage and what RWR "
             "imputes), and correcting ten SVD components afterwards cannot "
             "separate what is already mixed into the bins.  Must precede the "
             "per-bin quantile normalization, since QN turns a batch offset "
             "into a batch-ordered ranking and preserves it.  'center' "
             "(subtract each batch's per-bin mean) is the right default if "
             "you want one; it cannot fail the way harmony can when batch and "
             "cell type are partly confounded.  Needs --batch-column.  The "
             "before/after per-bin ANOVA F is logged.",
    )
    parser.add_argument(
        "--component-umaps", choices=["auto", "on", "off"], default="auto",
        help="In Step 7, additionally embed the four concordance COMPONENT "
             "matrices written by Step 5: directional (D = C·(2F−1)), "
             "coordination (C, the marginal-free magnitude), direction "
             "(F = O_open/(O_open+O_closed), the co-open share) and marginal "
             "(mean per-end openness p̄).  C and p̄ are the two INDEPENDENT "
             "degrees of freedom of the co-state distribution; F and D are "
             "derived from them but answer a different question — F is which "
             "way a bin's concordance points regardless of its strength, C is "
             "the strength regardless of direction.  Embedding each shows "
             "which one carries the cell-type structure.  (A co-open UMAP and "
             "a co-closed UMAP would NOT be independent — the co-open and "
             "co-closed excesses are identical — which is why these are the "
             "axes emitted.)  Each gets the same normalization / SVD / "
             "batch-correction treatment as the main UMAP; 'directional' "
             "duplicates the main UMAP when --concordance-combine is "
             "'concordant-signed'.  'auto' = on when --concordance-combine "
             "resolves to 'concordant-signed', off otherwise.",
    )
    parser.add_argument(
        "--dm-window-mb", type=float, default=0.0,
        help="For --baseline distance-matched / distance-pooled: pool the "
             "null end-pool over ±this many Mb of neighbouring bins "
             "(0 = the bin's own ends only).  Set e.g. 1.0 for a "
             "smoother expected at long distances where a single bin's "
             "end-pairs thin out.",
    )
    parser.add_argument(
        "--emit-null", action="store_true",
        help="Also emit concordance_<component>_null.parquet: the same "
             "statistics computed with a DRAWN null in place of the "
             "observed contacts.  A Q-Q plot of C or the direction needs a "
             "sample from the null, and no parametric reference is valid "
             "here -- the counts are DM posterior soft counts, RWR has "
             "already pooled evidence across the cell graph, and the "
             "effective n is fractional, so an N(0,1) reference would "
             "assert a sampling theory that does not hold.  This draws it "
             "instead: exactly n_linked random NON-contact end-pairs per "
             "(cell, bin), summed like the real contacts, carried through "
             "the same prior, imputation and normalisation.  DIAGNOSTIC "
             "RUNS ONLY -- the end pool is split in half so the expected "
             "and the drawn null share no read end (sharing them makes the "
             "two agree more than chance and the null comes back too "
             "narrow), which means the expected here is built on half the "
             "ends and is noisier than a production run's.  Compare runs "
             "with scripts/null_calibration_qq.py.",
    )
    parser.add_argument(
        "--dm-prior", choices=["separate", "shared"], default="separate",
        help="Whether the observed (linked) and expected (non-contact) "
             "pools get their own Dirichlet-Multinomial EB prior or one "
             "between them.  Both sides are posterior means "
             "(alpha_i + k_i)/(alpha_0 + n) and n_dm == n_linked, so they "
             "share a denominator and C carries the term "
             "(alpha_linked_concordant - alpha_expected_concordant)/"
             "(alpha_0 + n), which does not depend on the data.  'separate' "
             "(default, and what every existing run used) shrinks each pool "
             "toward its OWN global mean, and those means differ by the "
             "genome-wide average coordination -- so a bin with no "
             "information is handed the global average rather than zero and "
             "C cannot reach zero however good the null is.  Post-RWR the "
             "imputed n is usually < 1, so that term is most of C: on a "
             "mouse cortex run the gap was 0.0113 against a median C of "
             "0.011293.  'shared' fits one prior on both pools; the term is "
             "then exactly 0 at every n and C = 0 means the pools agree.  "
             "The run reports the gap either way.",
    )
    parser.add_argument(
        "--dm-dist-bin-bp", type=int, default=10_000,
        help="Distance-bin width (bp) for the distance-matched expected "
             "(10 kb).",
    )
    parser.add_argument(
        "--refinement-basis", choices=["auto", "raw", "postnorm"], default="auto",
        help="For RWR refinement passes (--rwr-refine-passes >= 1): which form "
             "of the previous pass's Δ_signed to build the next k-NN graph "
             "from.  'raw' = per-column-rank(Δ) only (legacy).  'postnorm' = "
             "apply the Step-7 normalization (per-cell rank → per-bin QN → row "
             "recenter) first, stripping per-cell shifts from the basis so "
             "depth/accessibility community structure isn't re-injected into "
             "the refined graph.  'postnorm' helps under --concordance-formula "
             "mle; it's a no-op gain (or slight loss) under dm.  'auto' "
             "(default): 'postnorm' for signed, 'raw' otherwise.",
    )
    parser.add_argument(
        "--per-cell-center", choices=["auto", "on", "off"], default="auto",
        help="DEPRECATED — use --per-cell-norm instead.  Kept for "
             "backward compatibility.  'auto' on / off / 'on' / 'off' "
             "translate to --per-cell-norm {center, off, center, off} "
             "respectively (with 'auto' resolved per --concordance-method).",
    )
    parser.add_argument(
        "--per-cell-norm",
        choices=["auto", "off", "center", "zscore", "rank"],
        default="auto",
        help="Per-cell normalization applied before per-bin "
             "normalization in Step 7.  Removes per-cell global shift / "
             "scale effects (depth, library complexity, RWR connectivity) "
             "that otherwise dominate the top SVD component.  "
             "'center': subtract per-cell mean.  "
             "'zscore': subtract mean AND divide by std (more aggressive, "
             "removes scale as well).  "
             "'rank': replace each cell's row with within-cell ranks in "
             "(-0.5, 0.5) — closest analogue to TF normalisation in "
             "LSI / scATAC-seq, most robust to outliers and structured "
             "noise.  "
             "'auto' (default): 'rank' for --concordance-method signed "
             "(empirically the most effective on NOMe-HiC data); 'off' "
             "for pearson and betabinom (legacy).",
    )
    parser.add_argument(
        "--plot-format", type=str, default="png",
        choices=["png", "pdf", "svg"],
        help="Figure format for the step-7 UMAP and SVD-scree plots. "
             "'pdf' (or 'svg') writes a vector figure for publication; "
             "default 'png'.",
    )
    parser.add_argument(
        "--color-column", type=str, default=None,
        help="Column in --samples holding the plot colour for each cell (a hex "
             "code like '#1f77b4' or a matplotlib colour name).  Every cell of "
             "a group should carry the same colour; the group's first colour "
             "wins (a warning names any group with more than one).  When set, "
             "the UMAP legend uses these colours instead of tab20, which also "
             "lifts the 20-group cap.",
    )
    parser.add_argument(
        "--group-order", type=str, default=None,
        help="COMMA-SEPARATED group order for the UMAP legend, e.g. "
             "'IT-L2/3,IT-L4,IT-L5'.  Groups present in --samples but absent "
             "here are appended in data order; names not in the data are "
             "ignored, so one order can be reused across subsets.",
    )
    parser.add_argument(
        "--batch-column", type=str, default=None,
        help="Column name(s) in --samples holding batch labels.  Accepts a "
             "single column (e.g. 'batch') or a COMMA-SEPARATED list of "
             "several batch sources (e.g. "
             "'culture_batch,libprep_batch,seq_batch').  When set, the mean "
             "accessibility matrix is batch-corrected (per --batch-correction) "
             "before the RWR k-NN graph is built.  With multiple columns, "
             "--batch-correction harmony harmonizes over all of them jointly "
             "(Harmony's native multi-covariate vars_use); center/zscore/"
             "combat collapse them to a single composite (cartesian-product) "
             "batch.  Names, not positional indices.  Cells lacking a label "
             "are pooled into '__unknown_batch__'.",
    )
    parser.add_argument(
        "--batch-correction",
        choices=["auto", "none", "center", "zscore", "combat", "harmony"],
        default="harmony",
        help="Batch-correction method applied to the mean accessibility "
             "matrix when --batch-column is set.  "
             "'center' = per-batch column-mean subtraction (no deps).  "
             "'zscore' = per-batch column-mean AND column-std normalization "
             "(no deps).  "
             "'combat' = empirical-Bayes batch correction (Johnson 2007) — "
             "requires `pip install combat` or `pip install inmoose`.  "
             "'harmony' = iterative cell-batch correction in PC space "
             "(Korsunsky 2019) — requires `pip install harmonypy`.  "
             "WARNING: center/zscore/combat subtract a per-batch mean, which "
             "removes the cell-type axis when cell-type composition is "
             "imbalanced across batches (empirically PC0 AUC 0.91→0.60).  "
             "Harmony is 'protected' but, in the full-set comparison, did not "
             "improve cell-type clustering under the DM default (DM+harmony"
             "+skip1 0.926 vs DM+none+skip1 0.937); its lone marginal win "
             "needs --skip-top-svd 0.  "
             "Default 'harmony' (needs --batch-column + `harmonypy`); it falls "
             "back to 'none' with a warning when no --batch-column is given.  "
             "'auto' resolves to 'harmony' when --batch-column is set, else "
             "'none'.",
    )
    parser.add_argument(
        "--harmony-theta", type=float, default=None,
        help="Harmony batch-correction STRENGTH (diversity-clustering penalty) "
             "for the Step-7 concordance-UMAP correction under "
             "--batch-correction harmony.  Higher mixes batches more "
             "aggressively; lower is gentler.  Broadcast to every --batch-column "
             "variable; omit to use harmonypy's default (2.0).")
    parser.add_argument(
        "--post-norm-recenter", choices=["auto", "on", "off"], default="auto",
        help="Apply per-cell mean re-centering as a FINAL step after "
             "per-bin normalization in Step 7.  Fixes the per-cell mean "
             "drift introduced by the non-commutativity of per-cell rank "
             "and per-bin QN: after rank → QN, the column re-ranking "
             "shuffles row values and breaks per-cell mean=0, which "
             "becomes the dominant SVD axis.  One extra row-mean "
             "subtraction restores per-cell mean=0 without re-disturbing "
             "per-bin centering (since col means ≈ 0 after QN).  "
             "Mathematically equivalent to a closed-form double-centering. "
             "'auto' (default): on for --concordance-method signed, off "
             "otherwise.  When on, eliminates the need for "
             "--skip-top-svd 1 (the artifact PC0 it was dropping is "
             "removed at the source).",
    )
    parser.add_argument(
        "--skip-top-svd", type=int, default=None,
        help="Drop the top N SVD components before UMAP / clustering.  "
             "Use when per-cell normalization isn't enough to remove a "
             "non-cell-type dominant axis (e.g., k-NN community structure "
             "from RWR, or a discrete depth-tier split).  Common values: "
             "1 (drop PC0 only).  Default depends on method: 1 for "
             "--concordance-method signed (empirically PC0 is a batch / "
             "technical axis on NOMe-HiC data even after per-cell "
             "rank normalization), 0 otherwise.  Set explicitly to 0 to "
             "force-disable.  Strictly orthogonal to --n-svd.",
    )
    parser.add_argument(
        "--posterior-threshold", type=float, default=0.5,
        help="Posterior cutoff above which a read end is called 'open' "
             "(only used by --concordance-method betabinom; default: 0.5). "
             "Not used by --concordance-method signed (which uses U-3 soft "
             "counts with no thresholding).",
    )
    parser.add_argument(
        "--posterior-threshold-open", type=float, default=0.6,
        help="Diagnostic-only threshold for calling an end 'open' "
             "(design-doc §6 #1). Used in QC summaries; "
             "the signed-Δ computation itself uses un-thresholded "
             "posteriors via U-3 soft counts.",
    )
    parser.add_argument(
        "--posterior-threshold-closed", type=float, default=0.4,
        help="Diagnostic-only threshold for calling an end 'closed' "
             "(design-doc §6 #1). See --posterior-threshold-open.",
    )
    parser.add_argument(
        "--min-linked-pairs-per-cell", type=int, default=100,
        help="Drop cells with fewer than this many qualifying linked "
             "pairs (design-doc §6 #14). Applied right "
             "after Step 2 so bad cells don't contaminate the mixture "
             "model, posterior calls, or downstream matrices. Set to 0 "
             "to disable.",
    )
    parser.add_argument(
        "--impute", choices=["none", "rwr"], default="rwr",
        help="Imputation method applied to the soft-count matrices "
             "before Layer-2 EB shrinkage (design-doc §6 #16). "
             "'rwr' (default) is required for typical NOMe-HiC data "
             "sparsity at 250 kb (median n_linked = 1 per (cell, bin) "
             "is prior-dominated without imputation). 'none' for "
             "diagnostic / debug. Only applies to --concordance-method "
             "signed; ignored for pearson and betabinom.",
    )
    parser.add_argument("--rwr-k", type=int, default=15,
                        help="Number of nearest neighbours per cell.")
    parser.add_argument("--rwr-restart-prob", type=float, default=0.5,
                        help="RWR restart probability r.")
    parser.add_argument("--rwr-refine-passes", type=str, default=None,
                        help="Iterative refinement passes: an integer (fixed "
                             "count) or 'auto'.  Each pass rebuilds the k-NN "
                             "from the previous pass's Δ_signed (see "
                             "--refinement-basis); RWR always restarts from the "
                             "original raw soft counts.  'auto' runs up to "
                             "--rwr-max-auto-passes and keeps the pass whose "
                             "post-norm Δ has the largest top SVD eigengap (a "
                             "label-free proxy for cell-type separation, "
                             "validated to peak with the cell-type AUC among "
                             "refined passes).  Default (unset): 'auto' for "
                             "--concordance-method signed, 0 otherwise.")
    parser.add_argument("--rwr-max-auto-passes", type=int, default=5,
                        help="Upper bound on refinement passes when "
                             "--rwr-refine-passes auto.")
    parser.add_argument("--rwr-solver", choices=["auto", "direct", "iterative"],
                        default="auto",
                        help="RWR linear solver. 'auto' uses direct (closed-"
                             "form LU) when n_cells <= 5000; iterative otherwise.")
    parser.add_argument("--rwr-max-iter", type=int, default=30,
                        help="Max iterations for the iterative RWR solver.")
    parser.add_argument("--rwr-tol", type=float, default=1e-4,
                        help="Convergence tolerance for the iterative RWR solver.")
    parser.add_argument(
        "--eb-min-obs", type=int, default=50,
        help="Minimum number of (cell, bin) observations with n_linked>=1 "
             "required to fit the empirical-Bayes prior on θ_L.  If fewer "
             "are available, falls back to Beta(1, 1) or Dirichlet(1,1,1).  "
             "Default: 50.",
    )
    parser.add_argument(
        "--n-svd", type=int, default=None,
        help="Reduce the normalized matrix to N components via TruncatedSVD "
             "before running UMAP (bulk-RNA / scATAC-seq style).  An explicit "
             "0 runs UMAP directly on the FULL bin matrix (also pass "
             "--skip-top-svd 0, since SVD still runs when skip>0).  When UNSET "
             "it auto-defaults: 10 for --concordance-method signed (the "
             "cell-type axis is PC0 but only a few %% of variance, so the full "
             "bin matrix swamps it), 0 otherwise.  A scree plot is always "
             "emitted to help tune this.  (An explicit value — including 0 — "
             "overrides the auto-default.)",
    )
    parser.add_argument(
        "--svd-scree-max", type=int, default=50,
        help="Number of top SVD components to fit for the scree-plot "
             "diagnostic. Clamped to min(this, n_cells-1, "
             "n_bins).",
    )

    # EM
    parser.add_argument("--em-max-iter", type=int, default=200,
                        help="Max EM iterations for the beta-binomial mixture.")
    parser.add_argument("--em-tol", type=float, default=1e-6,
                        help="EM convergence tolerance (log-likelihood delta).")
    parser.add_argument("--em-restarts", type=int, default=5,
                        help="Random restarts for EM fitting.")
    parser.add_argument("--short-range-max", type=int, default=1_000,
                        help="Max distance for short-range diagnostic class.")

    # Shrinkage
    parser.add_argument("--shrinkage-n0", type=str, default="auto",
                        help='Shrinkage pseudocount (float or "auto" = empirical Bayes).')

    # SNP / context
    parser.add_argument("--dbsnp-vcf", type=str, default=None,
                        help="Tabix-indexed dbSNP VCF for SNP filtering of context sites.")
    parser.add_argument("--contexts", type=str, default="GCH",
                        help='Comma-separated IUPAC cytosine contexts to analyse.')
    parser.add_argument(
        "--context-bed", type=str, default=None,
        help=(
            "Load context positions from tabix-indexed BED files instead "
            "of extracting from the .2bit reference (much faster). "
            "Format: CONTEXT:PATH[,CONTEXT:PATH,...] e.g. "
            '"GCH:/data/GCH.bed.gz" or '
            '"GCH:/data/GCH.bed.gz,HCG:/data/HCG.bed.gz". '
            "Each BED must be bgzipped with a .tbi index. "
            "Accepted columns: (chrom, start, end, strand_or_refbase) "
            "or standard BED6. When provided, Step 1 .2bit extraction "
            "is skipped and --contexts is inferred from the BED paths."
        ),
    )
    parser.add_argument(
        "--context-bed-offset", type=int, default=0,
        help=(
            "Offset from BED start (col 2, 0-based) to the cytosine "
            "position. Use 0 (default) when the BED interval already "
            "spans the C itself (e.g. start=10481 end=10482 means C "
            "is at 0-based position 10481). Use 1 if the BED start is "
            "the first base of the full trinucleotide motif (the G of "
            "GCH), so the C is at start+1."
        ),
    )

    # Read processing
    parser.add_argument("--trim-ends", type=int, default=5,
                        help="Bases to trim from read ends.")
    add_include_chrom_arg(parser)
    parser.add_argument(
        "--mixture-exclude-chrom", action="append", default=None,
        metavar="CHROM",
        help="Contigs kept OUT of the step-3 beta-binomial fit, repeatable "
             "and/or comma-separated.  chrM carries no nucleosomes, so its "
             "read ends are not draws from the accessible/inaccessible "
             "mixture being fitted; including them moves the components and "
             "with them the posterior of every NUCLEAR site, which turns "
             "adding a reference into silently changing every existing "
             "result.  DEFAULTS to whatever --include-chrom added, since "
             "such a contig is there to calibrate rather than to be "
             "modelled.  Posteriors are still computed for it, under the "
             "nuclear model -- which is what makes the reference comparable "
             "to the bins it corrects.  Pass 'none' to fit on everything.")
    parser.add_argument(
        "--keep-duplicates-chrom", action="append", default=None,
        metavar="CHROM",
        help="Contigs on which the BAM duplicate flag is IGNORED, "
             "repeatable and/or comma-separated.  chrM is present in "
             "hundreds to thousands of copies per cell, so on 16.6 kb two "
             "reads sharing coordinates collide by chance rather than by "
             "PCR, and the marker's calls there are wrong.  It costs more "
             "than depth: deduplication keeps one read per coordinate, "
             "collapsing molecules that differ in methylation, which "
             "destroys the between-molecule variance a reliability "
             "reference measures.  DEFAULTS to whatever --include-chrom "
             "added, since that is the only reason to add a contig; pass "
             "'none' to keep the normal filtering everywhere.  Every other "
             "contig is unaffected and keeps the fast path.")
    parser.add_argument("--mapq-min", type=int, default=30,
                        help="Minimum mapping quality on both ends.")
    parser.add_argument("--require-proper-pair", action="store_true",
                        help="Require BAM proper-pair flag. "
                             "BWA-MEM does NOT flag long-range Hi-C ligation "
                             "products as proper pairs, so enabling this filter "
                             "silently discards most cis Hi-C contacts. Only "
                             "enable for non-Hi-C short-insert paired-end data.")
    parser.add_argument("--keep-supplementary", action="store_true",
                        help="Keep supplementary alignments (chimeric-read "
                             "splits). Default off — standard Hi-C practice. "
                             "Enable if you suspect bhmem assigns 'primary' "
                             "to the wrong half of a junction-spanning chimera.")

    # EM fitting options
    parser.add_argument("--em-subsample", type=int, default=10_000_000,
                        help="Max read ends for EM fitting. "
                             "Use 0 to disable subsampling and fit on all read ends.")
    parser.add_argument("--em-cell-specific", action="store_true",
                        help="Also fit EM per-cell for diagnostic comparison.")

    # Execution
    parser.add_argument("--n-threads", type=int, default=4,
                        help="Number of parallel threads.")
    parser.add_argument("--steps", type=str, default="all",
                        help='Steps to run: comma-separated (1-7) or "all".')
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Additional log file.")

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _manifest_text(path: str) -> io.StringIO:
    """Manifest text with WHOLE-LINE ``#`` comments dropped.

    Deliberately not pandas' ``comment='#'``: that also strips INLINE ``#...``,
    which silently truncates a row at a hex colour like ``#FF9A4C`` — blanking
    that field *and every column to its right* (so --color-column and the batch
    columns all came back NaN).  Only full-line comments are removed here, which
    keeps ``#`` usable inside a field.
    """
    with open(path) as fh:
        kept = [ln for ln in fh if not ln.lstrip().startswith("#")]
    return io.StringIO("".join(kept))


def load_samples(path: str) -> pd.DataFrame:
    """Load a TSV manifest: cell_name, bam_path, optional group.

    Accepts files with or without a header row.  If the first row looks
    like a header (first field equals ``cell_name``), it is consumed as
    column names; otherwise the file is treated as headerless.  ``#`` starts a
    comment only at the START of a line, so hex colours are safe in any field.
    """
    # Peek at the first non-comment line to decide whether a header is present.
    first_line = ""
    with open(path) as fh:
        for line in fh:
            if line.strip() and not line.lstrip().startswith("#"):
                first_line = line.strip()
                break
    first_field = first_line.split("\t")[0].strip().lower()
    has_header = first_field in ("cell_name", "cellname", "cell", "sample", "sample_name")

    if has_header:
        df = pd.read_csv(_manifest_text(path), sep="\t", dtype=str)
        # Normalise column names coming from the header.
        col_map = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
        df.rename(columns=col_map, inplace=True)
        # Accept common synonyms.
        if "cellname" in df.columns:
            df.rename(columns={"cellname": "cell_name"}, inplace=True)
        if "sample" in df.columns and "cell_name" not in df.columns:
            df.rename(columns={"sample": "cell_name"}, inplace=True)
        if "sample_name" in df.columns and "cell_name" not in df.columns:
            df.rename(columns={"sample_name": "cell_name"}, inplace=True)
    else:
        df = pd.read_csv(
            _manifest_text(path), sep="\t", header=None,
            names=["cell_name", "bam_path", "group"],
            engine="python", dtype=str,
        )

    if "bam_path" not in df.columns:
        raise ValueError(
            f"Sample manifest {path} must have a 'bam_path' column (or be a "
            f"headerless TSV with columns: cell_name, bam_path, [group])."
        )

    if "group" not in df.columns:
        df["group"] = "unknown"
    df["group"] = df["group"].fillna("unknown")

    n_groups = df["group"].nunique()
    logger.info(f"Loaded {len(df)} samples, {n_groups} group(s).")
    return df


def group_colors_from_samples(samples: pd.DataFrame,
                              color_column: str | None) -> dict | None:
    """``group -> colour`` from a per-cell colour column in --samples.

    The legend is per group, so each group needs one colour: the group's first
    colour wins and any group carrying several is named in a warning.  Colours
    are validated up front (matplotlib hex/name) so a typo fails loudly here
    rather than deep inside plotting.  Returns None when no column is given.
    """
    if not color_column:
        return None
    if color_column not in samples.columns:
        raise ValueError(
            f"--color-column {color_column!r} not in the sample manifest. "
            f"Available columns: {samples.columns.tolist()}"
        )
    from matplotlib.colors import is_color_like

    sub = samples[["group", color_column]].dropna(subset=[color_column])
    mapping: dict = {}
    for g, vals in sub.groupby("group")[color_column]:
        uniq = list(dict.fromkeys(vals))          # unique, first-seen order
        if len(uniq) > 1:
            logger.warning(
                f"--color-column {color_column!r}: group {g!r} has "
                f"{len(uniq)} colours {uniq}; using {uniq[0]!r}."
            )
        if not is_color_like(uniq[0]):
            raise ValueError(
                f"--color-column {color_column!r}: {uniq[0]!r} (group {g!r}) is "
                f"not a matplotlib colour (use a hex code like '#1f77b4' or a "
                f"named colour)."
            )
        mapping[g] = uniq[0]
    missing = sorted(set(samples["group"]) - set(mapping))
    if missing:
        logger.warning(
            f"--color-column {color_column!r}: no colour for group(s) {missing}; "
            f"they fall back to the default palette."
        )
    return mapping


def _keep_dup_chroms(args):
    """Contigs whose duplicate flag is ignored, as canonical keys.

    Defaults to whatever --include-chrom added rather than to nothing.  The
    only reason to add a contig outside the default set is to use it as a
    reliability reference, and a deduplicated reference is not merely thinner
    -- it is wrong, because collapsing coordinate-sharing molecules removes
    the between-molecule variance being measured.  Defaulting to empty would
    make the quiet failure the easy one to hit, so opting out is explicit.
    """
    raw = getattr(args, "keep_duplicates_chrom", None)
    if raw is None:
        return {chrom_key(c)
                for c in parse_include_chroms(
                    getattr(args, "include_chrom", None))}
    names = parse_include_chroms(raw)
    if [n.lower() for n in names] == ["none"]:
        return set()
    return {chrom_key(c) for c in names}


def _mixture_exclude_chroms(args):
    """Contigs kept out of the beta-binomial fit, as canonical keys.

    Defaults to whatever --include-chrom added, for the same reason
    :func:`_keep_dup_chroms` does: a contig outside the default set is there
    to calibrate, not to be modelled.  chrM carries no nucleosomes, so its
    read ends are not draws from the accessible/inaccessible mixture at all;
    fitting on them moves the components and therefore the posterior of every
    NUCLEAR site -- turning "add a reference" into "silently change all the
    existing results".  Pass 'none' to fit on everything.
    """
    raw = getattr(args, "mixture_exclude_chrom", None)
    if raw is None:
        return {chrom_key(c)
                for c in parse_include_chroms(
                    getattr(args, "include_chrom", None))}
    names = parse_include_chroms(raw)
    if [n.lower() for n in names] == ["none"]:
        return set()
    return {chrom_key(c) for c in names}


def _parse_context_bed(spec: str) -> dict[str, str]:
    """Parse ``'GCH:/path/to/GCH.bed.gz,HCG:/path/to/HCG.bed.gz'`` into a dict."""
    result: dict[str, str] = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                f"--context-bed entries must be CONTEXT:PATH, got {token!r}"
            )
        ctx, path = token.split(":", 1)
        ctx = ctx.strip().upper()
        path = path.strip()
        if not path:
            raise ValueError(f"Empty path for context {ctx!r} in --context-bed")
        result[ctx] = path
    return result


def _parse_steps(spec: str) -> set[int]:
    """Parse ``'all'`` or e.g. ``'2,3,4'`` into an explicit set of steps."""
    spec = spec.strip().lower()
    if spec == "all":
        return set(range(1, 8))
    return {int(s) for s in spec.split(",") if s.strip()}


def _get_chrom_sizes(reference_path: str) -> dict[str, int]:
    """Read chromosome sizes from a .2bit reference."""
    import py2bit
    tb = py2bit.open(reference_path)
    try:
        return dict(tb.chroms())
    finally:
        tb.close()


def _count_linked_pairs_one_cell(
    parquet_path: str,
    cell_name: str,
    bin_size: int,
    min_distance: int,
    min_gch_sites: int,
) -> tuple[str, int]:
    """Count qualifying linked pairs in one cell (per-cell parquet from Step 2).

    Used by the cell-coverage filter to decide which cells to drop.  Does
    NOT require posteriors — only n1, n2, distance, and same-bin check.
    """
    df = pd.read_parquet(
        parquet_path,
        columns=["chrom", "pos1", "pos2", "n1", "n2", "distance"],
    )
    if len(df) == 0:
        return cell_name, 0
    pos1 = df["pos1"].to_numpy(dtype=np.int64)
    pos2 = df["pos2"].to_numpy(dtype=np.int64)
    bsz = int(bin_size)
    bin1 = pos1 // bsz
    bin2 = pos2 // bsz
    informative = (df["n1"].to_numpy() >= min_gch_sites) & (
        df["n2"].to_numpy() >= min_gch_sites
    )
    same_bin = bin1 == bin2
    long_range = df["distance"].to_numpy() >= int(min_distance)
    linked = same_bin & informative & long_range
    return cell_name, int(linked.sum())


def _filter_low_coverage_cells(
    cell_parquets: dict[str, str],
    min_linked_pairs: int,
    min_distance: int,
    min_gch_sites: int,
    bin_size: int,
    n_threads: int,
    qc_dir,
) -> dict[str, str]:
    """Drop cells with fewer than ``min_linked_pairs`` qualifying linked pairs.

    Saves the kept and dropped cell lists with their counts to ``qc_dir``
    so the user can inspect what was removed.  Returns a new dict
    containing only the kept cells.

    Design-doc §6 #14.
    """
    import numpy as np
    from joblib import Parallel, delayed

    logger.info(
        f"Cell coverage filter: dropping cells with < {min_linked_pairs} "
        f"qualifying linked pairs."
    )

    items = list(cell_parquets.items())
    if n_threads > 1 and len(items) > 1:
        counts = Parallel(n_jobs=n_threads)(
            delayed(_count_linked_pairs_one_cell)(
                path, name, bin_size, min_distance, min_gch_sites,
            )
            for name, path in items
        )
    else:
        counts = [
            _count_linked_pairs_one_cell(
                path, name, bin_size, min_distance, min_gch_sites,
            )
            for name, path in items
        ]

    # Build kept / dropped tables.
    counts_dict = dict(counts)  # cell_name -> n_linked
    kept: dict[str, str] = {}
    dropped_rows: list[tuple[str, int]] = []
    for name, parquet_path in items:
        n_linked = counts_dict[name]
        if n_linked >= min_linked_pairs:
            kept[name] = parquet_path
        else:
            dropped_rows.append((name, n_linked))

    kept_rows = [(name, counts_dict[name]) for name in kept]
    qc_path = ensure_dir(qc_dir)
    pd.DataFrame(kept_rows, columns=["cell", "n_linked"]).to_csv(
        qc_path / "cells_used.tsv", sep="\t", index=False,
    )
    if dropped_rows:
        pd.DataFrame(dropped_rows, columns=["cell", "n_linked"]).to_csv(
            qc_path / "cells_dropped.tsv", sep="\t", index=False,
        )
        logger.warning(
            f"  Dropped {len(dropped_rows)} of {len(items)} cells "
            f"(threshold n_linked >= {min_linked_pairs}). "
            f"Kept {len(kept)} cells. "
            f"See {qc_path / 'cells_dropped.tsv'} for the dropped cell list."
        )
    else:
        logger.info(
            f"  All {len(items)} cells pass n_linked >= {min_linked_pairs}."
        )

    return kept


def _get_chrom_sizes_from_bam(bam_path: str) -> dict[str, int]:
    """Read chromosome sizes from a BAM file's @SQ header lines."""
    import pysam
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        return {
            sq["SN"]: sq["LN"]
            for sq in bam.header.to_dict().get("SQ", [])
        }


def _correct_concordance_batch(df, batch_labels, method: str,
                               random_state: int, label: str):
    """Step-7 concordance-matrix batch correction.

    Lives in batch_correction so nome_hic_compare applies the identical
    transform to the same matrices; see
    :func:`batch_correction.correct_and_report` for why this stage exists at
    all and why it has to precede the per-bin normalization.
    """
    from . import batch_correction as _bc
    return _bc.correct_and_report(df, batch_labels, method, random_state,
                                  label)


def _resolve_concordance_combine(args) -> str:
    """Resolve ``--concordance-combine auto``.

    auto → log2-ratio for the non-contact O/E baselines (distance-matched /
    distance-pooled), signed-diff for the legacy short-range baseline
    (preserves the validated default).  Module-level so Step 7 can resolve
    the same value when Step 5 didn't run in this invocation.
    """
    if args.concordance_combine != "auto":
        return args.concordance_combine
    if args.baseline in ("distance-matched", "distance-pooled"):
        return "log2-ratio"
    return "signed-diff"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------



def _step5_batch_sweep(sig_result, posterior_paths, chrom_sizes, step5_kwargs,
                       samples, args, group_labels, out_dir):
    """Sweep the Step-5 (RWR-basis) batch correction and rank the results.

    This axis cannot be swept from a finished Δ matrix the way the Step-7 ones
    can: correcting the accessibility matrix changes the k-NN graph, so it
    changes what RWR imputes and therefore the concordance values themselves.
    Each setting needs its own Step 5.

    What makes that affordable is that steps 1-3 -- reading every cell's
    posteriors, aggregating soft counts, building the accessibility matrix --
    do not depend on the correction, which first acts at step 3.5.  They are
    computed once and replayed through ``precomputed``.

    Each resulting Δ is scored through the same Step-7 path the real run uses
    (process_concordance_matrix -> umap_input_scores -> label_agreement), with
    the downstream settings resolved by _resolve_step7_params.
    """
    import itertools

    from . import matrix_ops as _mo
    from . import metrics as _metrics
    from . import visualization as _vis

    methods = [m.strip() for m in args.step5_batch_sweep.split(",") if m.strip()]
    cols_all = [c.strip() for c in (args.batch_column or "").split(",")
                if c.strip()]
    if not cols_all:
        logger.warning("--step5-batch-sweep needs --batch-column; skipping.")
        return
    subsets = [tuple(cols_all[i] for i in idx)
               for r in range(1, len(cols_all) + 1)
               for idx in itertools.combinations(range(len(cols_all)), r)]
    cache = {k: sig_result[k] for k in
             ("raw_counts", "accessibility_raw", "cell_names", "sorted_bins",
              "uses_expected")}
    s7 = _resolve_step7_params(args)
    idx_s = samples.set_index("cell_name")

    rows = []
    combos = [(m, s) for m in methods for s in subsets
              if not (m == "none" and s != subsets[0])]
    logger.info(f"Step-5 batch sweep: {len(combos)} setting(s); steps 1-3 are "
                f"reused, so each costs only RWR + Layer 2.")
    for i, (method, cols) in enumerate(combos, 1):
        bl = idx_s[cols[0]] if len(cols) == 1 else idx_s[list(cols)]
        try:
            res = concordance_mod.compute_signed_concordance_matrix(
                posterior_paths, chrom_sizes, precomputed=cache,
                batch_labels=(None if method == "none" else bl),
                batch_correction=method, save_outputs=False, **step5_kwargs)
            delta = res["delta_signed"]
            norm = _mo.process_concordance_matrix(
                delta, min_cell_coverage=args.min_cell_coverage,
                output_dir="", matrix_norm=s7["matrix_norm"],
                per_cell_norm=s7["per_cell_norm"],
                post_norm_recenter=s7["post_norm_recenter"],
                save_outputs=False)["zscore"]
            common = norm.index.intersection(group_labels.index)
            X, _ = _vis.umap_input_scores(
                norm.loc[common], n_svd=s7["n_svd"],
                skip_top_svd=s7["skip_top_svd"], n_threads=args.n_threads,
                random_state=args.random_seed, batch_correction="none")
            m = _metrics.label_agreement(X, group_labels.loc[common],
                                         random_state=args.random_seed,
                                         n_perm=0)
            err = ""
        except Exception as exc:
            m = {k: float("nan") for k in
                 ("silhouette", "knn_purity", "knn_chance", "purity_index",
                  "ari", "nmi")}
            err = f"{type(exc).__name__}: {exc}"
            logger.warning(f"  [{i}/{len(combos)}] {method}/{'+'.join(cols)} "
                           f"FAILED — {err}")
        rows.append({"step5_batch_correction": method,
                     "step5_batch_column": ("-" if method == "none"
                                            else "+".join(cols)),
                     "knn_purity": m["knn_purity"],
                     "purity_index": m["purity_index"], "ari": m["ari"],
                     "nmi": m["nmi"], "silhouette": m["silhouette"],
                     "knn_chance": m["knn_chance"], "error": err})
        logger.info(f"  [{i}/{len(combos)}] {method}/{'+'.join(cols)}: "
                    f"purity={m['knn_purity']:.4f} ari={m['ari']:.4f} "
                    f"nmi={m['nmi']:.4f} asw={m['silhouette']:.4f}")

    tab = pd.DataFrame(rows).sort_values("knn_purity", ascending=False)
    path = Path(out_dir) / "step5_batch_sweep.tsv"
    tab.to_csv(path, sep="\t", index=False)
    logger.info(f"Step-5 batch sweep written to {path}")
    if tab["knn_purity"].notna().any():
        best = tab.iloc[0]
        logger.info(
            f"Best Step-5 setting by kNN purity: --batch-correction "
            f"{best['step5_batch_correction']} --batch-column "
            f"{best['step5_batch_column'].replace('+', ',')} "
            f"(purity {best['knn_purity']:.4f}). NOTE this changes the Δ "
            f"matrix, so steps 5-7 must be re-run with it — a Step-7-only "
            f"rerun cannot apply it.")


def _resolve_step7_params(args) -> dict:
    """Resolve the Step-7 'auto' settings once.

    Shared with --step5-batch-sweep so a swept setting is scored through the
    same normalization / SVD / skip choices the real Step 7 would apply; a
    second copy of these rules would let the sweep rank settings under a
    pipeline that never runs.
    """
    # Resolve --per-cell-norm.  --per-cell-center is the legacy
    # boolean form; --per-cell-norm wins if both are set explicitly
    # (i.e., neither at 'auto').
    if args.per_cell_norm != "auto":
        per_cell_norm = args.per_cell_norm
    elif args.per_cell_center == "on":
        per_cell_norm = "center"
    elif args.per_cell_center == "off":
        per_cell_norm = "off"
    else:  # both auto
        # 'rank' default for signed (empirically the most effective on
        # NOMe-HiC data — see methodology debug log in design doc).
        per_cell_norm = (
            "rank" if args.concordance_method == "signed" else "off"
        )
    per_cell_center = per_cell_norm == "center"  # for legacy boolean param

    # Resolve --post-norm-recenter auto → on for signed.
    if args.post_norm_recenter == "auto":
        post_norm_recenter = args.concordance_method == "signed"
    else:
        post_norm_recenter = args.post_norm_recenter == "on"

    # Resolve --matrix-norm auto → quantile for signed (rank-based QN
    # is the validated Step-7 normalization for Δ_signed), zscore
    # otherwise (the classical pipeline).
    if args.matrix_norm == "auto":
        matrix_norm = (
            "quantile" if args.concordance_method == "signed" else "zscore"
        )
    else:
        matrix_norm = args.matrix_norm

    # Resolve --skip-top-svd auto-default.  On the FULL cell set the
    # dominant PC0 of Δ_signed is a non-cell-type confound (the
    # low-coverage cells), with cell type on PC1 — for BOTH estimators.
    # So signed defaults to skip 1 (drop the confound; full-set kNN
    # 0.884 → 0.937 for DM).  The exception is explicit MLE, where skip
    # didn't help (kNN ~flat), so leave it at 0 there.  Non-signed
    # methods never skip.  User-explicit values (including 0) override.
    _signed_formula = (
        "mle" if args.concordance_formula == "mle" else "dm"
    )
    if args.skip_top_svd is None:
        if args.concordance_method == "signed" and _signed_formula == "dm":
            skip_top_svd = 1
        else:
            skip_top_svd = 0
    else:
        skip_top_svd = args.skip_top_svd

    # Resolve --n-svd auto-default for the UMAP pre-reduction.  SVD only
    # runs when n_svd>0 OR skip>0 (see visualization.run_umap_and_plot).
    # Under signed MLE, skip resolves to 0, so without an n_svd default
    # UMAP would run on the full ~10k-bin matrix and the cell-type PC0
    # signal (a few % variance) would be swamped by noise dimensions.
    # Default signed to 10 components — exactly the count signed used
    # before (skip=1 fit 11, dropped the top → 10), but now keeping PC0
    # since MLE puts cell type there.  The scree plot is always emitted
    # for tuning; user-explicit --n-svd overrides.
    # None = unset -> auto-default; an explicit value (INCLUDING 0, which
    # means "no SVD, UMAP on the full bin matrix") is honoured as given.
    if args.n_svd is None:
        n_svd = 10 if args.concordance_method == "signed" else 0
    else:
        n_svd = args.n_svd
    return {"per_cell_norm": per_cell_norm,
            "per_cell_center": per_cell_center,
            "post_norm_recenter": post_norm_recenter,
            "matrix_norm": matrix_norm,
            "skip_top_svd": skip_top_svd,
            "n_svd": n_svd}


def run_context(
    context: str,
    context_positions: dict[str, np.ndarray],
    samples: pd.DataFrame,
    ctx_out: Path,
    steps: set[int],
    args: argparse.Namespace,
    chrom_sizes: dict[str, int],
) -> None:
    """Run the per-context pipeline (steps 2-7) for a single context."""
    logger.info("#" * 60)
    logger.info(f"Processing context: {context}")
    logger.info("#" * 60)

    # Pick the per-end site threshold for THIS context.  --min-gch-sites
    # applies when context == GCH; --min-hcg-sites when context == HCG
    # (falling back to --min-gch-sites if not set).  Any other context
    # falls back to --min-gch-sites for now.
    if context == "GCH":
        min_sites = args.min_gch_sites
    elif context == "HCG":
        min_sites = (
            args.min_hcg_sites if args.min_hcg_sites is not None
            else args.min_gch_sites
        )
    else:
        min_sites = args.min_gch_sites
    logger.info(f"Using min sites per end = {min_sites} for context {context}")

    # Resolve batch correction ONCE, so it is available to BOTH Step 5 (the
    # mean-accessibility / RWR-basis correction) AND Step 7 (the concordance-
    # UMAP correction) -- even on a Step-7-only rerun.
    batch_labels = None
    if args.batch_column is not None:
        batch_cols = [c.strip() for c in args.batch_column.split(",")
                      if c.strip()]
        missing = [c for c in batch_cols if c not in samples.columns]
        if missing:
            raise ValueError(
                f"--batch-column {missing} not in {args.samples}. "
                f"Available columns: {samples.columns.tolist()}")
        idx = samples.set_index("cell_name")
        batch_labels = (idx[batch_cols[0]] if len(batch_cols) == 1
                        else idx[batch_cols])

    # The concordance-level correction may use a DIFFERENT set of batch
    # variables from the score-level one: they act on different matrices at
    # different stages, and a sweep can well find that e.g. the plate effect
    # shows up per bin while the experiment effect shows up in the embedding.
    # Falls back to --batch-column when unset, so the common case stays a
    # single flag.
    concordance_batch_labels = batch_labels
    if getattr(args, "concordance_batch_column", None):
        cbc_cols = [c.strip() for c in args.concordance_batch_column.split(",")
                    if c.strip()]
        missing = [c for c in cbc_cols if c not in samples.columns]
        if missing:
            raise ValueError(
                f"--concordance-batch-column {missing} not in {args.samples}. "
                f"Available columns: {samples.columns.tolist()}")
        idx2 = samples.set_index("cell_name")
        concordance_batch_labels = (idx2[cbc_cols[0]] if len(cbc_cols) == 1
                                    else idx2[cbc_cols])
        logger.info(f"Concordance-level batch correction will use "
                    f"{cbc_cols} (score-level uses "
                    f"{args.batch_column or 'nothing'}).")

    if args.batch_correction == "auto":
        batch_correction = "harmony" if batch_labels is not None else "none"
    else:
        batch_correction = args.batch_correction
    # Batch correction needs labels.  Since the default is 'harmony', a run
    # without --batch-column would otherwise fail; fall back to 'none' safely.
    if batch_correction != "none" and batch_labels is None:
        logger.warning(
            f"--batch-correction {batch_correction!r} needs --batch-column, "
            f"which was not provided; running with batch correction OFF.")
        batch_correction = "none"
    if batch_correction != "none" and 5 not in steps:
        logger.info(
            f"Batch correction ({batch_correction}): Step 5 not in --steps, so "
            f"the mean-accessibility / RWR-basis correction is not re-applied "
            f"(already baked into the concordance matrix on disk); the Step 7 "
            f"concordance-UMAP correction still runs.")

    ensure_dir(ctx_out)

    read_pair_dir = ctx_out / "02_read_pair_counts"
    mixture_dir = ctx_out / "03_mixture_model"
    posterior_dir = ctx_out / "04_posteriors"
    concordance_dir = ctx_out / "05_concordance"
    umap_dir = ctx_out / "06_umap"

    # ---- Step 2: BAM parsing ----------------------------------------------
    if 2 in steps:
        logger.info("=" * 60)
        logger.info(f"Step 2: Parsing BAM files for {context}")
        logger.info("=" * 60)
        cell_parquets = bam_parser.parse_all_cells(
            samples, context_positions,
            str(read_pair_dir),
            n_threads=args.n_threads,
            trim_ends=args.trim_ends, mapq_min=args.mapq_min,
            require_proper_pair=args.require_proper_pair,
            keep_supplementary=args.keep_supplementary,
            keep_duplicates_chroms=_keep_dup_chroms(args),
            blacklist_bed=args.blacklist_bed,
        )
    else:
        # Only load step 2 output if a downstream step needs it.
        if not ({3, 4, 5, 6, 7} & steps):
            logger.info("Step 2: Skipped (no downstream step requires it)")
            return
        logger.info(f"Step 2: Skipped (loading per-cell parquets from {read_pair_dir})")
        cell_parquets = {
            row["cell_name"]: str(read_pair_dir / f"{row['cell_name']}.read_pairs.parquet")
            for _, row in samples.iterrows()
        }
        missing = [c for c, p in cell_parquets.items() if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                f"Read-pair parquets missing for {len(missing)} cell(s); "
                f"re-run step 2 or populate {read_pair_dir}."
            )

    # ---- Step 2.5: Low-coverage cell filter ------------------------------
    # (design-doc §6 #14): drop cells with fewer than min_linked_pairs_per_cell
    # qualifying linked pairs, applied right after Step 2 so bad cells don't
    # contaminate the mixture model, posterior calls, or downstream matrices.
    if args.min_linked_pairs_per_cell > 0 and ({3, 4, 5, 6, 7} & steps):
        cell_parquets = _filter_low_coverage_cells(
            cell_parquets,
            min_linked_pairs=args.min_linked_pairs_per_cell,
            min_distance=args.min_distance,
            min_gch_sites=min_sites,
            bin_size=args.bin_size,
            n_threads=args.n_threads,
            qc_dir=ctx_out / "qc",
        )

    # ---- Step 3: Mixture model -------------------------------------------
    if 3 in steps:
        logger.info("=" * 60)
        logger.info(f"Step 3: Fitting beta-binomial mixture for {context}")
        logger.info("=" * 60)

        # Stream-subsample: reads one cell at a time instead of loading all
        # parquets into memory.  Peak RAM ≈ one parquet + subsample pool
        # (~400 MB) instead of ~60–95 GB for the full concatenation.
        fit_results = mixture_model.fit_and_compare_from_parquets(
            cell_parquets,
            short_range_max=args.short_range_max,
            min_distance=args.min_distance,
            output_dir=str(mixture_dir),
            max_subsample=args.em_subsample,
            random_state=args.random_seed,
            max_iter=args.em_max_iter,
            tol=args.em_tol,
            n_restarts=args.em_restarts,
            exclude_chroms=_mixture_exclude_chroms(args),
        )
        fitted_model = fit_results["models"]["all"]

        if args.em_cell_specific:
            logger.info("Fitting cell-specific mixture models for diagnostic...")
            mixture_model.fit_cell_specific_models(
                cell_parquets, fitted_model,
                n_threads=args.n_threads,
                output_dir=str(mixture_dir),
                group_labels=samples.set_index("cell_name")["group"],
                exclude_chroms=_mixture_exclude_chroms(args),
            )
    else:
        # Only load step 3 output if a downstream step needs it.
        if not ({4, 5, 6, 7} & steps):
            logger.info("Step 3: Skipped (no downstream step requires it)")
            return
        em_fit_path = mixture_dir / "em_fit_all.json"
        if not em_fit_path.exists():
            raise FileNotFoundError(
                f"Mixture fit JSON not found at {em_fit_path}; re-run step 3."
            )
        logger.info(f"Step 3: Skipped (loading fit from {em_fit_path})")
        fitted_model = mixture_model.BetaBinomialMixture.from_json(str(em_fit_path))

    # ---- Step 4: Posteriors ----------------------------------------------
    if 4 in steps:
        logger.info("=" * 60)
        logger.info(f"Step 4: Computing posteriors for {context}")
        logger.info("=" * 60)
        posterior_paths = posterior.compute_all_posteriors(
            cell_parquets, fitted_model, str(posterior_dir),
            n_threads=args.n_threads,
        )
    else:
        # Only load step 4 output if a downstream step needs it.
        if not ({5, 6, 7} & steps):
            logger.info("Step 4: Skipped (no downstream step requires it)")
            return
        # Build the candidate cell list from the post-Step-2 filter
        # (cell_parquets), not the raw manifest.  When the current
        # --min-linked-pairs-per-cell / --min-gch-sites etc. differ from
        # the values used in the original run, the filter may keep cells
        # that don't have posteriors yet (e.g., switching from
        # --min-gch-sites 3 to 2 makes more pairs qualify, letting some
        # previously-failing cells pass the current threshold).  We
        # silently drop such cells with a warning rather than crashing
        # — the user can re-run --steps 4 explicitly to process them.
        posterior_paths = {
            cell: str(posterior_dir / f"{cell}.posteriors.parquet")
            for cell in cell_parquets.keys()
        }
        missing = [c for c, p in posterior_paths.items() if not Path(p).exists()]
        if missing:
            logger.warning(
                f"Step 4: {len(missing)} cell(s) pass the current Step-2.5 "
                f"filter but have no posterior parquet on disk (likely "
                f"because the original Step-4 run used different "
                f"--min-gch-sites / --min-linked-pairs-per-cell). "
                f"Dropping them from downstream analysis.  To process "
                f"these cells, re-run --steps 4."
            )
            posterior_paths = {
                c: p for c, p in posterior_paths.items() if Path(p).exists()
            }
            if not posterior_paths:
                raise FileNotFoundError(
                    f"No posterior parquets found in {posterior_dir} after "
                    f"applying the cell filter; re-run --steps 4."
                )
        logger.info(
            f"Step 4: Skipped (loading {len(posterior_paths)} posteriors "
            f"from {posterior_dir})"
        )

    # ---- Step 5: Concordance + distance decay -----------------------------
    if 5 in steps:
        logger.info("=" * 60)
        logger.info(
            f"Step 5: Computing per-bin concordance for {context} "
            f"(method={args.concordance_method})"
        )
        logger.info("=" * 60)

        if args.concordance_method == "pearson":
            concordance_raw, pair_counts = concordance_mod.compute_concordance_matrix(
                posterior_paths, chrom_sizes,
                bin_size=args.bin_size,
                min_distance=args.min_distance,
                min_gch_sites=min_sites,
                min_pairs_per_bin=args.min_pairs_per_bin,
                n_threads=args.n_threads,
                output_dir=str(concordance_dir),
            )
        elif args.concordance_method == "betabinom":
            concordance_raw, pair_counts = concordance_mod.compute_betabinom_concordance_matrix(
                posterior_paths, chrom_sizes,
                bin_size=args.bin_size,
                min_distance=args.min_distance,
                min_gch_sites=min_sites,
                posterior_threshold=args.posterior_threshold,
                min_pairs_per_bin=args.min_pairs_per_bin,
                eb_min_obs=args.eb_min_obs,
                n_threads=args.n_threads,
                output_dir=str(concordance_dir),
            )
        elif args.concordance_method == "signed":
            # batch_labels / batch_correction were resolved once at the top of
            # run_context (shared with Step 7's concordance-UMAP correction).
            # Step 5 applies that correction to the mean-accessibility matrix
            # that seeds the RWR graph (harmony harmonizes over all
            # comma-separated --batch-column variables jointly; the other
            # methods use the composite).

            # Resolve signed-method auto-defaults.  These ship as a coherent
            # bundle: MLE keeps RWR's cell-type-aware smoothing intact (vs DM
            # double-shrinkage), the post-norm refinement basis strips per-cell
            # shifts from the refined k-NN graph, and one refine pass is needed
            # for MLE to lift cell-type onto the dominant PC0.
            # DM is the signed default.  A full-set comparison (kNN cell-type
            # accuracy on ALL cells, claimed labels) showed DM beats MLE:
            # DM+skip1 = 0.937 and DM+harmony+skip0 = 0.940, vs MLE's best
            # 0.930 (and MLE collapses to ~0.80 without harmony).  The earlier
            # MLE win was an artifact of validating on the high-coverage
            # labeled subset, where PC0 is always cell type; on the full set
            # the low-coverage cells put a confound on PC0 for MLE too.
            concordance_formula = (
                "dm" if args.concordance_formula == "auto"
                else args.concordance_formula
            )
            concordance_combine = _resolve_concordance_combine(args)
            refinement_basis = (
                "postnorm" if args.refinement_basis == "auto"
                else args.refinement_basis
            )
            # Resolve --rwr-refine-passes: 'auto' (or unset, for signed) →
            # eigengap auto-stop; an integer → fixed count.
            rp = args.rwr_refine_passes
            if rp is None:
                rp = "auto" if args.concordance_method == "signed" else "0"
            if str(rp).lower() == "auto":
                auto_select_passes = True
                rwr_refine_passes = 0  # ignored when auto_select_passes=True
                passes_desc = f"auto (≤{args.rwr_max_auto_passes})"
            else:
                try:
                    rwr_refine_passes = int(rp)
                except ValueError:
                    raise ValueError(
                        f"--rwr-refine-passes must be an integer or 'auto'; "
                        f"got {rp!r}."
                    )
                auto_select_passes = False
                passes_desc = str(rwr_refine_passes)
            logger.info(
                f"Signed estimator: formula={concordance_formula}, "
                f"refinement_basis={refinement_basis}, "
                f"refine_passes={passes_desc}"
            )

            sig_result = concordance_mod.compute_signed_concordance_matrix(
                posterior_paths, chrom_sizes,
                bin_size=args.bin_size,
                min_distance=args.min_distance,
                max_distance=args.max_distance,
                min_gch_sites=min_sites,
                min_pairs_per_bin=args.min_pairs_per_bin,
                min_cell_pairs=args.min_cell_pairs,
                min_bin_coverage=args.min_bin_coverage,
                statistic=args.concordance_statistic or "signed",
                concordance_formula=concordance_formula,
                baseline=args.baseline,
                concordance_combine=concordance_combine,
                dm_window_mb=args.dm_window_mb,
                dm_dist_bin=args.dm_dist_bin_bp,
                dm_prior=args.dm_prior,
                emit_null=args.emit_null,
                use_rwr=(args.impute == "rwr"),
                rwr_k=args.rwr_k,
                rwr_restart_prob=args.rwr_restart_prob,
                rwr_n_refine_passes=rwr_refine_passes,
                rwr_refinement_basis=refinement_basis,
                rwr_auto_select_passes=auto_select_passes,
                rwr_max_auto_passes=args.rwr_max_auto_passes,
                rwr_solver=args.rwr_solver,
                rwr_max_iter=args.rwr_max_iter,
                rwr_tol=args.rwr_tol,
                eb_min_obs=args.eb_min_obs,
                batch_labels=batch_labels,
                batch_correction=batch_correction,
                n_threads=args.n_threads,
                output_dir=str(concordance_dir),
                save_outputs=True,
                random_state=args.random_seed,
            )
            if args.step5_batch_sweep:
                # Reuses this run's steps 1-3 via `precomputed`, so each swept
                # setting costs only RWR + Layer 2 rather than a full Step 5.
                _s5_kwargs = dict(
                    bin_size=args.bin_size, min_distance=args.min_distance,
                    max_distance=args.max_distance, min_gch_sites=min_sites,
                    min_pairs_per_bin=args.min_pairs_per_bin,
                    min_cell_pairs=args.min_cell_pairs,
                    min_bin_coverage=args.min_bin_coverage,
                    statistic=args.concordance_statistic or "signed",
                    concordance_formula=concordance_formula,
                    baseline=args.baseline,
                    concordance_combine=concordance_combine,
                    dm_window_mb=args.dm_window_mb,
                    dm_dist_bin=args.dm_dist_bin_bp,
                dm_prior=args.dm_prior,
                emit_null=args.emit_null,
                    use_rwr=(args.impute == "rwr"), rwr_k=args.rwr_k,
                    rwr_restart_prob=args.rwr_restart_prob,
                    rwr_n_refine_passes=rwr_refine_passes,
                    rwr_refinement_basis=refinement_basis,
                    rwr_auto_select_passes=auto_select_passes,
                    rwr_max_auto_passes=args.rwr_max_auto_passes,
                    rwr_solver=args.rwr_solver, rwr_max_iter=args.rwr_max_iter,
                    rwr_tol=args.rwr_tol, eb_min_obs=args.eb_min_obs,
                    n_threads=args.n_threads, output_dir=str(concordance_dir),
                    random_state=args.random_seed)
                _step5_batch_sweep(
                    sig_result, posterior_paths, chrom_sizes, _s5_kwargs,
                    samples, args,
                    samples.set_index("cell_name")["group"].dropna(),
                    str(concordance_dir))
            # Headline matrix for Step 6/7 is whichever statistic was selected.
            stat = args.concordance_statistic or "signed"
            if stat == "open" and sig_result["delta_open"] is not None:
                concordance_raw = sig_result["delta_open"]
            else:
                concordance_raw = sig_result["delta_signed"]
            pair_counts = sig_result["pair_counts"]
            # Also save a "concordance_raw.parquet" symlink/copy for
            # downstream-step compatibility (Step 6/7 looks for this name).
            concordance_raw.to_parquet(
                concordance_dir / "concordance_raw.parquet"
            )
        else:
            raise ValueError(
                f"Unknown --concordance-method: {args.concordance_method!r}"
            )

        logger.info("Computing concordance vs. distance decay...")
        distance_analysis.concordance_vs_distance(
            posterior_paths,
            min_gch_sites=min_sites,
            n_threads=args.n_threads,
            output_dir=str(concordance_dir),
        )
    else:
        # Only load step 5 output if a downstream step (6 or 7) needs it.
        if not ({6, 7} & steps):
            logger.info("Step 5: Skipped (no downstream step requires it)")
            return
        raw_path = concordance_dir / "concordance_raw.parquet"
        cnt_path = concordance_dir / "pair_counts_per_bin.parquet"
        if not raw_path.exists() or not cnt_path.exists():
            raise FileNotFoundError(
                f"Concordance matrix not found at {raw_path}; re-run step 5."
            )
        logger.info(f"Step 5: Skipped (loading concordance matrix from {concordance_dir})")
        concordance_raw = pd.read_parquet(raw_path)
        pair_counts = pd.read_parquet(cnt_path)

    # ---- Step 6: Shrinkage -----------------------------------------------
    # For betabinom concordance, EB shrinkage is built into θ_L's Beta posterior.
    # The Stein-style Pearson-r shrinkage is only meaningful for the
    # `concordance_method=pearson` branch and would be inappropriate for Δ values.
    if 6 in steps:
        logger.info("=" * 60)
        logger.info(f"Step 6: Shrinkage analysis for {context}")
        logger.info("=" * 60)

        if args.concordance_method in ("betabinom", "signed"):
            logger.info(
                f"Step 6: Skipping Stein shrinkage — {args.concordance_method} "
                f"Δ values are already EB-shrunk via the Beta/Dirichlet prior."
            )
            concordance_shrunk = concordance_raw.copy()
            concordance_shrunk.to_parquet(concordance_dir / "concordance_shrunk.parquet")
            concordance_shrunk.to_csv(concordance_dir / "concordance_shrunk.tsv", sep="\t")
        else:
            if args.shrinkage_n0 == "auto":
                shrink_result = shrinkage.estimate_shrinkage_n0(
                    concordance_raw, pair_counts,
                    output_dir=str(concordance_dir),
                )
                n0 = shrink_result["n0"]
            else:
                n0 = float(args.shrinkage_n0)
                logger.info(f"Using user-supplied shrinkage n0 = {n0}")
                with open(concordance_dir / "shrinkage_n0.json", "w") as f:
                    json.dump({"n0": n0, "source": "user"}, f, indent=2)

            concordance_shrunk = shrinkage.apply_shrinkage(concordance_raw, pair_counts, n0)
            concordance_shrunk.to_parquet(concordance_dir / "concordance_shrunk.parquet")
            concordance_shrunk.to_csv(concordance_dir / "concordance_shrunk.tsv", sep="\t")
    else:
        # Only load step 6 output if step 7 needs it.
        if 7 not in steps:
            logger.info("Step 6: Skipped (no downstream step requires it)")
            return
        shrunk_path = concordance_dir / "concordance_shrunk.parquet"
        if not shrunk_path.exists():
            raise FileNotFoundError(
                f"Shrunk concordance not found at {shrunk_path}; re-run step 6."
            )
        logger.info(f"Step 6: Skipped (loading shrunk concordance from {shrunk_path})")
        concordance_shrunk = pd.read_parquet(shrunk_path)

    # ---- Step 7: Matrix ops + UMAP ---------------------------------------
    if 7 in steps:
        logger.info("=" * 60)
        logger.info(f"Step 7: Matrix processing + UMAP for {context}")
        logger.info("=" * 60)

        if concordance_shrunk.shape[0] == 0 or concordance_shrunk.shape[1] == 0:
            logger.warning(
                "Concordance matrix is empty (0 cells or 0 bins). "
                "This usually means no read pairs passed the filters "
                "(--min-distance, --min-gch-sites / --min-hcg-sites). "
                "Check that the context "
                "BED positions match the reference used for alignment, and "
                "that --context-bed-offset is correct. Skipping matrix "
                "processing and UMAP."
            )
            return

        _s7 = _resolve_step7_params(args)
        per_cell_norm = _s7["per_cell_norm"]
        per_cell_center = _s7["per_cell_center"]
        post_norm_recenter = _s7["post_norm_recenter"]
        matrix_norm = _s7["matrix_norm"]
        skip_top_svd = _s7["skip_top_svd"]
        n_svd = _s7["n_svd"]

        # Correct the concordance values themselves, before any normalization.
        cbc = args.concordance_batch_correction
        if cbc != "none":
            if concordance_batch_labels is None:
                logger.warning(
                    "--concordance-batch-correction needs --batch-column (or "
                    "--concordance-batch-column); skipping.")
                cbc = "none"
            else:
                concordance_shrunk = _correct_concordance_batch(
                    concordance_shrunk, concordance_batch_labels, cbc,
                    args.random_seed, "concordance")
                concordance_shrunk.to_parquet(
                    concordance_dir / "concordance_batchcorrected.parquet")

        matrices = matrix_ops.process_concordance_matrix(
            concordance_shrunk,
            min_cell_coverage=args.min_cell_coverage,
            output_dir=str(concordance_dir),
            matrix_norm=matrix_norm,
            per_cell_center=per_cell_center,
            per_cell_norm=per_cell_norm,
            post_norm_recenter=post_norm_recenter,
        )

        group_labels = samples.set_index("cell_name")["group"]
        group_colors = group_colors_from_samples(samples, args.color_column)
        group_order = ([s.strip() for s in args.group_order.split(",") if s.strip()]
                       if args.group_order else None)

        if matrices["imputed"].shape[1] == 0:
            logger.warning(
                "No bins retained after coverage filter; skipping UMAP."
            )
        elif matrices["imputed"].shape[0] < 2:
            logger.warning(
                "Fewer than 2 cells in the imputed matrix; skipping UMAP."
            )
        else:
            # Filename/title reflect the actual normalization used.
            norm_label = {
                "zscore": ("z-scored", "umap_zscore"),
                "quantile": ("quantile-normed", "umap_quantile"),
            }.get(matrix_norm, (matrix_norm, f"umap_{matrix_norm}"))
            norm_title, norm_prefix = norm_label

            # SVD scree-plot diagnostic on the normalized matrix.
            # Always produced so the user can pick --n-svd from the elbow.
            visualization.run_svd_scree(
                matrices["zscore"],
                max_components=args.svd_scree_max,
                output_prefix=f"svd_scree_{matrix_norm}",
                output_dir=str(umap_dir),
                random_state=args.random_seed,
                plot_format=args.plot_format,
            )

            svd_suffix = f"_svd{n_svd}" if n_svd and n_svd > 0 else ""
            if skip_top_svd and skip_top_svd > 0:
                svd_suffix += f"_skip{skip_top_svd}"
            svd_title = (
                f" + SVD-{n_svd}" if n_svd and n_svd > 0 else ""
            )
            if skip_top_svd and skip_top_svd > 0:
                svd_title += f" (skip top {skip_top_svd})"

            # Step 7 concordance-UMAP batch correction (in addition to the
            # Step 5 accessibility/RWR-basis correction).  Applied to the
            # NORMALIZED matrix only -- it is THE analysis UMAP.  The
            # un-normalized UMAP is left uncorrected on purpose, so it stays an
            # honest raw-structure (incl. raw batch) diagnostic that its
            # "un-normalized" title truthfully describes.
            bc_suffix = (f"_{batch_correction}" if batch_correction != "none"
                         else "")
            bc_title = (f" + {batch_correction}" if batch_correction != "none"
                        else "")
            # The concordance-level correction goes in the name and the title
            # too: otherwise a corrected run silently overwrites an
            # uncorrected one and the two figures are indistinguishable.
            if cbc != "none":
                bc_suffix += f"_cbc-{cbc}"
                bc_title += f" + concordance-{cbc}"

            visualization.run_umap_and_plot(
                matrices["imputed"], group_labels,
                title=f"UMAP (un-normalized{svd_title}, {context})",
                output_prefix=f"umap_unnormalized{svd_suffix}",
                output_dir=str(umap_dir),
                n_svd=n_svd,
                skip_top_svd=skip_top_svd,
                n_threads=args.n_threads,
                random_state=args.random_seed,
                group_colors=group_colors,
                group_order=group_order,
                plot_format=args.plot_format,
            )
            visualization.run_umap_and_plot(
                matrices["zscore"], group_labels,
                title=f"UMAP ({norm_title}{svd_title}{bc_title}, {context})",
                output_prefix=f"{norm_prefix}{svd_suffix}{bc_suffix}",
                output_dir=str(umap_dir),
                n_svd=n_svd,
                skip_top_svd=skip_top_svd,
                n_threads=args.n_threads,
                random_state=args.random_seed,
                group_colors=group_colors,
                group_order=group_order,
                batch_labels=batch_labels,
                batch_correction=batch_correction,
                harmony_theta=args.harmony_theta,
                plot_format=args.plot_format,
            )

            # ---- Companion UMAPs on the concordance components -----------
            # coordination (C) and marginal (p̄) are the two INDEPENDENT
            # degrees of freedom of the co-state distribution; direction (F)
            # and directional (D = C·(2F−1)) are derived from them but answer
            # a different question -- F is *which way* a bin's concordance
            # points (co-open share), independent of how strong it is, while
            # C is the strength independent of the direction.  All four are
            # embedded so it is visible which one carries the cell-type
            # structure.  D gets a stable filename here regardless of which
            # statistic is the headline Δ (it duplicates the main UMAP only
            # when --concordance-combine is 'concordant-signed').  Same
            # normalization / SVD / batch treatment as the main UMAP so they
            # are all directly comparable.
            if args.component_umaps == "on" or (
                args.component_umaps == "auto"
                and _resolve_concordance_combine(args) == "concordant-signed"
            ):
                # polar_R is here; polar_theta and polar_p are NOT, and
                # both omissions are deliberate.
                #
                #   polar_p = exp(-R^2/2) is a deterministic monotone
                #     function of R, so it carries nothing R does not, and it
                #     is crushed toward zero for everything interesting --
                #     the distances would be set by the null bins.
                #   polar_theta is CIRCULAR.  Euclidean distance on degrees
                #     calls +179 and -179 the far ends of the axis when they
                #     are two degrees apart, and that tear runs through the
                #     anti-coordinated sector.  Embed the sine and cosine
                #     TOGETHER (concordance_polar_theta_{sin,cos}.parquet)
                #     if a direction embedding is wanted, never the degrees;
                #     for colouring an existing UMAP, use a cyclic colormap.
                for comp, comp_title in (
                    ("directional_corrected",
                     "directional D' = C·(F−F_exp) (marginal-free)"),
                    ("plane",
                     "coordination plane (C and F−F_exp jointly)"),
                    ("polar_R",
                     "polar R (joint departure from the null)"),
                    ("coordination", "coordination C (marginal-free)"),
                    ("direction_excess",
                     "direction excess F−F_exp (co-open vs null)"),
                    # The same direction on the log-odds scale, where the
                    # variance is free of F so bins of different depth and
                    # different marginal are comparable.  Its own UMAP, not
                    # a second axis beside direction_excess: the two are
                    # monotone in the same F (see _decompose).
                    ("direction_logor",
                     "direction log-OR logit(F)−logit(F_exp)"),
                    ("directional", "directional D = C·(2F−1) (legacy)"),
                    ("direction", "direction F (co-open share)"),
                    ("marginal", "marginal p̄ (mean openness)"),
                ):
                    comp_path = concordance_dir / f"concordance_{comp}.parquet"
                    if not comp_path.exists():
                        logger.warning(
                            f"Component UMAP: {comp_path.name} not found "
                            f"(re-run Step 5 to emit it); skipping."
                        )
                        continue
                    # Components get the same per-bin correction as the
                    # headline matrix, or their UMAPs would answer a different
                    # question from it.
                    comp_raw = pd.read_parquet(comp_path).reindex(
                        index=concordance_shrunk.index)
                    if cbc != "none":
                        comp_raw = _correct_concordance_batch(
                            comp_raw, concordance_batch_labels, cbc,
                            args.random_seed, comp)
                    # Its own subdirectory: process_concordance_matrix always
                    # writes concordance_imputed / concordance_<norm>, which
                    # would otherwise clobber the headline matrices.
                    comp_mat = matrix_ops.process_concordance_matrix(
                        comp_raw,
                        min_cell_coverage=args.min_cell_coverage,
                        output_dir=str(concordance_dir / "components" / comp),
                        matrix_norm=matrix_norm,
                        per_cell_center=per_cell_center,
                        per_cell_norm=per_cell_norm,
                        post_norm_recenter=post_norm_recenter,
                    )
                    if comp_mat["zscore"].shape[1] == 0 or \
                            comp_mat["zscore"].shape[0] < 2:
                        logger.warning(
                            f"Component UMAP ({comp}): too few cells/bins "
                            f"after the coverage filter; skipping."
                        )
                        continue
                    visualization.run_umap_and_plot(
                        comp_mat["zscore"], group_labels,
                        title=(f"UMAP ({comp_title}, {norm_title}"
                               f"{svd_title}{bc_title}, {context})"),
                        output_prefix=(f"umap_{comp}_{matrix_norm}"
                                       f"{svd_suffix}{bc_suffix}"),
                        output_dir=str(umap_dir),
                        n_svd=n_svd,
                        skip_top_svd=skip_top_svd,
                        n_threads=args.n_threads,
                        random_state=args.random_seed,
                        group_colors=group_colors,
                        group_order=group_order,
                        batch_labels=batch_labels,
                        batch_correction=batch_correction,
                        harmony_theta=args.harmony_theta,
                        plot_format=args.plot_format,
                    )
    else:
        logger.info("Step 7: Skipped.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ---- Reconcile --concordance-method and --concordance-statistic ------
    # Legacy --concordance-method betabinom aliases to the new
    # --concordance-statistic open with a deprecation warning (design-doc
    # §6 #10).  --concordance-method signed (the new default) pairs with
    # --concordance-statistic ('signed', or 'both' if user opts in).
    if args.concordance_statistic is None:
        if args.concordance_method == "betabinom":
            args.concordance_statistic = "open"
        elif args.concordance_method == "signed":
            args.concordance_statistic = "signed"
        else:
            args.concordance_statistic = None  # pearson: not applicable

    if args.concordance_method == "betabinom":
        # Don't yell yet, but let the user know.
        pass  # informational logger message goes after setup_logging below.

    # Default --min-pairs-per-bin depends on the method (design-doc §6 #15).
    if args.min_pairs_per_bin is None:
        if args.concordance_method == "pearson":
            args.min_pairs_per_bin = 5
        else:  # signed or betabinom — both use EB shrinkage that handles n=1
            args.min_pairs_per_bin = 1

    out = ensure_dir(args.output_dir)
    log_file = args.log_file
    if log_file is None:
        log_file = str(out / "pipeline.log")
    setup_logging(logging.INFO, log_file=log_file)

    if args.concordance_method == "betabinom":
        logger.warning(
            "--concordance-method betabinom is the legacy one-sided path. "
            "The new default is --concordance-method signed (Dirichlet-"
            "Multinomial directional Δ with pre-Layer-2 RWR imputation). "
            "betabinom remains supported for A/B comparison."
        )

    # Save full config up front.
    config = vars(args).copy()
    with open(out / "pipeline_config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    steps = _parse_steps(args.steps)
    logger.info(f"Running steps: {sorted(steps)}")

    samples = load_samples(args.samples)

    # Resolve contexts: --context-bed overrides --contexts when provided.
    context_bed_paths: dict[str, str] | None = None
    if args.context_bed:
        context_bed_paths = _parse_context_bed(args.context_bed)
        contexts = list(context_bed_paths.keys())
        logger.info(
            f"Using pre-built BED files for contexts: "
            + ", ".join(f"{c}={p}" for c, p in context_bed_paths.items())
        )
    else:
        contexts = [c.strip() for c in args.contexts.split(",") if c.strip()]
        # Without --context-bed the .2bit reference is required for extraction.
        if args.reference is None:
            raise ValueError(
                "--reference is required when not using --context-bed."
            )
    if not contexts:
        raise ValueError("--contexts (or --context-bed) produced an empty list.")

    # ---- Step 1: context positions ----------------------------------------
    if context_bed_paths is not None:
        # Fast path: load from tabix-indexed BED files.
        # Always runs (context positions are needed by all downstream steps).
        logger.info("Loading context positions from BED files...")
        all_context_positions = reference.load_context_from_bed(
            context_bed_paths,
            n_threads=args.n_threads,
            c_offset=args.context_bed_offset,
        )
    elif 1 in steps:
        logger.info("=" * 60)
        logger.info("Step 1: Extracting context positions from reference")
        logger.info("=" * 60)
        all_context_positions = reference.extract_context_positions(
            args.reference,
            str(out / "01_context_extraction"),
            contexts=contexts,
            dbsnp_vcf=args.dbsnp_vcf,
            n_threads=args.n_threads,
            extra_chroms=parse_include_chroms(
                getattr(args, "include_chrom", None)),
        )
    else:
        logger.info("Step 1: Skipped (loading precomputed context positions)")
        all_context_positions = reference.load_context_positions(
            str(out / "01_context_extraction"), contexts
        )

    # Chromosome sizes — used for bin sorting in Step 5.
    if args.reference is not None:
        chrom_sizes = _get_chrom_sizes(args.reference)
    else:
        # Fall back to the first BAM file's @SQ header.
        first_bam = samples["bam_path"].iloc[0]
        logger.info(f"Reading chromosome sizes from BAM header: {first_bam}")
        chrom_sizes = _get_chrom_sizes_from_bam(first_bam)

    # Applied to chrom_sizes AND the context tables, so an excluded contig
    # cannot re-enter through either the binning or the site lists.
    excluded = parse_exclude_chroms(getattr(args, "exclude_chrom", None))
    if excluded:
        keep = set(report_excluded(excluded, list(chrom_sizes), "chromosome"))
        chrom_sizes = {c: n for c, n in chrom_sizes.items() if c in keep}
        all_context_positions = {
            ctx: {c: a for c, a in per.items()
                  if not chrom_excluded(c, excluded)}
            for ctx, per in all_context_positions.items()}

    # ---- Per-context processing (steps 2..7) -----------------------------
    for context in contexts:
        ctx_positions = all_context_positions.get(context, {})
        ctx_out = out / f"context_{context}"
        run_context(
            context=context,
            context_positions=ctx_positions,
            samples=samples,
            ctx_out=ctx_out,
            steps=steps,
            args=args,
            chrom_sizes=chrom_sizes,
        )

    logger.info("Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
