# Signed concordance + RWR imputation + comparison framework

## Design document v0.2 — locked decisions

> **Status:** §6 open questions are resolved (decisions listed in §6 below).
> Layer 2 model is Dirichlet-Multinomial, with U-3 (soft expected counts) for
> both linked and unlinked end handling. Implementation can proceed in the
> phase order in §5.

This document covers three coordinated changes to the `nome_hic_concordance` pipeline:

1. **Signed concordance Δ** — a new per-(cell, bin) statistic that captures coordination *with direction* (active vs silenced), in addition to keeping the existing one-sided Δ as a supported option.
2. **RWR imputation** — an optional denoising step inspired by scHiCluster that uses a k-NN graph across cells to smooth the per-(cell, bin) concordance values, applied to whichever Δ variant the user chose.
3. **Three-way comparison framework** — automated comparison of concordance (Δ_open or Δ_signed) against (a) bin-mean GCH accessibility and (b) per-cell compartment scores from scHiCluster.

The three pieces are designed to land together because they share infrastructure (matrix-level imputation, the comparison harness) but each can be toggled independently.

---

## 1. Motivation

### 1.1 Why a signed concordance statistic

The current Δ = E[θ_L | data] − p̂² measures the excess of **both-open** pairs over chance. It treats:

- "many co-open pairs" → high positive Δ ✓
- "many co-closed pairs" → Δ ≈ 0 (because both θ_L and p̂² shrink together)
- "random mixing"      → Δ ≈ 0 ✓

This means a heterochromatic bin with coordinated *closure* looks identical to a chaotic bin under the current statistic, which loses real cell-type information.

The proposed `Δ_signed` resolves this by carrying direction:

```
S = (k_open − k_closed) / n_linked          # signed per-pair score in [-1, +1]
Δ_signed = S − (2 p̂ − 1)                   # excess over independence baseline
```

Behaviour:

| Bin biology                          | Δ_signed |
|--------------------------------------|----------|
| Coordinated open  (active gene)      | high positive |
| Coordinated closed (heterochromatin) | high negative |
| Random / mixed                       | ≈ 0 |
| Anti-correlated ends                 | < 0 (rare) |

This single-number-with-sign formulation captures the cell-type-relevant biology more directly than the one-sided variant.

### 1.2 Why keep the one-sided Δ

For methods comparison (e.g. an upcoming paper), being able to A/B test the two definitions on the same dataset is valuable. The two should give correlated but distinguishable results, and the one-sided form has interpretability advantages in some contexts (it's literally the "co-open excess").

Decision: support **both**, selectable via a CLI flag. Default will be the signed form once we've validated it.

### 1.3 Why RWR imputation

Single-cell concordance is sparsity-limited. For 250 kb bins on typical NOMe-HiC data, many (cell, bin) entries have n_linked < min_pairs_per_bin and end up as NaN, and even non-NaN entries can be noisy because k_open / n_linked is estimated from a small denominator.

The empirical-Bayes prior already shrinks individual entries toward the population mean. But it ignores **cell-cell structure**: a cell's bins should be informed by what similar cells look like at those bins.

scHiCluster solved this for raw Hi-C contact matrices by building a k-NN graph across cells and propagating signal via random-walk-with-restart (RWR). The same idea applies to our cells × bins concordance matrix — and arguably *better* than for raw contact matrices, since our cells × bins entries are already partially aggregated (per bin, per cell) so the k-NN graph is operating on a more interpretable feature space.

The integration goal: an optional step in our pipeline that takes the Δ matrix (with NaNs) and emits a denoised matrix, with imputation parameters and diagnostics under user control.

---

## 2. Part A — Signed concordance statistic

### 2.0 Two layers of probabilistic modeling

To avoid the naming overlap that's been a source of confusion:

- **Layer 1 — per-end open/closed call** (Steps 3 and 4, already implemented). A
  two-component beta-binomial mixture is fitted to the genome-wide GCH counts,
  and each read end's GCH counts are turned into a posterior probability
  `p_i = P(end is open | counts)`. **This stays exactly as today.**
- **Layer 2 — per-(cell, bin) joint-state shrinkage** (Step 5, new). The
  per-pair joint state is modeled as a multinomial across three categories
  `(both-open, both-closed, mixed)`, with a **Dirichlet-Multinomial** EB prior
  fit by method-of-moments across all (cell, bin) observations. **This is the
  new piece introduced here.**

The two layers serve orthogonal purposes: Layer 1 denoises the per-end call
from a noisy "how many of my N GCHs got methylated" → calibrated `p_i`;
Layer 2 denoises the per-(cell, bin) joint-state distribution from sparse
`n_linked` per cell per bin → calibrated `(π_open, π_closed, π_mixed)`.

### 2.1 Per-pair classification — U-3 soft expected counts

We **do not threshold** the per-end posteriors before forming the joint counts.
Instead, every pair contributes its full posterior probabilities to all three
joint outcomes:

For each linked pair `(i, j)` (both ends in same bin, distance ≥ min_distance,
both informative i.e. `n_GCH ≥ min_gch_sites`):

```
expected contribution to k_open    = p_i · p_j
expected contribution to k_closed  = (1 − p_i) · (1 − p_j)
expected contribution to k_mixed   = p_i · (1 − p_j) + (1 − p_i) · p_j
```

Each pair contributes exactly 1 to `n_linked` and a vector summing to 1 to
`(k_open, k_closed, k_mixed)`. No pair is ever dropped on uncertainty grounds:
ends with mid-range posteriors split their weight across multiple outcomes,
which is the most information-preserving treatment we can give them.

**Rationale:** ends with mid-range `p_i` typically reflect low per-end GCH
coverage; thresholding them is forced to make a confident open/closed call
that the data doesn't actually support. Soft counts let Layer 2 see the
underlying uncertainty and weight it appropriately during shrinkage and
in the variance term of `Δ_signed`.

Dual posterior thresholds (`--posterior-threshold-open` default 0.6,
`--posterior-threshold-closed` default 0.4) are still exposed but used only
for **diagnostic** per-end classification (e.g. for QC plots and the
betabinom-counts long-form table). They do not affect the soft counts that
feed into Layer 2.

### 2.2 Per-(cell, bin) counts

For each linked pair `(i, j)` in a bin, soft expected counts (floats):

```
k_open    = Σ over linked pairs of  p_i · p_j
k_closed  = Σ over linked pairs of  (1 − p_i)(1 − p_j)
k_mixed   = Σ over linked pairs of  p_i(1 − p_j) + (1 − p_i)p_j
n_linked  = number of linked pairs (integer)
```

By construction, `k_open + k_closed + k_mixed = n_linked`.

For the unlinked baseline, the same soft treatment is applied — informative
unlinked ends are not thresholded:

```
k_unlinked_open    = Σ over informative unlinked ends of  p_i             (float)
k_unlinked_closed  = Σ over informative unlinked ends of  (1 − p_i)       (float)
n_unlinked         = number of informative unlinked ends                  (integer)
```

By construction, `k_unlinked_open + k_unlinked_closed = n_unlinked`.

The unlinked open-rate `p̂ = k_unlinked_open / n_unlinked` is now the **mean
posterior across all informative unlinked ends** in the bin, with no
thresholding. This is symmetric with the linked side: both numerator
(`k_open`, …) and denominator-baseline (`p̂`) carry the same calibrated per-end
information forward into Layer 2.

### 2.3 Two Δ statistics

**One-sided (existing form, kept as `--concordance-statistic open`).**

Reuses the current Beta-Binomial EB pipeline, but applied to the *soft*
`k_open` (a float in `[0, n_linked]`) instead of the hard count:

```
θ_open = E[π_open | data]                       # Beta-Binomial posterior on (k_open, n_linked)
Δ_open = θ_open − p̂²
Var(Δ_open) ≈ Var(θ_open) + 4 p̂² · p̂(1 − p̂) / n_unlinked
```

The method-of-moments fit for the Beta-Binomial works unchanged on soft
counts — `rate = k_open / n_linked` is now a float in `[0, 1]`, and the
moment expressions are identical. (The likelihood is no longer an exact
Beta-Binomial since `k_open` isn't a sum of Bernoullis; we're using MoM as
an EB shrinkage tool here, not as a full generative likelihood. Acceptable
trade-off for empirical Bayes.)

**Signed (new, default — `--concordance-statistic signed`).**

We model the per-pair joint outcome as multinomial across three categories
with a **Dirichlet-Multinomial** EB prior:

```
(π_open, π_closed, π_mixed) ~ Dirichlet(α_open, α_closed, α_mixed)
(k_open, k_closed, k_mixed | π) ~ Multinomial(n_linked, π)
```

Empirical-Bayes prior fit by method-of-moments on pooled (cell, bin)
observations. Sketch of the MoM:

```
μ̂_i  = (Σ_cb k_i,cb) / (Σ_cb n_linked,cb)              for i ∈ {open, closed, mixed}
s²_i  = weighted variance across (cell, bin) of  k_i,cb / n_linked,cb
n̄    = weighted mean of  n_linked,cb

ν̂    = (μ̂_i (1 − μ̂_i) − s²_i) / (s²_i − μ̂_i (1 − μ̂_i) / n̄)
       (pick category with most signal, e.g. `open`; or average across i)

α_i  = μ̂_i · ν̂        for i ∈ {open, closed, mixed}
α_total = Σ_i α_i = ν̂
```

Posterior means:

```
E[π_i | data] = (α_i + k_i) / (α_total + n_linked)              for i ∈ {open, closed, mixed}
```

Signed concordance:

```
S        = E[π_open | data] − E[π_closed | data]
baseline = p̂² − (1 − p̂)² = 2 p̂ − 1
Δ_signed = S − (2 p̂ − 1)
```

Variance via Dirichlet-Multinomial covariance formulas (with `μ_i = E[π_i | data]`):

```
Var(π_i)      = μ_i (1 − μ_i) / (α_total + n_linked + 1)
Cov(π_i, π_j) = −μ_i μ_j / (α_total + n_linked + 1)               for i ≠ j

Var(S) = Var(π_open) + Var(π_closed) − 2 Cov(π_open, π_closed)
       = [μ_open + μ_closed − (μ_open − μ_closed)²] / (α_total + n_linked + 1)

Var(2 p̂ − 1) = 4 · p̂(1 − p̂) / n_unlinked                        # delta-method

Var(Δ_signed) = Var(S) + Var(2 p̂ − 1)
```

Behavioural sanity-check on the examples we worked through earlier:

| Bin biology | p̂ | π_open | π_closed | S | 2p̂−1 | Δ_signed |
|---|---|---|---|---|---|---|
| Active, coordinated open  | 0.6 | 0.50 | 0.10 | +0.40 | +0.20 | **+0.20** |
| Silenced, coordinated closed | 0.4 | 0.10 | 0.50 | −0.40 | −0.20 | **−0.20** |
| Random at p̂=0.5 (independence) | 0.5 | 0.25 | 0.25 |    0 |    0 |     **0** |
| Random at p̂=0.7 (independence) | 0.7 | 0.49 | 0.09 | +0.40 | +0.40 |     **0** |
| Focal active (p̂=0.5, but pairs cluster open) | 0.5 | 0.45 | 0.05 | +0.40 |    0 | **+0.40** |

### 2.4 CLI surface

New flag:

```
--concordance-statistic {open,signed}     default: signed
```

Replaces the existing implicit choice. When `open` is selected we follow the current Beta-Binomial path; when `signed` we follow the Dirichlet-Multinomial path. Both can be computed and saved in a single run via:

```
--concordance-statistic both
```

which is what we'll use for comparisons.

The matrix that gets handed off to the downstream `matrix_ops` step is whichever statistic was named in `--concordance-statistic`. If `both`, the headline matrix is `signed`; `open` is saved as a side-product for diagnostics.

### 2.5 Outputs

In `output/.../05_concordance/`:

```
concordance_open.parquet            # cells × bins, Δ_open
concordance_signed.parquet          # cells × bins, Δ_signed
concordance_signed_var.parquet      # cells × bins, Var(Δ_signed)
concordance_open_var.parquet        # cells × bins, Var(Δ_open)
betabinom_prior.json                # one-sided EB prior  (α, β, μ, ν)
dirichlet_multinomial_prior.json    # signed EB prior     (α_open, α_closed, α_mixed)
betabinom_counts.parquet            # long-form per-(cell, bin) counts:
                                    #   n_linked, k_open, k_closed, k_mixed,
                                    #   n_unlinked, k_unlinked_open, k_unlinked_closed,
                                    #   p_hat, theta_open, theta_closed,
                                    #   delta_open, delta_signed,
                                    #   delta_open_var, delta_signed_var
pair_counts_per_bin.parquet         # cells × bins, n_linked (unchanged)
```

The `concordance_raw.parquet` filename used by downstream Step 6/7 becomes a symlink or copy of whichever variant is the "headline" (so backward compatibility is preserved).

---

## 3. Part B — RWR imputation

### 3.1 Where it fits in the pipeline

A new step inserted between Step 5 (concordance) and Step 6 (Stein shrinkage, which is a no-op for the EB-shrunk betabinom path) / Step 7 (matrix ops + UMAP):

```
Step 5  concordance              → cells × bins Δ matrix (with NaNs)
   │
   ▼
Step 5.5  RWR imputation         ← NEW, optional
   │
   ▼
Step 6  Stein shrinkage          (skipped for betabinom)
   │
   ▼
Step 7  matrix_ops + UMAP        → final embedding
```

Optional via flag: `--impute rwr` (default: `none`, no imputation). Other values reserved for future variants (e.g. `knn`, `magic`, …).

### 3.2 Algorithm

Given input matrix `F_0 ∈ ℝ^{N × B}` (N cells, B bins; NaN-tolerant), produce smoothed matrix `F ∈ ℝ^{N × B}`.

**Step 1 — Pre-fill NaNs for distance computation.**
NaN entries are filled with the column mean (or 0 after centering — we'll use 0 after the QN step). Result: `F_filled`.

**Step 2 — Cell-cell similarity matrix.**
Two strategies, exposed via `--rwr-neighbor-feature {self, accessibility}`:

- `self` (default): cosine similarity between rows of `F_filled` — i.e. cells are similar if their concordance profiles agree.
- `accessibility`: cosine similarity on the bin-mean-accessibility matrix (which is much denser than the concordance matrix). Reduces the circularity of using the noisy signal to define neighbours of itself.

Build a k-NN graph: for each cell, keep the top `k` most-similar other cells. Symmetrize (`W = (W + W^T) / 2`), then row-normalize so each row sums to 1 → transition matrix `P`.

**Step 3 — RWR.**
Closed-form for N ≤ 5000:

```
F = r · (I − (1 − r) · P)^(−1) · F_0_filled
```

Iterative for larger N:

```
F ← F_0_filled
repeat: F ← (1 − r) · P @ F + r · F_0_filled
until ‖F − F_prev‖_∞ < tol  or  iter ≥ max_iter
```

**Step 4 — Restore NaNs (optional).**
By default, NaN entries in the original matrix get the *imputed* value (that's the whole point — we're filling them in). But for safety we record the original NaN mask alongside the imputed matrix, so downstream code can choose to mask them again if needed.

### 3.3 Iterative refinement (optional second pass)

After the first imputation, the matrix has no NaNs. We can re-build the cell-cell similarity on the imputed matrix itself (which is now cleaner than the NaN-filled-with-0 starting point) and re-run RWR. One refinement pass is usually sufficient; we'll expose:

```
--rwr-refine-passes  default: 0    (0 = no refinement, just one pass)
```

### 3.4 Parameters and defaults

| Parameter | Flag | Default | Notes |
|---|---|---|---|
| k for k-NN | `--rwr-k` | 15 | Standard for moderate cell counts; scale to ~√N for very large N |
| restart probability r | `--rwr-restart` | 0.5 | scHiCluster's default |
| neighbour feature | `--rwr-neighbor-feature` | `self` | or `accessibility` |
| max iterations | `--rwr-max-iter` | 30 | Only used in iterative mode |
| convergence tolerance | `--rwr-tol` | 1e-4 | Iterative mode |
| refinement passes | `--rwr-refine-passes` | 0 | 0–2 useful range |
| solver | `--rwr-solver` | `auto` | `direct` / `iterative` / `auto` |

### 3.5 Outputs

In `output/.../06_imputed/` (new dir, parallel to existing `06_umap`):

```
concordance_signed_imputed.parquet      # cells × bins, RWR-imputed
concordance_open_imputed.parquet        # cells × bins, if --concordance-statistic both
cell_knn_graph.npz                       # k-NN adjacency + similarities (sparse)
rwr_diagnostics.json                     # {N, B, k, r, n_iter_used, converged, ...}
rwr_pre_post_distance_histogram.png      # QC: distribution of cell-cell distances
                                         #     before vs after imputation
```

### 3.6 Validation diagnostics

To catch over-smoothing or graph degeneracy:

- **Imputation strength per cell:** mean change in cell vector magnitude after RWR. Histogram across cells.
- **Cell-cell distance distribution shift:** if all cells collapse to the centroid after imputation, distances → 0 (bad). Plot before/after.
- **k-NN graph connectivity:** number of connected components. Should be 1 for a healthy graph; multiple components mean some cells are isolated.
- **Optional held-out validation:** mask 10% of non-NaN entries before imputation, then check reconstruction MSE. Optional flag `--rwr-validate` to enable.

---

## 4. Part C — Three-way comparison framework

### 4.1 Inputs

Three feature matrices, all in cells × bins shape:

1. **`concordance_*.parquet`** — Δ_open or Δ_signed (optionally RWR-imputed).
2. **`accessibility_mean.parquet`** — new, computed as a free byproduct of Step 5: mean of `posterior` across all informative ends in the bin per cell.
3. **`compartment_scores.parquet`** — from a sibling command `nome_hic_compartments` wrapping scHiCluster (separate scope, depends only on the per-cell read_pair parquets from Step 2).

The compartment matrix is the most expensive to produce and lives in its own sibling command for the reasons discussed earlier (scHiCluster dependency footprint, separate parameter surface). The comparison step accepts it as an input path:

```
--compartment-matrix PATH   path to cells × bins compartment-score parquet
```

### 4.2 Analyses

For each pair (concordance vs accessibility, concordance vs compartment), and for the triple (concordance | accessibility + compartment):

**(a) Marginal UMAPs** — same `matrix_ops → SVD scree → UMAP` recipe applied to each matrix separately. Visual comparison + silhouette / k-NN-purity if `group` labels are present.

**(b) Per-bin variance-partition regression.** For each bin b:

```
Δ_c,b ~ α_b + β1_b · access_c,b + β2_b · compartment_c,b + ε_c,b
```

Output:
- `regression_R2_per_bin.parquet` — R² per bin, plus coefficients
- `regression_R2_hist.png` — distribution of R² across bins (the "redundant vs unique-info" diagnostic)
- `delta_residual_matrix.parquet` — the residual `ε` matrix

**(c) Residual UMAP** — `matrix_ops → SVD → UMAP` applied to the residual matrix. If the residual matrix still separates cell types, concordance carries information not in the baselines.

**(d) Cross-validated classifier comparison** — only if `samples.group` is present. Logistic regression with L2 penalty, 5-fold CV, computing AUC for cell-type prediction using:
- accessibility only
- compartment only
- concordance only
- all three

Output: `classifier_comparison.tsv` with mean AUC and SD per feature set.

### 4.3 CLI

A new sub-command — concretely a separate entry point `nome_hic_compare` so the comparison scope stays separate from the heavy concordance pipeline:

```
nome_hic_compare \
  --concordance-matrix output/.../06_imputed/concordance_signed_imputed.parquet \
  --accessibility-matrix output/.../05_concordance/accessibility_mean.parquet \
  --compartment-matrix output/compartments/compartment_scores.parquet \
  --samples samples.tsv \
  --output-dir output/.../08_comparison \
  --matrix-norm quantile \
  --n-svd 5
```

Each `--*-matrix` is optional. If only one is provided, only marginal analyses for that one run. The cross-method comparisons require at least two.

### 4.4 Outputs

In `output/.../08_comparison/`:

```
umap_concordance.png         umap_accessibility.png         umap_compartment.png
umap_residual.png            (residual of concordance after access + compartment)

regression_R2_per_bin.parquet         regression_R2_hist.png
delta_residual_matrix.parquet

classifier_comparison.tsv             classifier_roc_curves.png

summary.json                          # all the headline numbers in one place
```

---

## 5. Implementation phases

Proposed staging — each phase is independently mergeable and produces working artefacts:

### Phase 1 — Signed concordance (in-place changes to Step 5)
- Add Dirichlet-Multinomial EB prior fit alongside existing Beta-Binomial.
- Add `--concordance-statistic`, `--posterior-threshold-{open,closed}` flags.
- Save both matrices and the count diagnostics when `both` is selected.
- Tests: math correctness of the multinomial posterior, behaviour on the sanity-check table in §2.3.

### Phase 2 — Accessibility byproduct
- Compute and save `accessibility_mean.parquet` from Step 5's per-cell parquets.
- ~30 lines, no dependencies, no new flags.
- Tests: simple per-bin mean recovery.

### Phase 3 — RWR imputation step
- New module `nome_hic_concordance/imputation.py` with `rwr_impute(F, k, r, ...)`.
- New optional Step 5.5 in `cli.py` triggered by `--impute rwr`.
- Tests: synthetic matrix with known low-rank structure; verify RWR reduces reconstruction error vs the input.
- Diagnostic plots.

### Phase 4 — scHiCluster compartment sibling command
- New entry point `nome_hic_compartments` in `pyproject.toml`.
- Converter from per-cell parquet → `.scool`.
- Wrapper around `schicluster.compartment.compartment_call`.
- Optional dep: `pip install nome_hic_concordance[compartments]`.
- Out of scope of this doc otherwise — separate sub-design for the wrapper.

### Phase 5 — Comparison harness
- New entry point `nome_hic_compare` in `pyproject.toml`.
- Implements §4.2.
- Tests: end-to-end on synthetic matrices with known relationships.

Each phase is independently useful: Phase 1 alone lets users get the signed concordance; Phase 1+2 lets them compare concordance to accessibility; the full set lets them run the methods comparison.

---

## 6. Resolved decisions

All items below are now locked. Decisions in **bold**.

1. **Dual posterior thresholds.** Defaults `--posterior-threshold-open 0.6`,
   `--posterior-threshold-closed 0.4`. These are exposed for diagnostic
   per-end classification (QC plots, the long-form `betabinom_counts`
   table) but **do not affect the actual concordance computation**, which
   uses U-3 soft expected counts on the un-thresholded posteriors.

2. **`--concordance-statistic signed` is the new default.** `open` remains
   selectable for the existing one-sided behaviour and for A/B comparison.
   `both` writes both matrices in one run.

3. **Compartment computation lives in a sibling command**
   `nome_hic_compartments`, scoped separately from this design doc.
   scHiCluster + cooler become optional deps via
   `pip install nome_hic_concordance[compartments]`.

4. **RWR k-NN neighbour feature defaults to `self`** (the concordance
   matrix itself). `--rwr-neighbor-feature accessibility` documented as
   a fallback when concordance is extremely sparse.

5. **RWR refinement passes default to 0** (one-shot RWR).

6. **When `--concordance-statistic both`, RWR is applied to both Δ
   matrices using the same k-NN graph**, derived from the headline
   (signed) matrix. Keeps the comparison fair.

7. **Layer-2 model: Dirichlet-Multinomial (Option A).** See §2.3.

8. **Linked-pair handling: U-3 soft expected counts.** See §2.1–§2.2.
   No pair is dropped for uncertainty; every pair contributes its full
   posterior probabilities to all three joint outcomes.

9. **Unlinked-end handling: U-3 soft expected counts.** Symmetric with
   the linked side. `p̂` is the mean per-end posterior across informative
   unlinked ends in the bin.

10. **Backward compatibility.** `--concordance-method betabinom` is
    aliased to `--concordance-statistic open` with a deprecation warning.
    Existing pipelines keep working unchanged.

11. **Comparison harness emits both PNG and TSV** for every plot.

12. **Tests for the new EB priors.** Phase 1 adds MoM tests on synthetic
    Dirichlet-Multinomial data, mirroring the existing Beta-Binomial test.

13. **Output naming.** Imputed matrices use the `_imputed` suffix
    (`concordance_signed_imputed.parquet`), not `_rwr`, to leave room for
    other imputation methods (`--impute knn`, `--impute magic`, ...) in
    the future without renaming files.

---

## 7. Non-goals

To keep the scope tractable:

- **Compartment computation itself.** The scHiCluster wrapper is a sibling sub-design; this document only specifies the interface the comparison harness expects.
- **Cross-context (GCH + HCG) integration.** Either context runs independently through this pipeline as today.
- **Trans-chromosomal Hi-C contacts.** All concordance and contact-based features remain cis-only.
- **scHiCluster-style imputation of individual contact matrices.** We're imputing the cells × bins concordance matrix, not the raw contact matrices. Doing the latter would essentially be re-implementing scHiCluster from scratch.
- **GPU acceleration of RWR.** Pure NumPy for now; if performance becomes a bottleneck on large datasets we can add CuPy as an optional dep later.

---

## Review checklist

- [x] Math for Dirichlet-Multinomial posterior and Δ_signed variance reviewed
- [x] All items in §6 resolved
- [x] Output naming convention agreed (`_imputed` suffix)
- [x] Comparison harness analyses covered in §4.2
- [x] Phase ordering agreed

**Ready for Phase 1 implementation.**
