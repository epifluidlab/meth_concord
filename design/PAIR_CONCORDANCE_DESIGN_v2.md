# Pair-Concordance Extension to meth_concord — Design v2

**Status:** Draft v2 (supersedes v1, which was written before the actual `meth_concord` source was available)
**Scope:** Add a pair-of-regions, contact-aware concordance layer on top of the existing `meth_concord` pipeline. Implements the hierarchical beta-binomial model with linked-vs-unlinked decomposition, three parallel per-cell embeddings (E_contact, E_coord, E_full), component-contribution analysis, and matched-contact-frequency comparison.
**Relationship to existing code:** New subpackage `nome_hic_concordance/pair_concordance/`. Reuses Steps 1–4 of the existing pipeline (context extraction, BAM parsing, mixture model, per-read-end posteriors) **unmodified**. Replaces Step 5 onward with a pair-of-regions framework.
**Validation strategy:** GM12878+IMR90 mixed cells first (SNP-based ground truth), then mouse cortex tissue.

---

## 1. What's already there and what's new

### 1.1 Reused from existing meth_concord (unchanged)

| Step | Module | Output we consume |
|------|--------|-------------------|
| 1 | `reference.py` | GCH context positions per chromosome |
| 2 | `bam_parser.py` | Per-cell `read_pairs.parquet` — every row is one Hi-C ligated chimera with `(chrom, pos1, k1, n1, pos2, k2, n2, distance)` |
| 3 | `mixture_model.py` | Pooled two-component beta-binomial fit (`em_fit_long.json`) — defines what "accessible" means at the read-end level |
| 4 | `posterior.py` | Per-cell `{cell}.posteriors.parquet` — adds `posterior1`, `posterior2` columns (P(accessible | k, n) per read end) |

**Critical observation:** every row in `posteriors.parquet` is **already a linked Hi-C read pair**. The existing pipeline doesn't distinguish "linked" vs "unlinked" because *all* of its observations are linked.

### 1.2 What the existing Step 5 does, and why we need to change it

Existing `concordance.py` (Step 5) bins each pair by `midpoint = (pos1+pos2)/2` into 5 Mb genomic bins, then computes per-(cell, bin) Pearson r of `(posterior1, posterior2)` across all pairs in the bin. This is:

- A **population-of-pairs** statistic aggregated per cell per genomic bin.
- It does NOT define a per-(pair-of-regions, cell) coordination quantity.
- It does NOT distinguish Step 1 (contact frequency / 3D folding) from Step 2 (functional coordination given a contact).
- It does NOT compute an independence baseline against which contact-conditional concordance can be measured.

For the Fig 4 Panel C and TF-folding-coordination story to work, we need a per-(pair-of-regions, cell) framework with explicit Step 1 / Step 2 decomposition. This is what the new subpackage provides.

### 1.3 What's new (this design)

A new subpackage `nome_hic_concordance/pair_concordance/` with these capabilities, layered on top of Step 4 outputs:

1. **Pair panel construction** — explicit pairs of small genomic regions (5–10 kb anchors) with category labels (co-folded, non-folded, TF-anchored, distance-matched random).
2. **Per-pair-per-cell linked vs unlinked aggregation** — for each pair (i, j) and each cell, count and characterize reads from `posteriors.parquet` whose `(pos1, pos2)` both land in `(i, j)` ("linked") vs reads where only one end lands in `i` or `j` ("unlinked-i" or "unlinked-j").
3. **Hierarchical model** — per-pair beta-binomial posterior on linked-concordance probability with empirical-Bayes priors pooled across pair categories; independence baseline from unlinked marginals; posterior coordination excess Δ with variance.
4. **Three parallel embeddings** — E_contact (Step 1 only), E_coord (Step 2 only via Δ), E_full (integrated).
5. **Component contribution analysis** — AUC per pair category per embedding, marginal contribution of E_coord over E_contact, matched-contact comparison.
6. **Validation pipeline** for GM12878+IMR90 with SNP-derived ground truth.
7. **Cortex application pipeline** with subtype-resolution analysis.

---

## 2. Module architecture

New subpackage parallel to existing modules, **no modifications to existing code**:

```
nome_hic_concordance/
├── __init__.py
├── cli.py                       # (existing - unchanged)
├── bam_parser.py                # (existing - unchanged)
├── concordance.py               # (existing - unchanged)
├── distance_analysis.py         # (existing - unchanged)
├── mixture_model.py             # (existing - unchanged)
├── posterior.py                 # (existing - unchanged)
├── reference.py                 # (existing - unchanged)
├── shrinkage.py                 # (existing - unchanged)
├── matrix_ops.py                # (existing - unchanged)
├── visualization.py             # (existing - unchanged)
├── utils.py                     # (existing - unchanged)
│
└── pair_concordance/            # NEW SUBPACKAGE
    ├── __init__.py
    ├── __main__.py              # CLI entry: python -m nome_hic_concordance.pair_concordance
    ├── cli.py                   # Argument parsing (parallel structure to top-level cli.py)
    │
    ├── panel.py                 # Pair panel construction (anchors, categories, distance bands)
    ├── tf_motifs.py             # TF motif scanning (JASPAR via FIMO wrapper)
    │
    ├── classify.py              # Per-cell read aggregation: linked vs unlinked at pair (i,j)
    ├── empirical_bayes.py       # Pooled EB priors per pair category
    ├── hierarchical.py          # Beta-binomial posterior on theta_L; Delta computation
    │
    ├── embed.py                 # E_contact / E_coord / E_full builders + masked-loss PCA
    ├── component_analysis.py    # AUC per category, marginal contribution, matched contact
    ├── subtype.py               # Subtype-resolution defense (cortex only)
    │
    ├── visualization.py         # Pair-panel-specific UMAP, forest plots, density panels
    │
    └── tests/
        ├── test_panel.py
        ├── test_classify.py             # synthetic: known linked/unlinked counts
        ├── test_hierarchical.py         # synthetic: known Delta effect, recover posterior
        ├── test_embed.py
        ├── test_end_to_end_small.py     # 20 synthetic cells, 50 pairs, full pipeline
        └── test_gm12878_imr90_smoke.py  # 50 real cells if available, smoke test
```

### 2.1 CLI

A second console entry point in `pyproject.toml`:

```toml
[project.scripts]
nome_hic_concordance = "nome_hic_concordance.cli:main"
nome_hic_pair_concordance = "nome_hic_concordance.pair_concordance.cli:main"
```

Invocation pattern mirrors the existing CLI:

```bash
nome_hic_pair_concordance \
    --posteriors-dir output/context_GCH/04_posteriors/ \
    --pair-panel pairs.bedpe \
    --output-dir output/pair_concordance/ \
    --eb-pool-by category \
    --n-threads 8
```

Pair panel is supplied as a BEDPE-style TSV (see §3.2). The pipeline does NOT re-parse BAMs — it operates entirely on existing `posteriors.parquet` outputs.

### 2.2 Step numbering (extends existing pipeline)

| Step | Stage | Existing or new |
|------|-------|-----------------|
| 1–4 | Context extraction → BAM parsing → mixture → posteriors | Existing (reused) |
| **5p** | **Pair panel construction** | **New** (replaces existing Step 5 for this analysis) |
| **6p** | **Linked/unlinked aggregation per pair, per cell** | **New** |
| **7p** | **EB priors + hierarchical Δ** | **New** |
| **8p** | **Three embeddings + UMAP** | **New** |
| **9p** | **Component-contribution analysis** | **New** |

Steps `Np` use the existing numbering style but with a `p` suffix to mark the pair-concordance branch. Existing Steps 5–7 (bin-based concordance, shrinkage, matrix ops) remain available; the two pipelines are independent and operate on the same Step 4 outputs.

---

## 3. Pair panel construction (Step 5p)

### 3.1 Anchors

Each "region" in a pair is a small anchor window: a fixed-size genomic interval (default 5 kb) centered on a feature of interest:

- For TF-anchored pairs: centered on a TF motif occurrence (FIMO p < 1e-4).
- For co-folded / non-folded pairs: centered on top-decile / bottom-decile pseudobulk contact frequency bins.
- For random_distance_matched: random 5 kb intervals at matched distance.

5 kb is the default because (a) at typical cellular contact density, this captures most pair-supporting reads while excluding off-target reads; (b) it matches the resolution at which Hi-C contact features are typically called; (c) it's large enough to contain ~10–30 GCH sites for binarization confidence.

### 3.2 Pair panel file format

A bedpe-style TSV consumed by `panel.py`:

```
pair_id    chrom_i  start_i  end_i  chrom_j  start_j  end_j  distance  category                       distance_band
EP_001     chr1     1000000  1005000  chr1     1500000  1505000  500000   tf_anchored_NEUROD2            mid
EP_002     chr2     2300000  2305000  chr2     5300000  5305000  3000000  random_distance_matched        long
EP_003     chr1     8000000  8005000  chr1     8800000  8805000  800000   co_folded                      mid
...
```

`pair_concordance/panel.py:build_panel()` constructs this from:
- A TF motif BED (output of FIMO scanning).
- A pseudobulk Hi-C cooler (for co-folded / non-folded selection).
- A configurable category specification YAML.

Distance bands: `short` (100–500 kb), `mid` (500 kb – 2 Mb), `long` (2–5 Mb).

### 3.3 Categories used in validation and cortex runs

| Category | Definition | Expected behavior under TF-folding hypothesis |
|----------|------------|----------------------------------------------|
| `co_folded` | Pseudobulk contact frequency in top decile at matched distance | Strong cell-type-specific Δ |
| `non_folded` | Pseudobulk contact frequency in bottom decile, distance-matched | Weak Δ (Step 1 negative control) |
| `tf_anchored_CTCF` | Both anchors contain CTCF motif (convergent for cortex) | Strong, broadly across cell types |
| `tf_anchored_NEUROD2` (cortex) | Both anchors NEUROD2 motif | Strong in Exc; null elsewhere |
| `tf_anchored_OLIG2` (cortex) | Both anchors OLIG2 motif | Strong in Oli |
| `tf_anchored_SPI1` (cortex) | Both anchors PU.1/SPI1 motif | Strong in Mic |
| `tf_anchored_NRF1` | Both anchors NRF1 motif (housekeeping) | NOT cell-type-specific — **falsification control** |
| `random_distance_matched` | Random pairs matched on distance distribution | Near-chance signal |

For GM12878+IMR90 validation, swap cortex TFs for cell-line-relevant TFs: PU.1/SPI1, EBF1, IRF4 (B-lymphoid) for GM12878; EOMES, TEAD, FOXA family for IMR90 fibroblast; CTCF and NRF1 panels apply to both.

Target panel size: **100,000–200,000 pairs total** across categories. Tractable on a single workstation.

### 3.4 Module sketch (panel.py)

```python
def build_panel(
    tf_motif_bed: dict[str, str],     # motif_name -> BED path of motif occurrences
    pseudobulk_cooler: str | None,    # for co_folded / non_folded selection
    chrom_sizes: dict[str, int],
    anchor_size: int = 5_000,
    distance_bands: dict[str, tuple[int, int]] = None,
    n_per_category_per_band: int = 5_000,
    output_path: str = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build a pair panel TSV. Returns the panel DataFrame."""
```

Deterministic given `random_state`. Panel is cached as parquet keyed on a config hash.

---

## 4. Per-pair-per-cell linked/unlinked aggregation (Step 6p)

### 4.1 Definition (reframed for actual meth_concord data)

For a given pair of regions (i, j) and cell c, every row in `posteriors.parquet` for cell c is one of:

| Class | Condition | Use |
|-------|-----------|-----|
| **Linked (L)** | (pos1 ∈ i AND pos2 ∈ j) OR (pos1 ∈ j AND pos2 ∈ i) | Joint observation of state at i AND j |
| **Unlinked-i (U_i)** | pos1 ∈ i AND pos2 ∉ j (and not pos1 ∈ j), OR symmetric on pos2 | Marginal state at i |
| **Unlinked-j (U_j)** | symmetric for j | Marginal state at j |
| **Other** | Neither pos1 nor pos2 in i ∪ j | Discarded (not informative for this pair) |

Distance filter from the existing pipeline (`min_distance`, default 20 kb) is reapplied — already enforced for the parquet rows; further restrict by `n_min` per end (existing `min_gch_sites`, default 3).

### 4.2 Binarization of per-read-end state

The existing pipeline produces continuous `posterior1`, `posterior2` ∈ [0, 1]. For the hierarchical model, binarize:

- **state = "open"** if posterior > τ (default 0.5; the existing pipeline's `--posterior-threshold`).
- **state = "closed"** if posterior ≤ τ.
- Reads with NaN posterior (n=0 ends) are discarded.

For each linked pair:
- **active-concordant** = both ends open
- **symmetric-concordant** = both ends same state (both open OR both closed)

For each unlinked-i read: state at end-in-i is "open" or "closed". Similarly for unlinked-j.

Active concordance is the primary outcome (reflects coordinated regulatory activation); symmetric is supplementary.

### 4.3 Per-pair, per-cell counts

For each (pair_id, cell_id), classify and tally:

| Quantity | Symbol | Definition |
|----------|--------|------------|
| Linked reads, total | $n_{ij}^{c,L}$ | Count of rows with end1∈i, end2∈j (or symmetric) and both ends informative |
| Linked reads, active-concordant | $k_{ij}^{c,L,A}$ | Of those, both ends open |
| Linked reads, symmetric-concordant | $k_{ij}^{c,L,S}$ | Of those, same state |
| Unlinked-i, total | $n_i^c$ | Reads with one end in i, mate ∉ j |
| Unlinked-i, open | $k_i^{c,\text{open}}$ | Of those, the in-i end open |
| Unlinked-j, total | $n_j^c$ | symmetric |
| Unlinked-j, open | $k_j^{c,\text{open}}$ | symmetric |

### 4.4 Implementation

`pair_concordance/classify.py:aggregate_pair_counts()`:

```python
def aggregate_pair_counts(
    posteriors_parquet: str,
    cell_name: str,
    panel: pd.DataFrame,            # output of panel.py
    posterior_threshold: float = 0.5,
    min_gch_sites: int = 3,
    min_distance: int = 20_000,
) -> pd.DataFrame:
    """
    For one cell, aggregate linked and unlinked counts at every panel pair.

    Returns DataFrame indexed by pair_id with columns:
        cell_name, pair_id,
        n_linked, k_linked_active, k_linked_symmetric,
        n_unlinked_i, k_unlinked_i_open,
        n_unlinked_j, k_unlinked_j_open,
        n_other  (diagnostic only)
    """
```

Implementation note for efficiency: build per-chromosome interval trees from the panel (anchor intervals → pair_ids), then sweep each row of `posteriors.parquet` once through `pos1` and `pos2` interval lookups. With ~200k pairs and ~5M read pairs per cell, this runs in seconds per cell; parallelize across cells via `joblib` (matching existing pipeline patterns).

Output: per-cell `{cell_name}.pair_counts.parquet`. Stacked across cells: one parquet table or a Zarr array indexed by (cell, pair_id) → counts.

### 4.5 Distance and coverage diagnostics

Per-cell summary outputs (mirroring existing `concordance_vs_distance.tsv` style):

- Histogram of `n_linked` per pair across the panel (median, fraction of pairs with ≥1, ≥5, ≥10 linked reads).
- Histogram of `n_unlinked_i + n_unlinked_j` per pair.
- Distance-binned summary of mean active-concordance rate among linked reads (sanity check against existing distance-decay plot).

If median `n_linked` per pair is <1 across the cell population at any distance band, the panel is too sparse for that band — flag in the diagnostic and downsample distance bands accordingly.

---

## 5. Hierarchical statistical model (Step 7p)

### 5.1 Linked-concordance posterior

For each (pair_id, cell_id) with $n_{ij}^{c,L} \geq 1$:

$$k_{ij}^{c,L,A} \mid \theta_{ij}^{c,L} \sim \text{Binomial}(n_{ij}^{c,L}, \theta_{ij}^{c,L})$$

$$\theta_{ij}^{c,L} \sim \text{Beta}(\alpha_{\text{cat}(ij)}^L, \beta_{\text{cat}(ij)}^L)$$

Posterior is closed-form:

$$\theta_{ij}^{c,L} \mid \text{data} \sim \text{Beta}(\alpha^L + k^L, \beta^L + n^L - k^L)$$

The prior $(\alpha_{\text{cat}}^L, \beta_{\text{cat}}^L)$ is pooled across all pairs in the same category (e.g., all `tf_anchored_NEUROD2` pairs share a prior). Estimated via method of moments from the cross-cell, cross-pair distribution of $k^L / n^L$ within that category, weighted by $n^L$:

$$\hat{\mu}_{\text{cat}}^L = \frac{\sum k_{ij}^{c,L}}{\sum n_{ij}^{c,L}}, \quad \hat{\sigma}^2_{\text{cat}}^L = \text{weighted variance of } (k^L/n^L)$$

$$\alpha_{\text{cat}}^L = \hat{\mu}_{\text{cat}}^L \cdot \nu_{\text{cat}}, \quad \beta_{\text{cat}}^L = (1-\hat{\mu}_{\text{cat}}^L) \cdot \nu_{\text{cat}}, \quad \nu_{\text{cat}} = \frac{\hat{\mu}(1-\hat{\mu})}{\hat{\sigma}^2} - 1$$

floored at $\nu \geq 2$ for numerical stability. Pooling by category (not per-pair) is essential because most pairs have very few linked reads per cell.

Optional refinement: pool by category × distance_band. Use this if cross-band variation in baseline concordance is large (check empirically).

`pair_concordance/empirical_bayes.py:fit_priors()` produces a table:

```
category    distance_band    alpha_L    beta_L    n_obs_pairs    n_obs_cells
tf_anchored_NEUROD2    mid    8.2    3.1    4521    347
...
```

### 5.2 Independence baseline (unlinked)

Per cell c, per pair (i, j):

$$\hat{p}_i^c = \frac{k_i^{c,\text{open}}}{n_i^c}, \quad \hat{p}_j^c = \frac{k_j^{c,\text{open}}}{n_j^c}$$

For active concordance:

$$\hat{\theta}_{ij}^{c,U,A} = \hat{p}_i^c \cdot \hat{p}_j^c$$

For symmetric concordance:

$$\hat{\theta}_{ij}^{c,U,S} = \hat{p}_i^c \hat{p}_j^c + (1-\hat{p}_i^c)(1-\hat{p}_j^c)$$

Variance via delta method (independence of $\hat{p}_i^c$, $\hat{p}_j^c$):

$$\text{Var}(\hat{\theta}^{c,U,A}) = (\hat{p}_j^c)^2 \cdot \frac{\hat{p}_i^c(1-\hat{p}_i^c)}{n_i^c} + (\hat{p}_i^c)^2 \cdot \frac{\hat{p}_j^c(1-\hat{p}_j^c)}{n_j^c}$$

If $n_i^c < 5$ or $n_j^c < 5$, fall back to a pooled-by-category $\hat{p}_{\text{cat}, i}$ across cells (similar shrinkage to the linked prior). This is the unavoidable cost of pair sparsity; flagged in diagnostics.

### 5.3 Coordination excess Δ

$$\Delta_{ij}^c = \mathbb{E}[\theta_{ij}^{c,L} \mid \text{data}] - \hat{\theta}_{ij}^{c,U,A}$$

$$= \frac{\alpha_{\text{cat}}^L + k_{ij}^{c,L,A}}{\alpha_{\text{cat}}^L + \beta_{\text{cat}}^L + n_{ij}^{c,L}} - \hat{p}_i^c \hat{p}_j^c$$

Posterior variance (independence approximation between the two terms; conservative since they may be weakly correlated through shared marginal state):

$$\sigma^2(\Delta_{ij}^c) = \text{Var}(\theta_{ij}^{c,L} \mid \text{data}) + \text{Var}(\hat{\theta}_{ij}^{c,U,A})$$

where the first term is the closed-form beta variance.

### 5.4 Worked examples (using the existing pipeline's posteriors)

**Case A:** Cell c at pair (i,j) — 2 linked read pairs, both with `posterior1 > 0.5` and `posterior2 > 0.5` (active-concordant). 1000 unlinked-i reads, 95% open. 1000 unlinked-j reads, 95% open.

- EB prior: assume `tf_anchored_NEUROD2` category gives Beta(6, 4) — mean 0.6, modest concentration.
- $\theta^L$ posterior: Beta(6+2, 4+0) = Beta(8, 4), mean = 0.667, SD ≈ 0.13.
- $\hat{\theta}^{U,A} = 0.95 \cdot 0.95 = 0.9025$, very tight.
- $\Delta = 0.667 - 0.9025 = -0.236$, SD ≈ 0.13.
- Interpretation: linked reads do NOT show concordance beyond what marginal accessibility predicts. The pair is co-active but Δ is strongly negative — coordination is marginal-state-driven, not contact-conditional. **Embedding weight is moderate (precision OK); value is negative.**

**Case B:** 1000 linked reads, 50% active-concordant. 2 unlinked-i, 2 unlinked-j.

- $\theta^L$ posterior: Beta(6+500, 4+500) = Beta(506, 504), mean = 0.501, SD ≈ 0.016. Tight.
- $\hat{p}_i^c, \hat{p}_j^c$ each from 2 reads each → very wide; fall back to category-pooled $\hat{p}_{\text{cat}, i, j}$. Say $\hat{p}_{\text{cat}} = 0.5$, so $\hat{\theta}^{U,A} = 0.25$, with moderate variance.
- $\Delta = 0.501 - 0.25 = +0.25$, SD dominated by unlinked-baseline shrinkage variance ≈ 0.15.
- Interpretation: linked reads show concordance well above the marginal-pair baseline — coordination IS contact-conditional. Note this contrasts with my earlier sketch, because we now apply category pooling for the unlinked baseline when it's too sparse, which makes Case B's signal interpretable. **Embedding weight moderate; value strongly positive.**

These two cases are distinct biological situations distinguished by Δ and its variance. The hierarchical model preserves the distinction.

### 5.5 Outputs

`pair_concordance/hierarchical.py` produces a per-cell parquet `{cell_name}.delta.parquet`:

```
pair_id    n_linked    k_linked_active    theta_L_mean    theta_L_var
           p_i    p_j    theta_U_active    theta_U_var
           delta    delta_var    delta_z
```

And a stacked table across cells (cells × pairs sparse): `delta_matrix.parquet` and `delta_var_matrix.parquet`. Storing as parquet with cell as row index, pair_id as columns, float32 values. NaN where the cell has no informative data at the pair.

---

## 6. Three parallel embeddings (Step 8p)

### 6.1 E_contact (Step 1 only)

Per-cell feature per pair: depth-normalized linked-read count.

$$u_{ij}^c = \log(1 + n_{ij}^{c,L}) - \log(\lambda_c)$$

where $\lambda_c = \sum_{(i,j) \in \text{panel}} n_{ij}^{c,L}$ is the cell's total linked-read count over the panel.

Standardize per pair (z-score across cells); PCA on the (cells × pairs) matrix → UMAP. This is "what the contact-only embedding gives" — analogous to a panel-restricted sn-m3C-seq embedding.

`pair_concordance/embed.py:build_contact_embedding()` produces `embedding_contact.h5ad` with `.obsm['X_pca']`, `.obsm['X_umap']`.

### 6.2 E_coord (Step 2 only, contact-conditional)

Per-cell feature per pair: precision-weighted Δ z-score, restricted to pairs where the cell has informative linked AND unlinked data.

$$v_{ij}^c = \begin{cases} \Delta_{ij}^c / \sigma(\Delta_{ij}^c) & \text{if } n_{ij}^{c,L} \geq 1 \text{ AND } \min(n_i^c, n_j^c) \geq 3 \\ \text{missing} & \text{otherwise} \end{cases}$$

The key property: **every contributing cell-pair observation conditions on contact being observed**. Step 1 axis is partialled out at the per-pair-per-cell level, not via post-hoc regression. Cells that happen to have many contacts at this pair drive the feature for this pair; cells with no contacts contribute nothing.

PCA cannot handle missingness natively. Two options implemented:

- **Masked-loss PCA** (default): iterative SVD with a binary observation mask. Implementation in `embed.py:sparse_pca_em()`, ~50 lines, runs in seconds for ~1500 × ~200k matrices.
- **Mean-imputation + PCA** (sanity-check fallback): substitute pair-mean for missing entries, run standard PCA. Faster but biases toward marginal pair structure. Available via `--pca-method mean_impute`.

UMAP from top 20 PCs → `embedding_coord.h5ad`.

### 6.3 E_full (integrated)

Concatenate features from E_contact and E_coord (after standardization), plus per-modality marginal accessibility features per anchor (mean posterior across all reads with at least one end in that anchor). Standardize; PCA; UMAP. → `embedding_full.h5ad`.

### 6.4 Diagnostics per embedding

For each embedding, output:

- 2D UMAP plot colored by cluster label and by total depth (QC).
- ARI of Leiden clustering vs ground-truth cluster labels.
- Silhouette coefficient per cluster.
- Cell-type-classification AUC (§7.1, 5-fold CV logistic on top 20 PCs).

---

## 7. Component-contribution analysis (Step 9p)

### 7.1 AUC per category per embedding

For each (embedding, pair category, distance band) combination, compute cell-type classification AUC. Procedure:

1. Restrict the cells × pairs matrix to pairs in the (category, band).
2. PCA top 20 components.
3. 5-fold CV logistic regression: `cell_type ~ PC1 + ... + PC20`.
4. Report mean AUC across folds with bootstrap CI.

Expected pattern under the TF-folding hypothesis:

| Embedding × Category | Expected |
|---|---|
| E_contact × any | Moderate to high AUC (contacts carry cell-type info) |
| E_coord × `tf_anchored_<lineage matched to cell type>` | High AUC, AND **statistically higher than E_contact at same pairs** |
| E_coord × `tf_anchored_NRF1` | Low AUC (housekeeping — falsification control) |
| E_coord × `random_distance_matched` | Near chance |
| E_full × any | ≥ max(E_contact, E_coord) |

Output: forest plot, rows = pair categories × bands, columns = embeddings, dots = AUC with CI bars. This is the **headline figure** for the component decomposition (Fig 4 Panel C2 in the parallel scnomehic_analysis plan).

`pair_concordance/component_analysis.py:auc_per_category()`.

### 7.2 Marginal contribution of E_coord over E_contact

Test whether E_coord adds cell-type information beyond what E_contact already carries. Two formulations:

**LR test (preferred):**

1. Fit base logistic: `cell_type ~ E_contact_PCs (top 20)`. Get log-likelihood $\ell_0$.
2. Fit augmented: `cell_type ~ E_contact_PCs + E_coord_PCs (top 20)`. Get log-likelihood $\ell_1$.
3. Statistic $2(\ell_1 - \ell_0) \sim \chi^2_{20}$ under H0.

**ΔAUC (interpretable companion):**

ΔAUC = AUC(augmented model) − AUC(base model), with bootstrap CI. Report alongside LR p-value.

Done globally and per pair category. Negative ΔAUC at NRF1 pairs (or near-zero) is expected and is the desired falsification pattern.

### 7.3 Matched contact frequency comparison

For each cell pair (c, c') in the full dataset:

1. Compute contact similarity $S_{\text{contact}}(c, c') = \cos(\mathbf{u}_c, \mathbf{u}_{c'})$ on the E_contact feature space.
2. Stratify cell pairs into quartiles of $S_{\text{contact}}$.
3. Within the top quartile, compute coordination similarity $S_{\text{coord}}(c, c') = \cos(\mathbf{v}_c, \mathbf{v}_{c'})$ on the E_coord feature space (with masking).
4. Regression: $S_{\text{coord}} \sim S_{\text{contact}} \cdot \text{same\_type} + S_{\text{contact}} + \text{same\_type}$.
5. The headline coefficient is `same_type` within the top contact-similarity quartile: cells with nearly identical contact patterns still differ in coordination by cell type → positive coefficient.

Memory-aware implementation: with ~1500 cells, there are ~10⁶ cell pairs. Compute in blocks; subsample to ~100k pairs for the regression. `pair_concordance/component_analysis.py:matched_contact_comparison()`.

### 7.4 Subtype resolution (cortex only)

For mouse cortex Exc subtypes (if cell counts permit ≥100 per subtype):

- Confusion matrix of subtype classification from E_contact (top 20 PCs, 5-fold CV multinomial).
- Same from E_coord.
- Δ-accuracy per subtype pair (e.g., L2/3 vs L5).
- Report which subtype contrasts benefit most from E_coord — those are the "uniquely scNOMe-HiC-enabled" findings.

If subtype counts are too low, defer to coarser contrasts: Pvalb vs Sst (inhibitory subtypes), or OPC vs Oli (similar lineage).

`pair_concordance/subtype.py:run_subtype_analysis()`.

---

## 8. Validation pipeline — GM12878 + IMR90 mixed cells (Step 9p in validation mode)

### 8.1 Cell-of-origin ground truth (run BEFORE pair-concordance pipeline)

External script (not in this package; user's existing analysis):

1. Genotype each cell at a panel of ~10,000 SNPs distinguishing NA12878 from IMR90.
2. Per cell, compute log-likelihood ratio: GM12878 vs IMR90 reference.
3. Hard-call cells with |LLR| > 5; discard ambiguous (likely doublets, <5% in clean mixes).
4. Output `ground_truth.tsv` with `cell_name`, `true_label`.

This file is the input to the validation pipeline.

### 8.2 Pipeline (`pair_concordance/pipelines/validate_gm12878_imr90.py`)

```bash
nome_hic_pair_concordance \
    --posteriors-dir output/context_GCH/04_posteriors/ \
    --pair-panel panels/gm12878_imr90.bedpe \
    --ground-truth ground_truth.tsv \
    --output-dir output/pair_concordance_gm12878_imr90/ \
    --eb-pool-by category \
    --n-threads 8 \
    --mode validation
```

Validation mode:

1. Build/load pair panel.
2. Aggregate linked/unlinked counts per cell.
3. Fit EB priors per category.
4. Compute Δ per pair per cell.
5. Build E_contact, E_coord, E_full.
6. Compute classification accuracy vs `ground_truth.tsv`.
7. Run AUC per category, matched contact comparison.
8. Generate validation report (JSON + PDF).

### 8.3 Pre-registered acceptance criteria

Going to mouse cortex is gated on passing all four. **Commit to these thresholds before running.**

| Criterion | Threshold | Rationale |
|---|---|---|
| E_full vs ground truth: classification accuracy | ≥ 0.95 | The mix is well-separated; lower means pipeline error |
| E_coord vs ground truth: classification accuracy | ≥ 0.85 | Coord alone should still discriminate two very different cell lines |
| AUC at `random_distance_matched` pairs (E_coord) | ≤ 0.65 | Confirms coord measures pair-specific signal, not noise |
| AUC at `tf_anchored_NRF1` pairs is significantly less than AUC at lineage TF pairs (E_coord) | LR test p < 0.05 | Housekeeping ≠ lineage — confirms TF-folding hypothesis structure |

If any criterion fails: STOP. Debug. Do not proceed to mouse cortex.

### 8.4 Likely failure modes and diagnostics

| Symptom | Likely cause | Check |
|---|---|---|
| E_coord ARI ≪ E_contact ARI | Coordination signal too sparse to embed | `median per-cell informative pair count`; need ≥500 |
| All Δ near zero | Independence baseline computed wrong | Inspect a few high-coverage pairs manually |
| NRF1 AUC ≈ lineage TF AUC | TF panel construction wrong, OR coord captures only marginal state, OR mixture model classifying everything as same state | Re-run mixture model diagnostic from existing Step 3; check motif scan output |
| Per-pair Δ has heavy tails | EB priors not regularizing enough at low-count pairs | Inspect `category` priors; consider per-band pooling; floor $\nu \geq 5$ |

### 8.5 Negative-control smoke test (before reporting any signal)

Within each cell, randomly relabel linked reads as unlinked-i (50%) and unlinked-j (50%), preserving per-cell counts. Recompute Δ. Should produce Δ ~ 0 with no cell-type-specific AUC. **Run before reporting any positive result.** This is the simplest possible falsification.

---

## 9. Mouse cortex application

Run only after GM12878+IMR90 validation passes all §8.3 criteria.

`pair_concordance/pipelines/run_mouse_cortex.py`:

- Same pipeline structure.
- Pair panel uses mouse TF motifs (NEUROD2, OLIG2, SPI1/PU.1, MEF2C, NRF1, CTCF).
- Cluster labels are inputs (assumed available from upstream tri-modal integration).
- Adds subtype analysis (§7.4).
- Outputs feed into Fig 4 Panel C (see parallel `SCNOMEHIC_FIG4_PLAN_v1.md` in the sibling repo).

---

## 10. Implementation details

### 10.1 Performance

For ~1,500 cells × 200,000 pairs:

- Linked/unlinked aggregation: minutes per cell (interval tree lookup over `posteriors.parquet` rows). Parallel across cells via `joblib`, matching existing pipeline conventions. ~10 minutes total on 8 threads.
- EB prior estimation: closed-form MoM per category; seconds.
- Δ computation: closed-form per pair-cell; minutes.
- Masked PCA: minutes for ~1500 × 200k matrix with masking.
- UMAP: minutes.
- **End-to-end: under 30 minutes on a workstation** (excluding the existing meth_concord Steps 1–4 that produce posteriors).

No MCMC, no iterative fitting. Everything closed-form.

### 10.2 Memory

Sparse Δ matrix (cells × pairs): with 200k pairs × 1500 cells × float32 = 1.2 GB if dense. Most entries are NaN, but parquet handles this; in-memory representation can be CSR sparse or numpy ndarray depending on density. With observation mask, ~2 GB peak. Fits comfortably on a laptop.

### 10.3 Dependencies (in addition to existing meth_concord deps)

`scanpy` (for `anndata` and UMAP plotting helpers), `pybedtools` or `pyranges` (interval ops), `scikit-learn` (logistic regression, PCA). All `pip install`. JASPAR motif scanning via FIMO (MEME suite — external tool, not pip; alternative: `pyJASPAR` for in-Python motif scoring).

### 10.4 Reproducibility

- Pair panel construction deterministic given `random_state`.
- Every config saved to `pipeline_config.json` matching existing pattern.
- Intermediate parquets versioned with a config hash in filename.

### 10.5 Testing strategy

- **Unit tests** matching existing test structure under `tests/`:
  - `test_panel.py`: panel construction (anchor sizes, distance bands, category balance).
  - `test_classify.py`: synthetic posteriors parquet with known linked/unlinked structure; verify counts.
  - `test_hierarchical.py`: synthetic data with known Δ effect; verify posterior recovery.
  - `test_embed.py`: known-cluster-structure synthetic data; verify embeddings recover structure.
- **End-to-end small** (`test_end_to_end_small.py`): 20 synthetic cells (10 cluster A, 10 cluster B), 50 pairs with known ground-truth coordination. Full pipeline should recover cluster assignment with accuracy ≥ 0.9.

---

## 11. Timeline

| Week | Deliverable |
|---|---|
| 1 | Panel module (`panel.py`, `tf_motifs.py`); unit tests |
| 2 | Linked/unlinked aggregation (`classify.py`); EB priors; unit tests on synthetic |
| 3 | Hierarchical model (`hierarchical.py`); Δ outputs; worked-example reproduction |
| 4 | Three embeddings (`embed.py`); masked PCA; UMAP |
| 5 | Component analysis (`component_analysis.py`); AUC, matched contact, LR test |
| 6 | GM12878+IMR90 validation run; check §8.3 criteria |
| 7 | If pass: mouse cortex run; subtype analysis |
| 8 | Figures, documentation, methods writing for manuscript |

Hard stop after Week 6 if validation fails. Debug in place until it passes.

---

## 12. Open decisions to resolve before starting

1. **HCG (methylation) parallel pipeline?** The existing meth_concord pipeline can extract HCG context with `--contexts GCH,HCG`. Run the contact-aware pipeline separately on HCG posteriors as a tri-modal extension. Recommend deferring to v3 of this design after the GCH validation passes — adds significant scope without changing the methodological story.
2. **Anchor size** (default 5 kb). Sensitivity: try 2.5 kb, 5 kb, 10 kb on validation. Larger = more linked reads per pair but less spatial resolution.
3. **`posterior_threshold` for binarization** (default 0.5, matches existing pipeline). Sensitivity: 0.4, 0.5, 0.6. Or use continuous posteriors and define soft concordance — keep this for v3.
4. **Active vs symmetric concordance as primary** — recommend active (more biologically interpretable); report both.
5. **EB pooling level**: category, category × band, or per-pair. Recommend category × band; per-pair as sensitivity.
6. **Whether to use the existing `--shrinkage-n0` empirical Bayes framework for Δ** instead of beta-binomial per-pair posteriors. Recommend: no. Δ has fundamentally different structure (mean shift, not correlation shrinkage) and the existing shrinkage is for Pearson r of population posteriors at the bin level.
7. **Re-use the existing GM12878+IMR90 posteriors output dir, or run end-to-end** — assume re-use, since Steps 1–4 of meth_concord must be done anyway.

---

## 13. Relationship to other planning documents

- **Existing `design/NOME_HIC_CONCORDANCE_PIPELINE_DESIGN.md`** in this repo: documents Steps 1–7 of the current bin-based concordance pipeline. This v2 design supplements that document with the pair-concordance subpackage (Steps 5p–9p).
- **`SCNOMEHIC_FIG4_PLAN_v1.md`** (sibling `scnomehic_analysis` repo): the contact-aware pipeline here implements the statistical core of Fig 4 Panel C (3D-conditioned tri-modal coupling). Mouse-cortex outputs from this pipeline feed directly into the figure.
- **Future companion doc**: `PAIR_CONCORDANCE_MATH_v1.md` should formalize the likelihood, prior specification, posterior derivations, and identifiability conditions for the manuscript methods section.

---

## 14. Version history

- **v1**: Generic design before meth_concord source was available; used a hypothetical adapter layer and assumed external data structures.
- **v2 (current):** Rewritten after reviewing actual `meth_concord` codebase. Builds directly on `04_posteriors/{cell_name}.posteriors.parquet` outputs. Reuses Steps 1–4 unchanged. Reframes "linked vs unlinked" as re-aggregation of existing parquet rows by pair-of-regions definition (not new data extraction). New subpackage under `nome_hic_concordance/pair_concordance/` with a second CLI entry point.

Future revisions:
- v3 after Week 5 sprint: refine based on actual implementation experience.
- v4 after Week 6 GM12878+IMR90 validation: lock or revise based on validation outcomes.
- v5: add HCG (tri-modal) extension if v2 results justify it.
