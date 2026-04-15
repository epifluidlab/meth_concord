# NOMe-HiC Long-Range Chromatin Accessibility Concordance Pipeline

## Design Document v1.0

---

## 1. Motivation and Overview

NOMe-HiC simultaneously profiles DNA methylation (HCG), chromatin accessibility (GCH), and 3D genome organization within individual cells. Read pairs that map to distant genomic loci (≥20kb) but are spatially proximal (via ligation) provide a unique opportunity: the GCH methylation at each read end reflects the local chromatin accessibility, and the *concordance* between ends measures whether spatially proximal loci share the same chromatin state.

This pipeline quantifies long-range GCH concordance per genomic bin per cell, producing a cells × bins matrix suitable for UMAP-based cell clustering.

### Pipeline Summary

```
Input BAMs + .2bit genome + dbSNP VCF (optional)
        │
        ▼
[1] Extract GCH contexts from reference (.2bit → GCH positions)
    └── Exclude GCG, GCC contexts; filter dbSNP positions
        │
        ▼
[2] Parse BAM files: extract read pairs, count GCH methylation per read end
    └── Reference comparison → call methylation (no strand tag needed)
        │
        ▼
[3] Fit two-component beta-binomial mixture (EM) on all read ends
    ├── Subsample to --em-subsample (default 10M) if needed
    ├── Fit on all pairs
    ├── Fit on short-range pairs (<1kb)
    ├── Fit on long-range pairs (>20kb)
    ├── Diagnostic: compare (α₁, β₁, α₂, β₂) across fits
    └── Optional: fit per-cell models for diagnostic (--em-cell-specific)
        │
        ▼
[4] Classify each read end: P(accessible | k, n) using fitted mixture
        │
        ▼
[5] Filter to qualifying pairs (distance ≥ D_min, GCH sites ≥ N_min per end)
        │
        ▼
[6] Compute per-bin concordance: Pearson r of (posterior₁, posterior₂)
        │
        ▼
[7] Shrinkage analysis → determine optimal n₀
        │
        ▼
[8] Apply shrinkage: r_shrunk = r_raw × n/(n + n₀)
        │
        ▼
[9] Filter bins, impute NAs, optional z-score
        │
        ▼
[10] UMAP visualization
```

---

## 2. Input / Output Specification

### 2.1 Inputs

**Sample manifest** (`samples.txt`): Tab-delimited, no header.

```
cell_name    /path/to/sample.bam    group_name(optional)
cell_001     /data/cell_001.bam     WT
cell_002     /data/cell_002.bam     KO
cell_003     /data/cell_003.bam
```

- Column 1: Cell identifier (unique string)
- Column 2: Path to coordinate-sorted, indexed BAM file (`.bai` must exist)
- Column 3 (optional): Group label for coloring UMAP plots

**Reference genome** (`.2bit`): Used to extract GCH/HCG/WCG trinucleotide contexts.

### 2.2 Configurable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-distance` | 20000 | Minimum genomic distance (bp) between read pair ends |
| `--min-gch-sites` | 3 | Minimum informative GCH sites per read end |
| `--bin-size` | 5000000 | Genomic bin size (bp) for concordance calculation |
| `--min-pairs-per-bin` | 5 | Bins with fewer pairs set to NA |
| `--min-cell-coverage` | 0.8 | Fraction of cells required non-NA to retain a bin |
| `--n-threads` | 4 | Number of parallel threads |
| `--em-max-iter` | 200 | Maximum EM iterations |
| `--em-tol` | 1e-6 | EM convergence tolerance (log-likelihood delta) |
| `--short-range-max` | 1000 | Upper bound for short-range pair definition (bp) |
| `--posterior-threshold` | 0.5 | Threshold for optional hard binarization |
| `--shrinkage-n0` | auto | Pseudocount for shrinkage (auto = empirical Bayes) |
| `--dbsnp-vcf` | None | Path to tabix-indexed dbSNP VCF for SNP filtering |
| `--em-subsample` | 10000000 | Max read ends for EM fitting (subsample if exceeded) |
| `--em-cell-specific` | False | Also fit EM per-cell for diagnostic comparison |

### 2.3 Outputs

```
output/
├── 01_context_extraction/
│   ├── context_GCH.{chrom}.npz            # Context positions per chromosome
│   ├── context_HCG.{chrom}.npz            # (if --contexts GCH,HCG)
│   └── ...
├── context_GCH/                            # Per-context output tree
│   ├── 02_read_pair_counts/
│   │   └── {cell_name}.read_pairs.parquet  # Per-pair counts (k, n per end)
│   ├── 03_mixture_model/
│   │   ├── em_fit_all.json                 # Fitted parameters (all pairs)
│   │   ├── em_fit_short.json               # Fitted parameters (short-range)
│   │   ├── em_fit_long.json                # Fitted parameters (long-range)
│   │   ├── mixture_comparison.png          # Diagnostic: parameter comparison
│   │   ├── mixture_fit_density.png         # Observed vs fitted distributions
│   │   ├── subsample_log.json              # Subsampling details
│   │   ├── cell_specific_mixture_params.tsv  # Per-cell params (if --em-cell-specific)
│   │   └── cell_vs_pooled_diagnostic.png   # Cell vs pooled (if --em-cell-specific)
│   ├── 04_posteriors/
│   │   └── {cell_name}.posteriors.parquet  # P(accessible | k, n) per read end
│   ├── 05_concordance/
│   │   ├── concordance_raw.parquet         # cells × bins raw Pearson r
│   │   ├── concordance_raw.tsv
│   │   ├── pair_counts_per_bin.parquet     # cells × bins number of pairs
│   │   ├── concordance_vs_distance.tsv     # Concordance decay with distance
│   │   ├── concordance_vs_distance.png     # Distance decay plot
│   │   ├── shrinkage_analysis.png          # Variance vs n_pairs, optimal n₀
│   │   ├── shrinkage_n0.json              # Chosen n₀ and fitting details
│   │   ├── concordance_shrunk.parquet      # After shrinkage
│   │   ├── concordance_imputed.parquet     # After NA imputation
│   │   ├── concordance_zscore.parquet      # After z-score per bin
│   │   └── retained_bins.txt              # Bins passing coverage filter
│   └── 06_umap/
│       ├── umap_no_zscore.png             # UMAP on imputed (no z-score)
│       ├── umap_zscore.png                # UMAP on z-scored
│       ├── umap_coordinates_no_zscore.tsv
│       └── umap_coordinates_zscore.tsv
├── context_HCG/                            # Same structure (if --contexts GCH,HCG)
│   └── ...
└── pipeline_config.json                    # Full parameter record
```

---

## 3. Module Architecture

```
nome_hic_concordance/
├── __init__.py
├── cli.py                    # Argument parsing, main entry point
├── reference.py              # Context extraction from .2bit (GCH/HCG/WCG)
├── bam_parser.py             # Read pair extraction and context counting
├── mixture_model.py          # Beta-binomial mixture EM fitting
├── posterior.py              # Posterior computation and classification
├── concordance.py            # Per-bin concordance calculation
├── distance_analysis.py      # Concordance vs. distance decay analysis
├── shrinkage.py              # Empirical Bayes shrinkage analysis
├── matrix_ops.py             # Filtering, imputation, z-scoring
├── visualization.py          # UMAP, diagnostic plots
└── utils.py                  # I/O, logging, parallelization helpers
```

---

## 4. Detailed Module Specifications

### 4.1 `reference.py` — Cytosine Context Extraction

**Purpose:** Parse `.2bit` reference to identify all positions for a user-specified cytosine context genome-wide. Supported contexts: GCH (chromatin accessibility), HCG (DNA methylation), WCG (CpG methylation subset). Extracts positions on **both strands** of the reference. Optionally filter positions overlapping dbSNP variants.

**Context definitions:**

| Context | Valid trinucleotides | Excludes | Measurement |
|---------|---------------------|----------|-------------|
| GCH | GCA, GCC, GCT | GCG (ambiguous) | Chromatin accessibility |
| HCG | ACG, CCG, TCG | GCG (ambiguous) | DNA methylation |
| WCG | ACG, TCG | CCG, GCG | CpG methylation (weak-base only) |

**Algorithm:**
1. For each chromosome, scan the + strand for matching trinucleotide patterns
2. Record the C position and strand (+) for + strand matches
3. Scan the − strand by finding complement patterns on the + strand:
   - GCH on − strand = HGC on + strand (e.g., AGC, CGC excluded, TGC)
   - HCG on − strand = CGH on + strand (e.g., CGA, CGC excluded, CGT)
   - Record the G position on + strand (which is the C on − strand) and strand (−)
4. If dbSNP VCF provided: exclude sites where any trinucleotide position overlaps a known SNP
5. Store as sorted arrays per chromosome, saved as `.npz`

**Key data structure:**

Each context site is stored as: `(position, strand, ref_base)` where:
- `position`: 0-based coordinate of the cytosine (or its complement G) on the + strand reference
- `strand`: '+' or '−'
- `ref_base`: 'C' (+ strand site) or 'G' (− strand site, i.e., C on − strand)

**Key function:**

```python
def extract_context_positions(twobit_path: str, output_dir: str,
                              contexts: list[str] = ['GCH'],
                              dbsnp_vcf: str = None,
                              n_threads: int = 4) -> dict[str, dict]:
    """
    Extract cytosine context positions from .2bit reference genome.

    Parameters
    ----------
    twobit_path : str
        Path to .2bit reference file.
    output_dir : str
        Directory to save per-chromosome .npz files.
    contexts : list of str
        Contexts to extract. Each is a 3-letter string using IUPAC codes:
        H={A,C,T}, W={A,T}, S={C,G}, etc. Default: ['GCH'].
    dbsnp_vcf : str, optional
        Path to tabix-indexed dbSNP VCF. Positions overlapping SNPs excluded.
    n_threads : int
        Parallel workers for chromosome processing.

    Returns
    -------
    dict mapping context_name → dict mapping chrom → structured array with fields:
        'pos' (int64): 0-based position on + strand reference
        'ref_base' (char): 'C' for + strand sites, 'G' for − strand sites
    """
```

**Dependencies:** `twobitreader` or `py2bit`, `pysam` (for VCF reading)

**Threading:** Parallelize across chromosomes via `joblib`.

**Implementation:** See `reference.py` in Phase 1 code (Section 7) for full implementation with IUPAC matching, dual-strand extraction, and dbSNP filtering.

---

### 4.2 `bam_parser.py` — Read Pair Context Counting

**Purpose:** For each cell's BAM file, extract properly-paired reads and count methylation at each read end for the specified cytosine context.

**Algorithm per read pair:**

1. Read BAM with `pysam`, filter to properly paired, primary alignments, MAPQ ≥ 20
2. For each read in a pair:
   - Intersect read alignment coordinates with context positions (from reference)
   - Exclude positions within `--trim-ends` bp of read ends (default: 5bp) to avoid end-repair artifacts
   - For each context site covered by the read, use reference comparison:
     - If `ref_base='C'` (+ strand site): read C = methylated, T = unmethylated, else skip
     - If `ref_base='G'` (− strand site): read G = methylated, A = unmethylated, else skip
   - Record: `(k_methylated, n_total)` for this read end
3. Compute genomic distance = |pos_read1 - pos_read2| (midpoint to midpoint)
4. Store per pair: `(chrom, pos1, k1, n1, pos2, k2, n2, distance)`
   - Only intra-chromosomal pairs are used

**Key data structure:**

```python
@dataclass
class ReadPairContext:
    chrom: str
    pos1: int             # Midpoint of read 1
    k1: int               # Methylated context count, end 1
    n1: int               # Total informative context sites, end 1
    pos2: int             # Midpoint of read 2
    k2: int               # Methylated context count, end 2
    n2: int               # Total informative context sites, end 2
    distance: int         # Genomic distance (bp)
```

**Output:** Per-cell parquet file with columns: `[chrom, pos1, k1, n1, pos2, k2, n2, distance]`

**Threading:** Parallelize across cells. Each cell processes independently.

**Bisulfite strand handling for bhmem (`-nonDirectional -pbat`):**

Bhmem is a BWA-MEM-based bisulfite Hi-C aligner. With `-nonDirectional -pbat`, all four bisulfite strand combinations (End1-CT/End2-CT, End1-GA/End2-GA, End1-CT/End2-GA, End1-GA/End2-CT) are attempted, and the best pair is selected by MAPQ → alignment score → mismatches → cigar match length. **Critically, bhmem does NOT write any bisulfite strand tag (no ZS, XG, or YD).** Instead, the **original read sequence** is stored in the BAM, oriented to match the reference strand.

**Methylation calling strategy — reference comparison:**

Since the BAM contains the original (unconverted) read sequence aligned to the reference, and we don't know which conversion strand each read came from, we call methylation by direct comparison of the read base to the reference at each cytosine context position. This works because:

- **C→T conversion strands (OT/CTOB):** At a reference C position, unmethylated C was converted to T. Read shows C (methylated) or T (unmethylated).
- **G→A conversion strands (OB/CTOT):** At a reference G position (complement of C on − strand), unmethylated G→A. Read shows G (methylated) or A (unmethylated).

For a context site (e.g., GCH) defined on the **+ strand** of the reference, the informative C is at position p where ref=C:
- Read base = **C** → methylated (accessible for GCH)
- Read base = **T** → unmethylated (closed for GCH)
- Read base = G, A, or N → skip (uninformative — this read came from the opposite conversion strand, or there's a SNP/error)

For the same context on the **− strand** (complement: HGC on + strand), the C on − strand corresponds to G on + strand at position q:
- Read base = **G** → methylated (accessible for GCH)
- Read base = **A** → unmethylated (closed for GCH)
- Read base = C, T, or N → skip

This approach is **strand-agnostic** — we don't need to know which conversion strand the read came from. We simply check whether the read shows the expected methylated or unmethylated base at each context position, and skip uninformative positions. Approximately half of reads covering a given context site will be informative (from the matching conversion strand), and the other half will be skipped.

**Context exclusions (per NOMe-HiC protocol):**
- Exclude **GCG** positions: ambiguous between GCH accessibility and HCG methylation

---

### 4.3 `mixture_model.py` — Beta-Binomial Mixture EM

**Purpose:** Fit a two-component beta-binomial mixture model to the marginal distribution of (k, n) across all read ends. Provides the generative model for classifying each end as accessible or closed.

#### 4.3.1 Mathematical Formulation

**Generative model:**

For read end i with k_i methylated out of n_i GCH sites:

```
z_i ~ Bernoulli(π)                          # Latent state: 1 = accessible, 0 = closed
p_i | z_i ~ Beta(α_{z_i}, β_{z_i})         # Per-read methylation rate
k_i | p_i, n_i ~ Binomial(n_i, p_i)        # Observed counts
```

Marginalizing over p_i:

```
P(k_i | n_i, z_i=j) = BetaBinomial(k_i | n_i, α_j, β_j)
                     = C(n_i, k_i) × B(k_i + α_j, n_i - k_i + β_j) / B(α_j, β_j)
```

where B(·,·) is the Beta function.

**Mixture likelihood:**

```
P(k_i | n_i, θ) = π · BB(k_i | n_i, α₁, β₁) + (1 - π) · BB(k_i | n_i, α₂, β₂)
```

where θ = (π, α₁, β₁, α₂, β₂).

**Note on parameterization:** We use (α, β) rather than (μ, φ) to stay close to the conjugate structure. The mean is μ = α/(α+β) and the overdispersion parameter is 1/(α+β+1). For reporting, we convert to (mean, overdispersion) for interpretability.

#### 4.3.2 EM Algorithm

**E-step:** Compute responsibilities

```
γ_i = π · BB(k_i | n_i, α₁, β₁) / P(k_i | n_i, θ)
```

γ_i is P(accessible | k_i, n_i, θ).

**M-step:**

Update π:
```
π_new = (1/N) Σ γ_i
```

Update (α_j, β_j) for each component j:

There is no closed-form M-step for beta-binomial. Use numerical optimization (L-BFGS-B) of the weighted log-likelihood within each M-step:

```
For component j=1 (accessible):
  maximize_{α₁, β₁}  Σ_i  γ_i · log BB(k_i | n_i, α₁, β₁)

For component j=0 (closed):
  maximize_{α₀, β₀}  Σ_i  (1 - γ_i) · log BB(k_i | n_i, α₀, β₀)
```

**Convergence:** Monitor total log-likelihood; stop when |ΔLL| < `--em-tol` or `--em-max-iter` reached.

**Initialization:**
- π₀ = 0.3 (prior: minority of loci are accessible)
- Component 1 (accessible): α₁ = 8, β₁ = 2 (mean ~0.8)
- Component 0 (closed): α₀ = 2, β₀ = 8 (mean ~0.2)
- Run 5 random restarts (perturbing initial parameters ±30%) and keep the fit with highest final log-likelihood

#### 4.3.3 Implementation

```python
class BetaBinomialMixture:
    """Two-component beta-binomial mixture model fitted via EM."""

    def __init__(self, max_iter=200, tol=1e-6, n_restarts=5, random_state=42):
        self.max_iter = max_iter
        self.tol = tol
        self.n_restarts = n_restarts
        self.random_state = random_state

        # Fitted parameters
        self.pi_ = None          # Mixing weight (accessible)
        self.alpha_ = None       # [α_closed, α_accessible]
        self.beta_ = None        # [β_closed, β_accessible]
        self.converged_ = False
        self.n_iter_ = 0
        self.log_likelihood_ = None

    def fit(self, k: np.ndarray, n: np.ndarray) -> "BetaBinomialMixture":
        """
        Fit mixture model via EM.

        Parameters
        ----------
        k : array of shape (N,)
            Methylated GCH counts per read end.
        n : array of shape (N,)
            Total GCH counts per read end. Must be >= 1.

        Returns
        -------
        self
        """

    def predict_proba(self, k: np.ndarray, n: np.ndarray) -> np.ndarray:
        """
        Compute P(accessible | k, n) for each read end.

        Returns array of shape (N,) with posterior probabilities.
        """

    def to_dict(self) -> dict:
        """Serialize fitted parameters to dict for JSON export."""

    @classmethod
    def from_dict(cls, d: dict) -> "BetaBinomialMixture":
        """Reconstruct fitted model from serialized dict."""
```

**Beta-binomial log-PMF implementation:**

```python
def betabinom_logpmf(k, n, alpha, beta):
    """
    Log probability mass function of the beta-binomial distribution.

    Uses log-Beta function for numerical stability:
    log P(k|n,α,β) = log C(n,k) + log B(k+α, n-k+β) - log B(α, β)

    where B(a,b) = Γ(a)Γ(b)/Γ(a+b)

    Implementation uses scipy.special.betaln for log B.
    """
    from scipy.special import betaln, gammaln
    return (
        gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        + betaln(k + alpha, n - k + beta)
        - betaln(alpha, beta)
    )
```

#### 4.3.4 Diagnostic: Three-Way Fit Comparison

Run the EM on three subsets of read ends:

| Fit | Read ends included |
|-----|--------------------|
| `all` | All read ends from all pairs |
| `short` | Only read ends from pairs with distance < `--short-range-max` |
| `long` | Only read ends from pairs with distance ≥ `--min-distance` |

**Comparison criteria:**
- Compute mean (μ_j = α_j / (α_j + β_j)) and overdispersion (φ_j = 1/(α_j + β_j + 1)) for each component
- Report parameter table side-by-side
- Flag if |Δμ| > 0.05 or |Δφ| > 0.1 between any pair of fits → warning to investigate technical biases

**Diagnostic plot:** Overlay the three fitted mixture densities against the observed (k/n) histogram, faceted by n (e.g., n=3, n=5, n=10, n=20).

#### 4.3.5 Subsampling Strategy

With hundreds of cells each contributing 1–5 million read pairs, the total number of read ends can reach 10⁹. The EM only estimates 5 parameters (π, α₁, β₁, α₂, β₂), so convergence is achieved with far fewer observations. Subsampling provides identical parameter estimates with dramatically faster fitting.

**Strategy:**
1. Pool all read ends across all cells (stacking both ends of each pair)
2. If total exceeds `--em-subsample` (default 10M), randomly subsample without replacement
3. Stratify the subsample proportionally by n (number of GCH sites per end) to preserve the marginal distribution of coverage depths
4. Verify: re-fit on two independent subsamples and confirm parameter agreement within tolerance

```python
def subsample_for_em(k_all: np.ndarray, n_all: np.ndarray,
                     max_samples: int = 10_000_000,
                     random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Subsample read ends for EM fitting, stratified by n.

    Preserves the marginal distribution of n (coverage per read end)
    to avoid biasing the mixture components toward high- or low-coverage ends.
    """
    if len(k_all) <= max_samples:
        return k_all, n_all

    rng = np.random.RandomState(random_state)

    # Stratify by n to preserve coverage distribution
    unique_n = np.unique(n_all)
    indices = []
    for n_val in unique_n:
        mask = n_all == n_val
        n_in_stratum = mask.sum()
        # Proportional allocation
        n_to_sample = max(1, int(max_samples * n_in_stratum / len(k_all)))
        stratum_indices = np.where(mask)[0]
        if n_to_sample < len(stratum_indices):
            chosen = rng.choice(stratum_indices, size=n_to_sample, replace=False)
        else:
            chosen = stratum_indices
        indices.append(chosen)

    indices = np.concatenate(indices)
    rng.shuffle(indices)

    logger.info(f"Subsampled {len(k_all):,} → {len(indices):,} read ends for EM fitting")
    return k_all[indices], n_all[indices]
```

#### 4.3.6 Cell-Specific vs. Pooled Parameter Comparison

With 1–5M reads per cell, each individual cell has enough data (~2–10M read ends) to fit the mixture model independently. This enables a critical diagnostic: are the mixture components consistent across cells, or do some cells have systematically different accessibility distributions?

**Rationale:** If the accessible and closed state parameters (α, β) vary across cells, this could indicate:
- Technical batch effects (library quality, bisulfite conversion efficiency)
- Genuine biological differences (e.g., a cell type with intermediate accessibility)
- Need for cell-specific posteriors rather than shared posteriors

**Implementation:**

```python
def fit_cell_specific_models(
    cell_parquet_paths: dict[str, str],
    pooled_model: BetaBinomialMixture,
    max_samples_per_cell: int = 1_000_000,
    n_threads: int = 4,
    output_dir: str = "output/03_mixture_model",
) -> pd.DataFrame:
    """
    Fit beta-binomial mixture independently per cell and compare to pooled.

    Returns DataFrame with per-cell parameters:
        cell_name, pi, alpha_closed, beta_closed, alpha_accessible, beta_accessible,
        mean_closed, mean_accessible, n_read_ends, converged
    """

    def _fit_one_cell(cell_name, parquet_path):
        df = pd.read_parquet(parquet_path)
        k_all = np.concatenate([df['k1'].values, df['k2'].values])
        n_all = np.concatenate([df['n1'].values, df['n2'].values])
        mask = n_all >= 1
        k_all, n_all = k_all[mask], n_all[mask]

        if len(k_all) > max_samples_per_cell:
            rng = np.random.RandomState(hash(cell_name) % 2**31)
            idx = rng.choice(len(k_all), max_samples_per_cell, replace=False)
            k_all, n_all = k_all[idx], n_all[idx]

        model = BetaBinomialMixture(max_iter=100, tol=1e-5, n_restarts=3)
        model.fit(k_all, n_all)

        return {
            'cell_name': cell_name,
            'n_read_ends': len(k_all),
            **model.to_dict(),
        }

    results = Parallel(n_jobs=n_threads)(
        delayed(_fit_one_cell)(name, path)
        for name, path in cell_parquet_paths.items()
    )

    cell_params = pd.DataFrame(results)
    cell_params.to_csv(
        Path(output_dir) / "cell_specific_mixture_params.tsv", sep='\t', index=False
    )

    # Generate diagnostic plot
    _plot_cell_vs_pooled(cell_params, pooled_model, output_dir)

    return cell_params


def _plot_cell_vs_pooled(cell_params, pooled_model, output_dir):
    """
    Diagnostic plot: cell-specific parameters vs. pooled.

    Four panels:
    1. Scatter: mean_accessible per cell (x) vs mean_closed per cell (y),
       with pooled estimate as crosshair
    2. Histogram: π (accessible fraction) across cells
    3. Scatter: overdispersion_accessible vs overdispersion_closed per cell
    4. Violin: mean_accessible grouped by group label (if available)
    """
    ...
```

**Decision logic:** After inspecting the diagnostic:

| Observation | Action |
|-------------|--------|
| Cell parameters tightly clustered around pooled | Use pooled model (default). Cell-specific fitting confirms stability. |
| A few outlier cells with divergent parameters | Investigate those cells for QC issues (low conversion, contamination). Exclude or flag. Use pooled model for remaining cells. |
| Systematic shift between groups (e.g., WT vs KO) | Still use pooled model for posterior computation — the component shapes are driven by nucleosome biology, not cell state. But the shift in π (mixing weight) is itself biologically interesting and should be reported. |
| Bimodal distribution of cell parameters | May indicate distinct library preparation batches or fundamentally different cell types. Consider fitting group-specific models. |

**Default recommendation:** Always use the pooled model for posterior computation (more stable, especially for cells with fewer reads). The cell-specific fitting is purely diagnostic. The posteriors from the pooled model will still correctly classify read ends for cells with shifted π because the component shapes (α, β) are shared biology.

---

### 4.4 `posterior.py` — Read End Classification

**Purpose:** Apply fitted mixture model to compute P(accessible | k_i, n_i) for every read end.

```python
def compute_posteriors(
    read_pairs_parquet: str,
    mixture_model: BetaBinomialMixture,
) -> pd.DataFrame:
    """
    Add posterior columns to read pair data.

    Adds columns:
        posterior1: P(accessible | k1, n1)
        posterior2: P(accessible | k2, n2)
    """
```

This is a simple vectorized operation — no fitting, just applying Bayes' rule with the pre-fitted parameters. Parallelized across cells.

---

### 4.5 `concordance.py` — Per-Bin Concordance

**Purpose:** For each cell and each genomic bin, compute the Pearson correlation of (posterior1, posterior2) across all qualifying pairs.

#### 4.5.1 Pair Assignment to Bins

A pair is assigned to a bin based on the **midpoint** of the two read ends:

```
bin = floor(((pos1 + pos2) / 2) / bin_size)
```

Alternative considered: assign to the bin of end 1 only, or assign to both bins. Midpoint is chosen because:
- Both ends contribute equally to the concordance signal
- Avoids double-counting
- For very large bins (5Mb), the exact assignment rule rarely matters

#### 4.5.2 Concordance Calculation

For each (cell, bin) with qualifying pairs, compute:

```python
def pearson_concordance(posterior1: np.ndarray, posterior2: np.ndarray) -> tuple[float, int]:
    """
    Compute Pearson correlation between posterior accessibility
    probabilities of paired read ends.

    Returns (r, n_pairs).
    r is set to NaN if n_pairs < min_pairs_per_bin or if
    standard deviation of either array is zero (all same state).
    """
    n = len(posterior1)
    if n < min_pairs:
        return (np.nan, n)

    # Handle edge case: if all posteriors on one side are identical
    # (e.g., all ~1.0), correlation is undefined → return NaN
    if np.std(posterior1) < 1e-10 or np.std(posterior2) < 1e-10:
        return (np.nan, n)

    r = np.corrcoef(posterior1, posterior2)[0, 1]
    return (r, n)
```

**Edge case — constant posteriors:** If all pairs in a bin have both ends highly accessible (posteriors ~1.0), the correlation is undefined (zero variance). This typically occurs in constitutively open regions (e.g., housekeeping promoter clusters). Setting these to NaN is correct — there is no *variation* to measure concordance on.

**Output:** Two matrices saved as parquet:
- `concordance_raw`: cells × bins matrix of Pearson r values (with NaN)
- `pair_counts_per_bin`: cells × bins matrix of n_pairs

**Threading:** Parallelize across cells. Within each cell, pairs are grouped by bin using pandas groupby, then the correlation is computed per group.

---

### 4.6 `distance_analysis.py` — Concordance vs. Distance Decay

**Purpose:** Characterize how GCH concordance decays with genomic distance. This is both a diagnostic (does the signal exist at all?) and a biological result (what distance scales carry concordance?).

**Analysis:**

For each cell (or pooled across cells), bin read pairs by genomic distance into log-spaced bins (e.g., 1kb–5kb, 5kb–20kb, 20kb–100kb, 100kb–500kb, 500kb–1Mb, 1Mb–5Mb, 5Mb–20Mb) and compute the Pearson correlation of (posterior₁, posterior₂) within each distance bin.

```python
def concordance_vs_distance(
    cell_posterior_paths: dict[str, str],
    distance_bins: list[tuple[int, int]] = None,
    min_gch_sites: int = 3,
    n_threads: int = 4,
    output_dir: str = "output/05_concordance",
) -> pd.DataFrame:
    """
    Compute concordance as a function of genomic distance.

    Default distance bins (log-spaced):
        [1000, 5000], [5000, 20000], [20000, 100000], [100000, 500000],
        [500000, 1000000], [1000000, 5000000], [5000000, 20000000]

    Returns DataFrame with columns:
        cell_name, dist_bin_start, dist_bin_end, r, n_pairs, dist_label

    Also generates:
    - Per-cell concordance vs distance curves (thin lines)
    - Mean + 95% CI across cells (thick line with ribbon)
    - Horizontal dashed line at r=0 (independence)
    """

    if distance_bins is None:
        distance_bins = [
            (1000, 5000), (5000, 20000), (20000, 100000),
            (100000, 500000), (500000, 1_000_000),
            (1_000_000, 5_000_000), (5_000_000, 20_000_000),
        ]

    def _process_cell(cell_name, path):
        df = pd.read_parquet(path)
        df = df[(df['n1'] >= min_gch_sites) & (df['n2'] >= min_gch_sites)]
        rows = []
        for d_start, d_end in distance_bins:
            mask = (df['distance'] >= d_start) & (df['distance'] < d_end)
            sub = df[mask]
            n_pairs = len(sub)
            if n_pairs >= 5 and sub['posterior1'].std() > 1e-10 and sub['posterior2'].std() > 1e-10:
                r = np.corrcoef(sub['posterior1'], sub['posterior2'])[0, 1]
            else:
                r = np.nan
            rows.append({
                'cell_name': cell_name,
                'dist_bin_start': d_start,
                'dist_bin_end': d_end,
                'r': r,
                'n_pairs': n_pairs,
                'dist_label': f'{d_start//1000}kb-{d_end//1000}kb',
            })
        return rows

    all_rows = Parallel(n_jobs=n_threads)(
        delayed(_process_cell)(name, path)
        for name, path in cell_posterior_paths.items()
    )

    result = pd.DataFrame([r for cell_rows in all_rows for r in cell_rows])
    result.to_csv(Path(output_dir) / "concordance_vs_distance.tsv", sep='\t', index=False)

    _plot_distance_decay(result, output_dir)
    return result


def _plot_distance_decay(df, output_dir):
    """
    Plot concordance vs distance with per-cell curves and population mean.

    X-axis: log-scale genomic distance (midpoint of each bin)
    Y-axis: Pearson r of (posterior₁, posterior₂)
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Per-cell thin lines
    for cell in df['cell_name'].unique():
        cell_df = df[df['cell_name'] == cell].sort_values('dist_bin_start')
        midpoints = np.sqrt(cell_df['dist_bin_start'] * cell_df['dist_bin_end'])
        ax.plot(midpoints, cell_df['r'], alpha=0.1, color='steelblue', linewidth=0.5)

    # Population mean + CI
    grouped = df.groupby('dist_bin_start').agg(
        r_mean=('r', 'mean'),
        r_std=('r', 'std'),
        r_count=('r', lambda x: x.notna().sum()),
        dist_bin_end=('dist_bin_end', 'first'),
    ).reset_index()
    midpoints = np.sqrt(grouped['dist_bin_start'] * grouped['dist_bin_end'])
    ci = 1.96 * grouped['r_std'] / np.sqrt(grouped['r_count'])

    ax.fill_between(midpoints, grouped['r_mean'] - ci, grouped['r_mean'] + ci,
                    alpha=0.3, color='darkblue')
    ax.plot(midpoints, grouped['r_mean'], 'o-', color='darkblue', linewidth=2,
            label='Mean ± 95% CI')

    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('Genomic distance (bp)')
    ax.set_ylabel('Concordance (Pearson r)')
    ax.set_title('Long-range chromatin accessibility concordance vs. distance')
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "concordance_vs_distance.png", dpi=150, bbox_inches='tight')
    plt.close()
```

**Expected behavior:** Concordance should be highest at short distances (within-TAD contacts, <100kb) and decay with increasing distance. A residual positive concordance at >1Mb suggests compartment-level coordination. If concordance is near zero at all distances, the signal may be too sparse for this resolution.

---

### 4.7 `shrinkage.py` — Empirical Bayes Shrinkage

**Purpose:** Determine the optimal shrinkage pseudocount n₀ and apply it.

#### 4.6.1 Empirical Bayes Derivation

Assume the true concordance r*_{c,b} for cell c in bin b is drawn from a prior:

```
r*_{c,b} ~ Normal(0, σ²_prior)
```

The observed r_{c,b} given r*_{c,b} has sampling variance approximately:

```
Var(r_{c,b} | r*_{c,b}) ≈ (1 - r*²)² / (n_{c,b} - 3) ≈ 1 / (n_{c,b} - 3)
```

(Using the delta method approximation for Pearson r, and assuming r* is not extreme.)

The Bayes-optimal shrinkage estimator is:

```
r̂_{c,b} = r_{c,b} × σ²_prior / (σ²_prior + 1/(n_{c,b} - 3))
```

Which simplifies to the form `r × n/(n + n₀)` where `n₀ = 1/σ²_prior` when we approximate the sampling variance as `1/n` (valid for moderate n).

#### 4.6.2 Estimating σ²_prior

Use the high-coverage bins to estimate the prior variance:

```python
def estimate_prior_variance(r_values: np.ndarray, n_pairs: np.ndarray,
                            n_min_for_estimation: int = 50) -> float:
    """
    Estimate σ²_prior from bins with sufficient coverage.

    1. Select (cell, bin) entries with n_pairs >= n_min_for_estimation
    2. Compute observed variance of r across these entries
    3. Subtract average sampling variance: E[1/(n-3)]
    4. σ²_prior = Var_observed - Var_sampling (floored at 0.01)

    Returns σ²_prior.
    """
```

The optimal n₀ is then `1 / σ²_prior`.

#### 4.6.3 Shrinkage Diagnostic Plot

Generate a plot with three panels:

1. **Variance vs n_pairs:** Scatter plot of Var(r) within bins of n_pairs (binned in groups of 10). Shows the expected 1/n decay with the asymptotic prior variance σ²_prior as horizontal line.

2. **Shrinkage curve:** Plot the multiplier n/(n + n₀) as a function of n, with vertical lines at n₀, 2×n₀, 5×n₀.

3. **Before/after distributions:** Histogram of r_raw vs r_shrunk, showing the compression toward zero for low-coverage bins.

```python
def shrinkage_analysis(
    concordance_raw: pd.DataFrame,    # cells × bins
    pair_counts: pd.DataFrame,        # cells × bins
    n_min_for_estimation: int = 50,
    output_dir: str = "output/05_concordance",
) -> dict:
    """
    Perform shrinkage analysis.

    Returns dict with:
        'n0': optimal pseudocount
        'sigma2_prior': estimated prior variance
        'n_high_coverage_entries': number used for estimation
    """
```

---

### 4.8 `matrix_ops.py` — Filtering, Imputation, Z-Scoring

#### 4.7.1 Processing Steps

```python
def process_concordance_matrix(
    concordance_shrunk: pd.DataFrame,  # cells × bins, with NaN
    min_cell_coverage: float = 0.8,
) -> dict[str, pd.DataFrame]:
    """
    Apply filtering, imputation, and z-scoring.

    Steps:
    1. Filter bins: retain only bins where fraction of non-NA cells >= min_cell_coverage
    2. Impute: fill remaining NAs with 0 (the shrinkage target)
    3. Z-score: per-bin standardization across cells

    Returns dict with keys:
        'retained_bins': list of bin identifiers passing filter
        'imputed': cells × bins after imputation (no z-score)
        'zscore': cells × bins after z-scoring
    """
```

#### 4.7.2 Z-Score Rationale

Z-scoring per bin removes the bin-level mean, so that constitutively concordant regions (e.g., large heterochromatin blocks) don't dominate the UMAP embedding. The clustering then reflects *relative* concordance differences between cells, not absolute levels.

Both versions (with and without z-score) are output because:
- **Without z-score:** Preserves absolute concordance; may separate cells by global chromatin state (e.g., quiescent vs. active)
- **With z-score:** Emphasizes cell-to-cell variation within each locus; may reveal finer substructure

---

### 4.9 `visualization.py` — UMAP and Diagnostics

```python
def run_umap_and_plot(
    matrix: pd.DataFrame,            # cells × bins
    group_labels: pd.Series | None,  # Optional group labels per cell
    title: str,
    output_path: str,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run UMAP on cells × bins matrix and generate scatter plot.

    Returns DataFrame with (cell, UMAP1, UMAP2, group).
    """
```

**UMAP considerations:**
- Uses `umap-learn` library
- Default metric: euclidean (appropriate for z-scored or bounded correlation values)
- Alternative: cosine distance (if cells have very different total coverage, cosine may be more robust)
- If group labels provided, color points by group; otherwise, color by total read count as a QC check

---

## 5. Threading and Parallelization Strategy

| Step | Parallelization unit | Method |
|------|---------------------|--------|
| GCH extraction | Per chromosome | `joblib.Parallel(n_jobs=n_threads)` |
| BAM parsing | Per cell | `joblib.Parallel(n_jobs=n_threads)` |
| EM fitting | Within M-step optimization | `scipy.optimize` internal parallelism |
| Posterior computation | Per cell | `joblib.Parallel(n_jobs=n_threads)` |
| Concordance computation | Per cell | `joblib.Parallel(n_jobs=n_threads)` |
| UMAP | Internal (uses `n_jobs`) | `umap.UMAP(n_jobs=n_threads)` |

Most steps are embarrassingly parallel across cells, which is the dominant cost for large datasets (hundreds to thousands of cells).

**Memory consideration:** Each cell's BAM may be large. With `n_threads` workers each holding a BAM file handle, memory usage scales as O(n_threads × per_cell_memory). For cells with high contact counts, this could be significant. The BAM parser should process reads in a streaming fashion (sorted order traversal) rather than loading all pairs into memory.

---

## 6. CLI Interface

```
usage: nome_hic_concordance [-h] --samples SAMPLES --reference REFERENCE
                            --output-dir OUTPUT_DIR
                            [--min-distance MIN_DISTANCE]
                            [--min-gch-sites MIN_GCH_SITES]
                            [--bin-size BIN_SIZE]
                            [--min-pairs-per-bin MIN_PAIRS_PER_BIN]
                            [--min-cell-coverage MIN_CELL_COVERAGE]
                            [--n-threads N_THREADS]
                            [--em-max-iter EM_MAX_ITER]
                            [--em-tol EM_TOL]
                            [--short-range-max SHORT_RANGE_MAX]
                            [--shrinkage-n0 SHRINKAGE_N0]
                            [--dbsnp-vcf DBSNP_VCF]
                            [--em-subsample EM_SUBSAMPLE]
                            [--em-cell-specific]
                            [--trim-ends TRIM_ENDS]
                            [--mapq-min MAPQ_MIN]
                            [--steps STEPS]

NOMe-HiC Long-Range Chromatin Accessibility Concordance Pipeline

required arguments:
  --samples SAMPLES       Path to sample manifest (TSV)
  --reference REFERENCE   Path to .2bit reference genome
  --output-dir OUTPUT_DIR Path to output directory

optional arguments:
  --steps STEPS           Comma-separated list of steps to run
                          (1,2,3,4,5,6,7,8 or "all"). Allows resuming
                          from intermediate outputs.
  --trim-ends TRIM_ENDS   Bases to trim from read ends (default: 5)
  --mapq-min MAPQ_MIN     Minimum mapping quality (default: 20)
  --dbsnp-vcf DBSNP_VCF   Path to tabix-indexed dbSNP VCF for SNP filtering
  --em-subsample N        Max read ends for EM fitting (default: 10000000)
  --em-cell-specific      Also fit EM per-cell for diagnostic comparison
```

---

## 7. Code Plan — Implementation Order

### Phase 1: Core Data Extraction (reference.py, bam_parser.py)

**File: `reference.py`**

Full implementation with IUPAC matching, dual-strand extraction, and dbSNP filtering — see Section 4.1 for data structure specification.

```python
import numpy as np
import pysam
from pathlib import Path
import py2bit
from joblib import Parallel, delayed
import logging

logger = logging.getLogger(__name__)

# IUPAC ambiguity codes
IUPAC = {
    'A': {'A'}, 'C': {'C'}, 'G': {'G'}, 'T': {'T'},
    'H': {'A', 'C', 'T'}, 'W': {'A', 'T'}, 'S': {'C', 'G'},
    'R': {'A', 'G'}, 'Y': {'C', 'T'}, 'K': {'G', 'T'},
    'M': {'A', 'C'}, 'B': {'C', 'G', 'T'}, 'D': {'A', 'G', 'T'},
    'V': {'A', 'C', 'G'}, 'N': {'A', 'C', 'G', 'T'},
}

IUPAC_COMP = {
    'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
    'H': 'D', 'D': 'H', 'W': 'W', 'S': 'S',
    'R': 'Y', 'Y': 'R', 'K': 'M', 'M': 'K',
    'B': 'V', 'V': 'B', 'N': 'N',
}


def _matches_context(tri: str, context: str) -> bool:
    """Check if a 3-base trinucleotide matches a context pattern."""
    return all(base in IUPAC[code] for base, code in zip(tri, context))


def _load_snp_positions(vcf_path: str, chrom: str) -> set[int]:
    """Load SNP positions for a chromosome from a tabix-indexed VCF."""
    snp_positions = set()
    try:
        vcf = pysam.VariantFile(vcf_path)
        for record in vcf.fetch(chrom):
            snp_positions.add(record.pos - 1)  # 0-based
        vcf.close()
    except (ValueError, StopIteration):
        pass
    return snp_positions


def _extract_chrom_contexts(twobit_path, chrom, contexts, snp_positions=None):
    """
    Extract context positions for one chromosome, both strands.
    Excludes GCG as ambiguous for all contexts.

    Returns dict: context → structured array with 'pos' (int64) and 'ref_base' (uint8).
    'pos' is 0-based on + strand reference.
    'ref_base' is ord('C') for + strand sites, ord('G') for − strand sites.
    """
    tb = py2bit.open(twobit_path)
    seq = tb.sequence(chrom).upper()
    tb.close()

    results = {}
    for context in contexts:
        positions = []
        ref_bases = []

        # + strand scan
        for i in range(len(seq) - 2):
            tri = seq[i:i + 3]
            if not _matches_context(tri, context):
                continue
            c_pos = i + 1  # C position in trinucleotide
            # Exclude GCG (ambiguous between GCH and HCG)
            if seq[i:i + 3] == 'GCG':
                continue
            if snp_positions and any(p in snp_positions for p in (i, i+1, i+2)):
                continue
            positions.append(c_pos)
            ref_bases.append(ord('C'))

        # − strand scan: reverse-complement context on + strand
        minus_ctx = ''.join(IUPAC_COMP[c] for c in reversed(context))
        for i in range(len(seq) - 2):
            tri = seq[i:i + 3]
            if not _matches_context(tri, minus_ctx):
                continue
            g_pos = i + 1  # G on + strand = C on − strand
            if seq[i:i + 3] == 'CGC':  # complement of GCG
                continue
            if snp_positions and any(p in snp_positions for p in (i, i+1, i+2)):
                continue
            positions.append(g_pos)
            ref_bases.append(ord('G'))

        # Sort and deduplicate by position
        if positions:
            arr = np.array(list(zip(positions, ref_bases)),
                           dtype=[('pos', np.int64), ('ref_base', np.uint8)])
            arr.sort(order='pos')
            results[context] = arr
        else:
            results[context] = np.array([],
                           dtype=[('pos', np.int64), ('ref_base', np.uint8)])

    return chrom, results


def extract_context_positions(twobit_path, output_dir, contexts=None,
                              dbsnp_vcf=None, n_threads=4):
    """
    Extract cytosine context positions from .2bit reference, both strands.

    Parameters
    ----------
    contexts : list of str
        IUPAC context patterns. Default: ['GCH'].
        Valid examples: 'GCH', 'HCG', 'WCG'.

    Returns
    -------
    dict: context → dict: chrom → structured array with 'pos', 'ref_base'.
    """
    if contexts is None:
        contexts = ['GCH']

    tb = py2bit.open(twobit_path)
    chroms = [c for c in tb.chroms()
              if (c.startswith('chr') and c[3:].isdigit()) or c in ('chrX', 'chrY')]
    tb.close()

    snp_per_chrom = {}
    if dbsnp_vcf:
        logger.info(f"Loading dbSNP positions from {dbsnp_vcf}...")
        for c in chroms:
            snp_per_chrom[c] = _load_snp_positions(dbsnp_vcf, c)

    logger.info(f"Extracting contexts {contexts} from {len(chroms)} chromosomes (both strands)...")

    results = Parallel(n_jobs=n_threads)(
        delayed(_extract_chrom_contexts)(
            twobit_path, c, contexts, snp_per_chrom.get(c)
        ) for c in chroms
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_positions = {ctx: {} for ctx in contexts}
    for chrom, ctx_results in results:
        for ctx, arr in ctx_results.items():
            all_positions[ctx][chrom] = arr
            np.savez_compressed(
                out_path / f"context_{ctx}.{chrom}.npz",
                pos=arr['pos'], ref_base=arr['ref_base']
            )
            logger.info(f"  {chrom} {ctx}: {len(arr):,} sites (both strands)")

    return all_positions
```

**File: `bam_parser.py`**

```python
import pysam
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
import logging

logger = logging.getLogger(__name__)


def _count_context_in_read(read, context_sites_chrom,
                           trim_ends: int = 5) -> tuple[int, int]:
    """
    Count methylated and total context sites in a single read.

    Uses reference-comparison approach (no bisulfite strand tag needed).
    For each context site:
      - If ref_base='C' (+ strand site): read C=methylated, T=unmethylated, else skip
      - If ref_base='G' (− strand site): read G=methylated, A=unmethylated, else skip

    Parameters
    ----------
    read : pysam.AlignedSegment
    context_sites_chrom : structured array with fields 'pos' and 'ref_base'
    trim_ends : bases to exclude at each end of the read

    Returns
    -------
    (k_methylated, n_total)
    """
    start = read.reference_start + trim_ends
    end = read.reference_end - trim_ends
    if start >= end:
        return (0, 0)

    # Find context sites within trimmed read span
    positions = context_sites_chrom['pos']
    idx_start = np.searchsorted(positions, start, side='left')
    idx_end = np.searchsorted(positions, end, side='right')
    sites_in_read = context_sites_chrom[idx_start:idx_end]

    if len(sites_in_read) == 0:
        return (0, 0)

    # Build dict of aligned pairs: ref_pos → query_pos
    aligned_pairs = dict(read.get_aligned_pairs(matches_only=True))

    k = 0
    n = 0

    for site in sites_in_read:
        ref_pos = int(site['pos'])
        ref_base = chr(site['ref_base'])  # 'C' or 'G'
        query_pos = aligned_pairs.get(ref_pos)
        if query_pos is None:
            continue

        read_base = read.query_sequence[query_pos]

        if ref_base == 'C':
            # + strand context site: C→T conversion
            if read_base == 'C':
                k += 1  # Methylated
                n += 1
            elif read_base == 'T':
                n += 1  # Unmethylated
            # G, A, N → skip (read from opposite conversion strand or error)
        elif ref_base == 'G':
            # − strand context site: G→A conversion
            if read_base == 'G':
                k += 1  # Methylated
                n += 1
            elif read_base == 'A':
                n += 1  # Unmethylated
            # C, T, N → skip

    return (k, n)

    return (k, n)


def parse_cell_bam(bam_path: str, context_positions: dict[str, np.ndarray],
                   trim_ends: int = 5, mapq_min: int = 20,
                   short_range_max: int = 1000,
                   min_distance: int = 20000) -> pd.DataFrame:
    """
    Parse a single cell's BAM file, extract context methylation counts per read pair end.

    Uses reference-comparison approach: no bisulfite strand tag needed.
    Approximately half of reads at each site will be informative (from
    the matching conversion strand); the other half are automatically
    skipped by the C/T or G/A base check.

    Parameters
    ----------
    bam_path : str
        Path to coordinate-sorted, indexed BAM (from bhmem).
    context_positions : dict
        Chromosome → structured array with 'pos' and 'ref_base' fields.
    trim_ends : int
        Bases to trim from read ends.
    mapq_min : int
        Minimum mapping quality.

    Returns
    -------
    DataFrame with columns:
        chrom, pos1, k1, n1, pos2, k2, n2, distance
    """
    records = []

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        read_buffer = {}

        for read in bam.fetch():
            if read.is_unmapped or read.mate_is_unmapped:
                continue
            if read.mapping_quality < mapq_min:
                continue
            if not read.is_proper_pair:
                continue
            if read.is_secondary or read.is_supplementary:
                continue
            if read.reference_name != read.next_reference_name:
                continue  # Intra-chromosomal only

            qname = read.query_name
            if qname in read_buffer:
                mate = read_buffer.pop(qname)
                if read.reference_start <= mate.reference_start:
                    r1, r2 = read, mate
                else:
                    r1, r2 = mate, read

                chrom = r1.reference_name
                if chrom not in context_positions:
                    continue

                ctx_chrom = context_positions[chrom]

                k1, n1 = _count_context_in_read(r1, ctx_chrom, trim_ends)
                k2, n2 = _count_context_in_read(r2, ctx_chrom, trim_ends)

                pos1 = (r1.reference_start + r1.reference_end) // 2
                pos2 = (r2.reference_start + r2.reference_end) // 2
                distance = abs(pos2 - pos1)

                records.append({
                    'chrom': chrom,
                    'pos1': pos1, 'k1': k1, 'n1': n1,
                    'pos2': pos2, 'k2': k2, 'n2': n2,
                    'distance': distance,
                })
            else:
                read_buffer[qname] = read

    df = pd.DataFrame(records)
    logger.info(f"  {bam_path}: {len(df):,} intra-chromosomal pairs extracted")
    return df


def parse_all_cells(samples: pd.DataFrame, context_positions: dict,
                    output_dir: str, n_threads: int = 4,
                    **kwargs) -> dict[str, str]:
    """
    Parse all cell BAMs in parallel.

    Parameters
    ----------
    samples : DataFrame with columns [cell_name, bam_path, group (optional)]
    context_positions : dict from reference.extract_context_positions (single context)
    output_dir : str
    n_threads : int

    Returns
    -------
    dict mapping cell_name to parquet output path.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _process_one(row):
        cell = row['cell_name']
        bam = row['bam_path']
        logger.info(f"Processing {cell}...")
        df = parse_cell_bam(bam, context_positions, **kwargs)
        parquet_path = str(out_path / f"{cell}.read_pairs.parquet")
        df.to_parquet(parquet_path, index=False)
        return cell, parquet_path

    results = Parallel(n_jobs=n_threads)(
        delayed(_process_one)(row) for _, row in samples.iterrows()
    )

    return dict(results)
```

### Phase 2: Mixture Model (mixture_model.py, posterior.py)

**File: `mixture_model.py`**

```python
import numpy as np
from scipy.special import betaln, gammaln
from scipy.optimize import minimize
import json
import logging

logger = logging.getLogger(__name__)


def betabinom_logpmf(k, n, alpha, beta_param):
    """
    Vectorized log-PMF of the beta-binomial distribution.

    P(k|n,α,β) = C(n,k) × B(k+α, n-k+β) / B(α,β)

    All inputs can be arrays (broadcast-compatible).
    """
    return (
        gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        + betaln(k + alpha, n - k + beta_param)
        - betaln(alpha, beta_param)
    )


class BetaBinomialMixture:
    """Two-component beta-binomial mixture model fitted via EM."""

    def __init__(self, max_iter=200, tol=1e-6, n_restarts=5, random_state=42):
        self.max_iter = max_iter
        self.tol = tol
        self.n_restarts = n_restarts
        self.random_state = random_state

        # Fitted parameters: component 0 = closed, component 1 = accessible
        self.pi_ = None
        self.alpha_ = None       # [α_closed, α_accessible]
        self.beta_ = None        # [β_closed, β_accessible]
        self.converged_ = False
        self.n_iter_ = 0
        self.log_likelihood_ = None

    def _e_step(self, k, n):
        """Compute responsibilities γ_i = P(accessible | k_i, n_i)."""
        log_p0 = betabinom_logpmf(k, n, self.alpha_[0], self.beta_[0])
        log_p1 = betabinom_logpmf(k, n, self.alpha_[1], self.beta_[1])

        log_w0 = np.log(1 - self.pi_) + log_p0
        log_w1 = np.log(self.pi_) + log_p1

        # Log-sum-exp for numerical stability
        log_total = np.logaddexp(log_w0, log_w1)
        gamma = np.exp(log_w1 - log_total)

        return gamma, np.sum(log_total)

    def _m_step_component(self, k, n, weights, alpha_init, beta_init):
        """Optimize (α, β) for one component given responsibilities."""

        def neg_weighted_ll(params):
            a, b = params
            if a <= 0 or b <= 0:
                return 1e10
            ll = np.sum(weights * betabinom_logpmf(k, n, a, b))
            return -ll

        result = minimize(
            neg_weighted_ll,
            x0=[alpha_init, beta_init],
            method='L-BFGS-B',
            bounds=[(1e-4, 500), (1e-4, 500)],
        )

        return result.x[0], result.x[1]

    def _fit_once(self, k, n, pi_init, alpha_init, beta_init):
        """Single EM run from given initialization."""
        self.pi_ = pi_init
        self.alpha_ = alpha_init.copy()
        self.beta_ = beta_init.copy()

        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            # E-step
            gamma, ll = self._e_step(k, n)

            # Check convergence
            if abs(ll - prev_ll) < self.tol:
                return ll, iteration + 1, True
            prev_ll = ll

            # M-step: update mixing weight
            self.pi_ = np.mean(gamma)

            # M-step: update component 1 (accessible)
            self.alpha_[1], self.beta_[1] = self._m_step_component(
                k, n, gamma, self.alpha_[1], self.beta_[1]
            )

            # M-step: update component 0 (closed)
            self.alpha_[0], self.beta_[0] = self._m_step_component(
                k, n, 1 - gamma, self.alpha_[0], self.beta_[0]
            )

        return prev_ll, self.max_iter, False

    def fit(self, k: np.ndarray, n: np.ndarray) -> "BetaBinomialMixture":
        """
        Fit two-component beta-binomial mixture via EM with restarts.

        Parameters
        ----------
        k : array (N,) — methylated GCH counts per read end
        n : array (N,) — total GCH counts per read end (>= 1)
        """
        rng = np.random.RandomState(self.random_state)

        # Filter to valid entries
        mask = n >= 1
        k, n = k[mask].astype(float), n[mask].astype(float)

        logger.info(f"Fitting beta-binomial mixture on {len(k):,} read ends...")

        best_ll = -np.inf
        best_params = None

        base_inits = {
            'pi': 0.3,
            'alpha': np.array([2.0, 8.0]),
            'beta': np.array([8.0, 2.0]),
        }

        for restart in range(self.n_restarts):
            # Perturb initialization
            if restart == 0:
                pi_init = base_inits['pi']
                alpha_init = base_inits['alpha'].copy()
                beta_init = base_inits['beta'].copy()
            else:
                pi_init = np.clip(base_inits['pi'] + rng.uniform(-0.15, 0.15), 0.05, 0.95)
                alpha_init = base_inits['alpha'] * rng.uniform(0.5, 2.0, size=2)
                beta_init = base_inits['beta'] * rng.uniform(0.5, 2.0, size=2)

            ll, n_iter, converged = self._fit_once(k, n, pi_init, alpha_init, beta_init)

            logger.info(f"  Restart {restart}: LL={ll:.2f}, iters={n_iter}, converged={converged}")

            if ll > best_ll:
                best_ll = ll
                best_params = {
                    'pi': self.pi_,
                    'alpha': self.alpha_.copy(),
                    'beta': self.beta_.copy(),
                    'n_iter': n_iter,
                    'converged': converged,
                }

        # Restore best parameters
        self.pi_ = best_params['pi']
        self.alpha_ = best_params['alpha']
        self.beta_ = best_params['beta']
        self.n_iter_ = best_params['n_iter']
        self.converged_ = best_params['converged']
        self.log_likelihood_ = best_ll

        # Ensure component 0 is "closed" (lower mean) and component 1 is "accessible"
        mean0 = self.alpha_[0] / (self.alpha_[0] + self.beta_[0])
        mean1 = self.alpha_[1] / (self.alpha_[1] + self.beta_[1])
        if mean0 > mean1:
            # Swap components
            self.alpha_ = self.alpha_[::-1]
            self.beta_ = self.beta_[::-1]
            self.pi_ = 1 - self.pi_
            logger.info("  Swapped components to ensure component 1 = accessible.")

        self._log_summary()
        return self

    def predict_proba(self, k: np.ndarray, n: np.ndarray) -> np.ndarray:
        """Compute P(accessible | k, n) for each read end."""
        gamma, _ = self._e_step(k.astype(float), n.astype(float))
        return gamma

    def _log_summary(self):
        for j, label in enumerate(['Closed', 'Accessible']):
            mean = self.alpha_[j] / (self.alpha_[j] + self.beta_[j])
            od = 1 / (self.alpha_[j] + self.beta_[j] + 1)
            logger.info(f"  {label}: α={self.alpha_[j]:.3f}, β={self.beta_[j]:.3f}, "
                        f"mean={mean:.3f}, overdispersion={od:.4f}")
        logger.info(f"  π(accessible)={self.pi_:.3f}, LL={self.log_likelihood_:.2f}")

    def to_dict(self) -> dict:
        return {
            'pi': float(self.pi_),
            'alpha_closed': float(self.alpha_[0]),
            'beta_closed': float(self.beta_[0]),
            'alpha_accessible': float(self.alpha_[1]),
            'beta_accessible': float(self.beta_[1]),
            'mean_closed': float(self.alpha_[0] / (self.alpha_[0] + self.beta_[0])),
            'mean_accessible': float(self.alpha_[1] / (self.alpha_[1] + self.beta_[1])),
            'overdispersion_closed': float(1 / (self.alpha_[0] + self.beta_[0] + 1)),
            'overdispersion_accessible': float(1 / (self.alpha_[1] + self.beta_[1] + 1)),
            'log_likelihood': float(self.log_likelihood_),
            'n_iter': int(self.n_iter_),
            'converged': bool(self.converged_),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BetaBinomialMixture":
        obj = cls()
        obj.pi_ = d['pi']
        obj.alpha_ = np.array([d['alpha_closed'], d['alpha_accessible']])
        obj.beta_ = np.array([d['beta_closed'], d['beta_accessible']])
        obj.log_likelihood_ = d['log_likelihood']
        obj.n_iter_ = d['n_iter']
        obj.converged_ = d['converged']
        return obj


def fit_and_compare(
    all_read_ends: pd.DataFrame,     # Must have columns k, n, distance
    short_range_max: int = 1000,
    min_distance: int = 20000,
    output_dir: str = "output/03_mixture_model",
    max_subsample: int = 10_000_000,
    **em_kwargs,
) -> dict:
    """
    Fit beta-binomial mixture on three subsets and compare parameters.
    Subsamples if total read ends exceed max_subsample.

    Returns dict with fitted models and comparison statistics.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    subsets = {
        'all': all_read_ends,
        'short': all_read_ends[all_read_ends['distance'] < short_range_max],
        'long': all_read_ends[all_read_ends['distance'] >= min_distance],
    }

    results = {}
    for name, df in subsets.items():
        # Stack both ends
        k_all = np.concatenate([df['k1'].values, df['k2'].values])
        n_all = np.concatenate([df['n1'].values, df['n2'].values])

        # Filter to ends with ≥1 GCH site
        mask = n_all >= 1
        k_all, n_all = k_all[mask], n_all[mask]

        # Subsample if needed
        k_all, n_all = subsample_for_em(k_all, n_all, max_subsample)

        logger.info(f"\nFitting mixture on '{name}' subset ({len(k_all):,} read ends)...")
        model = BetaBinomialMixture(**em_kwargs)
        model.fit(k_all, n_all)

        results[name] = model

        with open(Path(output_dir) / f"em_fit_{name}.json", 'w') as f:
            json.dump(model.to_dict(), f, indent=2)

    # Compare parameters
    comparison = {}
    for name, model in results.items():
        comparison[name] = {
            'mean_closed': model.alpha_[0] / (model.alpha_[0] + model.beta_[0]),
            'mean_accessible': model.alpha_[1] / (model.alpha_[1] + model.beta_[1]),
            'od_closed': 1 / (model.alpha_[0] + model.beta_[0] + 1),
            'od_accessible': 1 / (model.alpha_[1] + model.beta_[1] + 1),
            'pi': model.pi_,
        }

    # Check for divergence
    for param in ['mean_closed', 'mean_accessible']:
        vals = [comparison[s][param] for s in ['all', 'short', 'long']]
        max_diff = max(vals) - min(vals)
        if max_diff > 0.05:
            logger.warning(f"  WARNING: {param} differs by {max_diff:.3f} across subsets. "
                           f"Investigate potential technical biases between short/long-range pairs.")

    return {'models': results, 'comparison': comparison}
```

**File: `posterior.py`**

```python
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
import logging

logger = logging.getLogger(__name__)


def compute_cell_posteriors(
    parquet_path: str,
    mixture_model,  # BetaBinomialMixture
    output_dir: str,
    cell_name: str,
) -> str:
    """
    Compute P(accessible | k, n) for both ends of every pair in one cell.
    """
    df = pd.read_parquet(parquet_path)

    df['posterior1'] = mixture_model.predict_proba(
        df['k1'].values, df['n1'].values
    )
    df['posterior2'] = mixture_model.predict_proba(
        df['k2'].values, df['n2'].values
    )

    out_path = Path(output_dir) / f"{cell_name}.posteriors.parquet"
    df.to_parquet(out_path, index=False)
    return str(out_path)


def compute_all_posteriors(
    cell_parquet_paths: dict[str, str],   # cell_name → parquet path
    mixture_model,
    output_dir: str,
    n_threads: int = 4,
) -> dict[str, str]:
    """Compute posteriors for all cells in parallel."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = Parallel(n_jobs=n_threads)(
        delayed(compute_cell_posteriors)(
            path, mixture_model, output_dir, cell_name
        )
        for cell_name, path in cell_parquet_paths.items()
    )

    return dict(zip(cell_parquet_paths.keys(), results))
```

### Phase 3: Concordance, Shrinkage, Matrix (concordance.py, shrinkage.py, matrix_ops.py)

**File: `concordance.py`**

```python
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
import logging

logger = logging.getLogger(__name__)


def _compute_cell_concordance(
    posteriors_path: str,
    cell_name: str,
    bin_size: int,
    min_distance: int,
    min_gch_sites: int,
    min_pairs_per_bin: int,
    chrom_sizes: dict[str, int],
) -> tuple[str, dict[str, float], dict[str, int]]:
    """
    Compute per-bin concordance for a single cell.

    Returns
    -------
    (cell_name, concordance_dict, pair_count_dict)
    where keys are bin identifiers like "chr1:0-5000000"
    """
    df = pd.read_parquet(posteriors_path)

    # Filter: distance and GCH site thresholds
    df = df[
        (df['distance'] >= min_distance) &
        (df['n1'] >= min_gch_sites) &
        (df['n2'] >= min_gch_sites)
    ].copy()

    if len(df) == 0:
        return cell_name, {}, {}

    # Assign each pair to a bin based on midpoint
    df['midpoint'] = ((df['pos1'] + df['pos2']) / 2).astype(int)
    df['bin_start'] = (df['midpoint'] // bin_size) * bin_size
    df['bin_id'] = df['chrom'] + ':' + df['bin_start'].astype(str)

    concordance = {}
    pair_counts = {}

    for bin_id, group in df.groupby('bin_id'):
        n_pairs = len(group)
        pair_counts[bin_id] = n_pairs

        if n_pairs < min_pairs_per_bin:
            concordance[bin_id] = np.nan
            continue

        p1 = group['posterior1'].values
        p2 = group['posterior2'].values

        # Check for zero variance
        if np.std(p1) < 1e-10 or np.std(p2) < 1e-10:
            concordance[bin_id] = np.nan
            continue

        r = np.corrcoef(p1, p2)[0, 1]
        concordance[bin_id] = r

    return cell_name, concordance, pair_counts


def compute_concordance_matrix(
    cell_posterior_paths: dict[str, str],
    chrom_sizes: dict[str, int],
    bin_size: int = 5_000_000,
    min_distance: int = 20_000,
    min_gch_sites: int = 3,
    min_pairs_per_bin: int = 5,
    n_threads: int = 4,
    output_dir: str = "output/05_concordance",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute concordance matrix for all cells.

    Returns
    -------
    (concordance_df, pair_counts_df): both cells × bins DataFrames.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = Parallel(n_jobs=n_threads)(
        delayed(_compute_cell_concordance)(
            path, cell_name, bin_size, min_distance,
            min_gch_sites, min_pairs_per_bin, chrom_sizes
        )
        for cell_name, path in cell_posterior_paths.items()
    )

    # Assemble into DataFrames
    all_bins = sorted(set(
        b for _, conc, _ in results for b in conc.keys()
    ))

    concordance_data = {}
    pair_count_data = {}
    for cell_name, conc, counts in results:
        concordance_data[cell_name] = {b: conc.get(b, np.nan) for b in all_bins}
        pair_count_data[cell_name] = {b: counts.get(b, 0) for b in all_bins}

    concordance_df = pd.DataFrame(concordance_data).T  # cells × bins
    pair_counts_df = pd.DataFrame(pair_count_data).T

    concordance_df.columns.name = 'bin'
    concordance_df.index.name = 'cell'
    pair_counts_df.columns.name = 'bin'
    pair_counts_df.index.name = 'cell'

    # Save
    concordance_df.to_parquet(Path(output_dir) / "concordance_raw.parquet")
    pair_counts_df.to_parquet(Path(output_dir) / "pair_counts_per_bin.parquet")
    concordance_df.to_csv(Path(output_dir) / "concordance_raw.tsv", sep='\t')
    pair_counts_df.to_csv(Path(output_dir) / "pair_counts_per_bin.tsv", sep='\t')

    logger.info(f"Concordance matrix: {concordance_df.shape[0]} cells × {concordance_df.shape[1]} bins")

    return concordance_df, pair_counts_df
```

**File: `shrinkage.py`**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


def estimate_shrinkage_n0(
    concordance_raw: pd.DataFrame,
    pair_counts: pd.DataFrame,
    n_min_for_estimation: int = 50,
    output_dir: str = "output/05_concordance",
) -> dict:
    """
    Empirical Bayes estimation of optimal shrinkage pseudocount n₀.

    Method:
    1. Select entries with n_pairs >= n_min_for_estimation
    2. Compute Var(r) observed across these high-coverage entries
    3. Subtract average sampling variance E[1/(n-3)]
    4. σ²_prior = Var_observed - Var_sampling (floor at 0.01)
    5. n₀ = 1 / σ²_prior

    Returns dict with n₀, σ²_prior, and diagnostic info.
    """
    out_path = Path(output_dir)

    # Flatten matrices to vectors
    r_flat = concordance_raw.values.ravel()
    n_flat = pair_counts.values.ravel()

    # Remove NaN entries
    valid = ~np.isnan(r_flat) & (n_flat >= 3)  # Need n >= 3 for Pearson r
    r_valid = r_flat[valid]
    n_valid = n_flat[valid]

    # High-coverage subset for prior estimation
    high_cov = n_valid >= n_min_for_estimation
    r_high = r_valid[high_cov]
    n_high = n_valid[high_cov]

    if len(r_high) < 20:
        logger.warning("Too few high-coverage entries for reliable σ²_prior estimation. "
                       "Using default n₀=30.")
        n0 = 30.0
        sigma2_prior = 1.0 / n0
    else:
        var_observed = np.var(r_high)
        var_sampling = np.mean(1.0 / (n_high - 3))
        sigma2_prior = max(var_observed - var_sampling, 0.01)
        n0 = 1.0 / sigma2_prior

    logger.info(f"Shrinkage analysis:")
    logger.info(f"  High-coverage entries (n >= {n_min_for_estimation}): {len(r_high)}")
    logger.info(f"  σ²_prior = {sigma2_prior:.4f}")
    logger.info(f"  Optimal n₀ = {n0:.1f}")

    result = {
        'n0': float(n0),
        'sigma2_prior': float(sigma2_prior),
        'n_high_coverage_entries': int(len(r_high)),
        'n_min_for_estimation': n_min_for_estimation,
    }

    with open(out_path / "shrinkage_n0.json", 'w') as f:
        json.dump(result, f, indent=2)

    # Generate diagnostic plot
    _plot_shrinkage_diagnostics(r_valid, n_valid, n0, sigma2_prior, out_path)

    return result


def apply_shrinkage(
    concordance_raw: pd.DataFrame,
    pair_counts: pd.DataFrame,
    n0: float,
) -> pd.DataFrame:
    """
    Apply shrinkage: r_shrunk = r_raw × n / (n + n₀).

    NaN entries remain NaN.
    """
    multiplier = pair_counts.values / (pair_counts.values + n0)
    shrunk = concordance_raw.values * multiplier

    # Preserve NaN from original
    shrunk[np.isnan(concordance_raw.values)] = np.nan

    return pd.DataFrame(
        shrunk,
        index=concordance_raw.index,
        columns=concordance_raw.columns,
    )


def _plot_shrinkage_diagnostics(r_valid, n_valid, n0, sigma2_prior, out_path):
    """Generate three-panel shrinkage diagnostic plot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Variance vs n_pairs
    ax = axes[0]
    bin_edges = np.arange(5, np.percentile(n_valid, 95), 10)
    bin_vars = []
    bin_centers = []
    for i in range(len(bin_edges) - 1):
        mask = (n_valid >= bin_edges[i]) & (n_valid < bin_edges[i + 1])
        if np.sum(mask) >= 5:
            bin_vars.append(np.var(r_valid[mask]))
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)

    ax.scatter(bin_centers, bin_vars, s=20, alpha=0.7)
    x_theory = np.linspace(5, max(bin_centers), 100)
    ax.plot(x_theory, sigma2_prior + 1.0 / (x_theory - 3),
            'r-', label=f'σ²_prior + 1/(n-3)')
    ax.axhline(sigma2_prior, color='gray', linestyle='--',
               label=f'σ²_prior = {sigma2_prior:.4f}')
    ax.set_xlabel('Number of pairs in bin')
    ax.set_ylabel('Variance of r')
    ax.set_title('Variance vs. pair count')
    ax.legend(fontsize=8)

    # Panel 2: Shrinkage curve
    ax = axes[1]
    n_range = np.arange(1, 200)
    shrink_factor = n_range / (n_range + n0)
    ax.plot(n_range, shrink_factor, 'b-')
    ax.axvline(n0, color='r', linestyle='--', label=f'n₀ = {n0:.0f}')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Number of pairs')
    ax.set_ylabel('Shrinkage multiplier n/(n + n₀)')
    ax.set_title('Shrinkage strength')
    ax.legend()

    # Panel 3: Before/after distribution
    ax = axes[2]
    r_shrunk = r_valid * n_valid / (n_valid + n0)
    ax.hist(r_valid, bins=50, alpha=0.5, label='Raw r', density=True)
    ax.hist(r_shrunk, bins=50, alpha=0.5, label='Shrunk r', density=True)
    ax.set_xlabel('Concordance (r)')
    ax.set_ylabel('Density')
    ax.set_title('Raw vs. shrunk distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path / "shrinkage_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
```

**File: `matrix_ops.py`**

```python
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def process_concordance_matrix(
    concordance_shrunk: pd.DataFrame,
    min_cell_coverage: float = 0.8,
    output_dir: str = "output/05_concordance",
) -> dict[str, pd.DataFrame]:
    """
    Filter bins, impute NAs, z-score.

    Returns dict with 'retained_bins', 'imputed', 'zscore' DataFrames.
    """
    out_path = Path(output_dir)
    n_cells, n_bins_orig = concordance_shrunk.shape

    # Step 1: Filter bins by cell coverage
    coverage_frac = concordance_shrunk.notna().mean(axis=0)
    retained_mask = coverage_frac >= min_cell_coverage
    retained_bins = concordance_shrunk.columns[retained_mask].tolist()

    filtered = concordance_shrunk[retained_bins].copy()
    logger.info(f"Bin filter: {n_bins_orig} → {len(retained_bins)} bins "
                f"(coverage ≥ {min_cell_coverage:.0%} of cells)")

    # Save retained bins
    with open(out_path / "retained_bins.txt", 'w') as f:
        for b in retained_bins:
            f.write(b + '\n')

    # Step 2: Impute remaining NAs with 0
    na_count = filtered.isna().sum().sum()
    total_entries = filtered.shape[0] * filtered.shape[1]
    logger.info(f"Imputing {na_count} NAs ({na_count / total_entries:.1%}) with 0")

    imputed = filtered.fillna(0.0)

    # Step 3: Z-score per bin across cells
    zscore = imputed.apply(lambda col: (col - col.mean()) / col.std()
                           if col.std() > 1e-10 else col * 0, axis=0)

    # Save all versions
    imputed.to_parquet(out_path / "concordance_imputed.parquet")
    imputed.to_csv(out_path / "concordance_imputed.tsv", sep='\t')
    zscore.to_parquet(out_path / "concordance_zscore.parquet")
    zscore.to_csv(out_path / "concordance_zscore.tsv", sep='\t')

    return {
        'retained_bins': retained_bins,
        'imputed': imputed,
        'zscore': zscore,
    }
```

### Phase 4: Visualization (visualization.py)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def run_umap_and_plot(
    matrix: pd.DataFrame,
    group_labels: pd.Series = None,
    title: str = "UMAP",
    output_prefix: str = "umap",
    output_dir: str = "output/06_umap",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    n_threads: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run UMAP and generate scatter plot.

    Returns DataFrame with (cell, UMAP1, UMAP2, group).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running UMAP on {matrix.shape[0]} cells × {matrix.shape[1]} bins...")

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_jobs=n_threads,
        random_state=random_state,
    )

    embedding = reducer.fit_transform(matrix.values)

    result = pd.DataFrame({
        'cell': matrix.index,
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
    })

    if group_labels is not None:
        result['group'] = group_labels.reindex(matrix.index).values
    else:
        result['group'] = 'unknown'

    # Save coordinates
    result.to_csv(out_path / f"{output_prefix}_coordinates.tsv", sep='\t', index=False)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    if group_labels is not None and result['group'].nunique() <= 20:
        groups = result['group'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(groups)))
        for g, c in zip(groups, colors):
            mask = result['group'] == g
            ax.scatter(result.loc[mask, 'UMAP1'], result.loc[mask, 'UMAP2'],
                       c=[c], label=g, s=15, alpha=0.7)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    else:
        ax.scatter(result['UMAP1'], result['UMAP2'], s=15, alpha=0.7, c='steelblue')

    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path / f"{output_prefix}.png", dpi=150, bbox_inches='tight')
    plt.close()

    return result
```

### Phase 5: CLI Entry Point (cli.py)

```python
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

from . import reference, bam_parser, mixture_model, posterior
from . import concordance, shrinkage, matrix_ops, visualization

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('nome_hic_concordance')


def parse_args():
    parser = argparse.ArgumentParser(
        description='NOMe-HiC Long-Range Chromatin Accessibility Concordance Pipeline'
    )

    # Required
    parser.add_argument('--samples', required=True,
                        help='Path to sample manifest (TSV: cell_name, bam_path, group)')
    parser.add_argument('--reference', required=True,
                        help='Path to .2bit reference genome')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory')

    # Filtering
    parser.add_argument('--min-distance', type=int, default=20000,
                        help='Min genomic distance for qualifying pairs (default: 20000)')
    parser.add_argument('--min-gch-sites', type=int, default=3,
                        help='Min GCH sites per read end (default: 3)')
    parser.add_argument('--bin-size', type=int, default=5_000_000,
                        help='Genomic bin size in bp (default: 5000000)')
    parser.add_argument('--min-pairs-per-bin', type=int, default=5,
                        help='Min pairs per bin to compute r (default: 5)')
    parser.add_argument('--min-cell-coverage', type=float, default=0.8,
                        help='Fraction of cells required non-NA to retain bin (default: 0.8)')

    # EM parameters
    parser.add_argument('--em-max-iter', type=int, default=200)
    parser.add_argument('--em-tol', type=float, default=1e-6)
    parser.add_argument('--short-range-max', type=int, default=1000,
                        help='Max distance for short-range class (default: 1000)')

    # Shrinkage
    parser.add_argument('--shrinkage-n0', type=str, default='auto',
                        help='Shrinkage pseudocount (float or "auto")')

    # SNP filtering
    parser.add_argument('--dbsnp-vcf', type=str, default=None,
                        help='Path to tabix-indexed dbSNP VCF for SNP filtering')

    # Context selection
    parser.add_argument('--contexts', type=str, default='GCH',
                        help='Comma-separated cytosine contexts to analyze. '
                             'Uses IUPAC codes: H={A,C,T}, W={A,T}. '
                             'Default: GCH. Examples: GCH,HCG or WCG')

    # Read processing
    parser.add_argument('--trim-ends', type=int, default=5,
                        help='Bases to trim from read ends (default: 5)')
    parser.add_argument('--mapq-min', type=int, default=20,
                        help='Minimum mapping quality (default: 20)')

    # EM fitting options
    parser.add_argument('--em-subsample', type=int, default=10_000_000,
                        help='Max read ends for EM fitting; subsample if exceeded (default: 10M)')
    parser.add_argument('--em-cell-specific', action='store_true', default=False,
                        help='Also fit EM per-cell for diagnostic comparison')

    # Execution
    parser.add_argument('--n-threads', type=int, default=4,
                        help='Number of parallel threads (default: 4)')
    parser.add_argument('--steps', type=str, default='all',
                        help='Steps to run: comma-separated (1-8) or "all"')

    return parser.parse_args()


def load_samples(path: str) -> pd.DataFrame:
    """Load sample manifest."""
    df = pd.read_csv(path, sep='\t', header=None,
                     names=['cell_name', 'bam_path', 'group'])
    if 'group' not in df.columns or df['group'].isna().all():
        df['group'] = 'unknown'
    df['group'] = df['group'].fillna('unknown')
    logger.info(f"Loaded {len(df)} samples, {df['group'].nunique()} groups")
    return df


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(out / "pipeline_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    steps = set(range(1, 9)) if args.steps == 'all' else set(map(int, args.steps.split(',')))
    samples = load_samples(args.samples)

    # ── Step 1: Context extraction ──
    if 1 in steps:
        logger.info("="*60 + "\nStep 1: Extracting context positions from reference\n" + "="*60)
        contexts = [c.strip() for c in args.contexts.split(',')]
        all_context_positions = reference.extract_context_positions(
            args.reference, str(out / "01_context_extraction"),
            contexts=contexts, dbsnp_vcf=args.dbsnp_vcf,
            n_threads=args.n_threads,
        )
    else:
        logger.info("Step 1: Loading precomputed context positions...")
        contexts = [c.strip() for c in args.contexts.split(',')]
        all_context_positions = reference.load_context_positions(
            str(out / "01_context_extraction"), contexts
        )

    # ── Run pipeline per context ──
    for context in contexts:
        logger.info(f"\n{'#'*60}\nProcessing context: {context}\n{'#'*60}")
        context_positions = all_context_positions[context]
        ctx_out = Path(str(out)) / f"context_{context}"
        ctx_out.mkdir(parents=True, exist_ok=True)

    # ── Step 2: BAM parsing ──
    if 2 in steps:
        logger.info("="*60 + f"\nStep 2: Parsing BAM files for {context}\n" + "="*60)
        cell_parquets = bam_parser.parse_all_cells(
            samples, context_positions,
            str(ctx_out / "02_read_pair_counts"),
            n_threads=args.n_threads, trim_ends=args.trim_ends,
            mapq_min=args.mapq_min, short_range_max=args.short_range_max,
            min_distance=args.min_distance,
        )
    else:
        cell_parquets = {
            row['cell_name']: str(ctx_out / "02_read_pair_counts" / f"{row['cell_name']}.read_pairs.parquet")
            for _, row in samples.iterrows()
        }

    # ── Step 3: Mixture model fitting ──
    if 3 in steps:
        logger.info("="*60 + f"\nStep 3: Fitting beta-binomial mixture for {context}\n" + "="*60)
        # Pool all read pairs across cells
        all_pairs = pd.concat(
            [pd.read_parquet(p) for p in cell_parquets.values()],
            ignore_index=True,
        )
        fit_results = mixture_model.fit_and_compare(
            all_pairs, short_range_max=args.short_range_max,
            min_distance=args.min_distance,
            output_dir=str(ctx_out / "03_mixture_model"),
            max_iter=args.em_max_iter, tol=args.em_tol,
            max_subsample=args.em_subsample,
        )
        fitted_model = fit_results['models']['all']

        # Optional: cell-specific EM fitting for diagnostic
        if args.em_cell_specific:
            logger.info("Fitting cell-specific mixture models for diagnostic...")
            cell_params = mixture_model.fit_cell_specific_models(
                cell_parquets, fitted_model,
                n_threads=args.n_threads,
                output_dir=str(ctx_out / "03_mixture_model"),
            )
    else:
        fitted_model = mixture_model.BetaBinomialMixture.from_dict(
            json.load(open(ctx_out / "03_mixture_model" / "em_fit_all.json"))
        )

    # ── Step 4: Posterior computation ──
    if 4 in steps:
        logger.info("="*60 + "\nStep 4: Computing posteriors\n" + "="*60)
        posterior_paths = posterior.compute_all_posteriors(
            cell_parquets, fitted_model,
            str(ctx_out / "04_posteriors"), args.n_threads,
        )
    else:
        posterior_paths = {
            row['cell_name']: str(ctx_out / "04_posteriors" / f"{row['cell_name']}.posteriors.parquet")
            for _, row in samples.iterrows()
        }

    # ── Step 5: Concordance computation ──
    if 5 in steps:
        logger.info("="*60 + "\nStep 5: Computing per-bin concordance\n" + "="*60)
        # Get chromosome sizes from reference
        import py2bit
        tb = py2bit.open(args.reference)
        chrom_sizes = {c: tb.chroms()[c] for c in tb.chroms()}
        tb.close()

        concordance_raw, pair_counts = concordance.compute_concordance_matrix(
            posterior_paths, chrom_sizes,
            bin_size=args.bin_size, min_distance=args.min_distance,
            min_gch_sites=args.min_gch_sites,
            min_pairs_per_bin=args.min_pairs_per_bin,
            n_threads=args.n_threads,
            output_dir=str(ctx_out / "05_concordance"),
        )
    else:
        concordance_raw = pd.read_parquet(ctx_out / "05_concordance" / "concordance_raw.parquet")
        pair_counts = pd.read_parquet(ctx_out / "05_concordance" / "pair_counts_per_bin.parquet")

    # ── Step 5b: Concordance vs distance analysis ──
    if 5 in steps:
        logger.info("Computing concordance vs. distance decay...")
        distance_analysis.concordance_vs_distance(
            posterior_paths,
            min_gch_sites=args.min_gch_sites,
            n_threads=args.n_threads,
            output_dir=str(ctx_out / "05_concordance"),
        )

    # ── Step 6: Shrinkage ──
    if 6 in steps:
        logger.info("="*60 + "\nStep 6: Shrinkage analysis\n" + "="*60)
        if args.shrinkage_n0 == 'auto':
            shrink_result = shrinkage.estimate_shrinkage_n0(
                concordance_raw, pair_counts,
                output_dir=str(ctx_out / "05_concordance"),
            )
            n0 = shrink_result['n0']
        else:
            n0 = float(args.shrinkage_n0)

        concordance_shrunk = shrinkage.apply_shrinkage(concordance_raw, pair_counts, n0)
        concordance_shrunk.to_parquet(ctx_out / "05_concordance" / "concordance_shrunk.parquet")
        concordance_shrunk.to_csv(ctx_out / "05_concordance" / "concordance_shrunk.tsv", sep='\t')
    else:
        concordance_shrunk = pd.read_parquet(ctx_out / "05_concordance" / "concordance_shrunk.parquet")

    # ── Step 7: Matrix processing + UMAP ──
    if 7 in steps:
        logger.info("="*60 + "\nStep 7: Matrix processing and UMAP\n" + "="*60)
        matrices = matrix_ops.process_concordance_matrix(
            concordance_shrunk, args.min_cell_coverage,
            output_dir=str(ctx_out / "05_concordance"),
        )

        group_labels = samples.set_index('cell_name')['group']

        visualization.run_umap_and_plot(
            matrices['imputed'], group_labels,
            title="UMAP (no z-score)", output_prefix="umap_no_zscore",
            output_dir=str(ctx_out / "06_umap"), n_threads=args.n_threads,
        )

        visualization.run_umap_and_plot(
            matrices['zscore'], group_labels,
            title="UMAP (z-scored)", output_prefix="umap_zscore",
            output_dir=str(ctx_out / "06_umap"), n_threads=args.n_threads,
        )

    logger.info("Pipeline complete.")


if __name__ == '__main__':
    main()
```

---

## 8. Dependencies

```
# requirements.txt
numpy>=1.24
scipy>=1.10
pandas>=2.0
pyarrow>=12.0       # parquet I/O
pysam>=0.21         # BAM reading + tabix-indexed VCF for dbSNP filtering
py2bit>=0.3         # .2bit reference genome reading
scikit-learn>=1.3
umap-learn>=0.5
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

| Module | Key tests |
|--------|-----------|
| `reference.py` | Known context positions in a test sequence; GCG exclusion; both-strand extraction; SNP filtering with mock VCF; IUPAC matching for GCH/HCG/WCG |
| `bam_parser.py` | Reference-comparison methylation calling: C/T at ref=C sites, G/A at ref=G sites; uninformative bases correctly skipped; trim-ends exclusion |
| `mixture_model.py` | Recovery of known parameters from simulated data; convergence; component swap logic; subsampling stability |
| `distance_analysis.py` | Monotonic decay in simulated data; correct distance binning; NaN for sparse bins |
| `concordance.py` | Perfect concordance (r=1) when all pairs have identical posteriors; NaN for < min_pairs |
| `shrinkage.py` | Shrinkage to zero when n₀ >> n; identity when n >> n₀ |
| `matrix_ops.py` | Bin filtering at exact threshold; z-score mean=0, std=1 |

### 9.2 Integration Test

Simulate a dataset with:
- 2 cell types (100 cells each)
- Type A: high concordance in bins 1–50, low in 51–100
- Type B: opposite pattern
- Verify UMAP separates the two types

### 9.3 Validation Checks

- Compare mixture model parameters across short/long subsets (Step 3 diagnostic)
- Verify concordance values are bounded [-1, 1]
- Confirm shrinkage only reduces magnitude, never increases
- Check that z-scored matrix has per-column mean ≈ 0, std ≈ 1

---

## 10. Known Limitations and Future Extensions

1. **Inter-chromosomal pairs excluded.** Trans contacts could provide concordance signal across chromosomes but are much sparser and harder to bin.

2. **Bin assignment by midpoint.** For pairs spanning multiple bins, the midpoint may not represent either end's local environment. An alternative is to assign to the bin of the shorter-distance end, but this adds complexity for marginal gain at 5Mb resolution.

3. **Single distance threshold per run.** The concordance vs. distance analysis (Section 4.6) characterizes decay, but the final matrix uses a single `--min-distance` cutoff. Future: compute concordance at multiple distance scales and concatenate features before UMAP.

4. **~50% of reads uninformative per site.** The reference-comparison approach (no strand tag) means that reads from the non-matching conversion strand are silently skipped at each context site. This is inherent to bhmem's lack of a strand tag and reduces effective coverage by roughly half. If strand information were available, all reads would be usable.

5. **Multiple contexts produce separate matrices.** When running with `--contexts GCH,HCG`, each context produces an independent concordance matrix. Joint analysis (e.g., concatenating GCH + HCG features before UMAP) is left to the user as a downstream step.

6. **GCG exclusion is absolute.** All context definitions exclude GCG as ambiguous between GCH and HCG. This is correct for NOMe-HiC but means ~8% of CpG sites (those preceded by G) are never assessed for either context.

7. **Cell-specific EM is diagnostic only.** The pipeline always uses the pooled model for posterior computation. If the cell-specific diagnostic reveals genuinely divergent component shapes (not just shifted π), the user should investigate manually.

8. **dbSNP filtering is conservative.** Excluding all context sites where any trinucleotide position overlaps a SNP removes some valid sites. The number of context sites genome-wide is large enough (~50–80M on both strands for GCH) that this loss is negligible.
