# Pair-Concordance Extension to meth_concord — Design v2.2

**Status:** Draft v2.2 (supersedes v2.1 — adds two-tier panel design combining a hypothesis-driven TF-anchored panel with an unbiased multi-scale genomic-bin panel, with a Tier 1 / Tier 2 consistency check as a new acceptance criterion and explicit mapping to Fig 4 panels A and C)
**Scope:** Add a pair-of-regions, contact-aware concordance layer on top of the existing `meth_concord` pipeline. Implements the hierarchical beta-binomial model with linked-vs-unlinked decomposition, three parallel per-cell embeddings (E_contact, E_coord, E_full), component-contribution analysis, and matched-contact-frequency comparison.
**Relationship to existing code:** New subpackage `nome_hic_concordance/pair_concordance/`. Reuses Steps 1–4 of the existing pipeline (context extraction, BAM parsing, mixture model, per-read-end posteriors) **unmodified**. Replaces Step 5 onward with a pair-of-regions framework.
**Validation strategy:** GM12878+IMR90 mixed cells first (ground-truth cell-type labels supplied in the sample metadata TSV, previously determined by HCG-methylation clustering against bulk WGBS references), then mouse cortex tissue.

---

## 0. Biological motivation: three-step TF-mediated regulation

The framework tests a specific mechanistic hypothesis about how transcription factor complexes implement long-range gene regulation. TF-mediated regulation proceeds through three sequential steps:

- **Step 1 — Build contacts.** TF complexes (lineage TFs, cohesin/CTCF, mediator) bring two distant regulatory regions into 3D physical proximity.
- **Step 2 — Exclude nucleosomes / open chromatin.** Once in proximity, the TF complex displaces nucleosomes at both contacting regions, producing local chromatin accessibility (captured by GpC methylation in NOMe-HiC).
- **Step 3 — Local DNA methylation reprogramming.** The accessible, TF-bound regions undergo HCG methylation changes (typically hypomethylation at active regulatory elements) that lock in the functional state.

Each step is mechanistically distinct, and the modalities of NOMe-HiC are aligned with the steps: 3D contacts measure Step 1, GCH methylation measures Step 2, HCG methylation measures Step 3. This design (v2.1) operationalizes Step 1 vs Step 2 separation using GCH; an HCG extension capturing Step 3 conditional on Steps 1+2 is deferred to v3 (§12). Bulk methods and contact-only methods conflate these steps; pseudobulk integration of separate single-cell methods cannot link them at the single-molecule level. The pair-concordance framework below explicitly separates Step 1 (contact frequency) from Step 2 (functional coordination given contact) through the linked / unlinked read decomposition and the three parallel embeddings (E_contact, E_coord, E_full).

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

### 3.0 Two-tier panel design

The pair panel is split into two tiers that serve complementary purposes and use the same downstream pipeline (classification, EB priors, Δ, embeddings, AUC) — they differ only in how their pairs are constructed.

**Tier 1 — Hypothesis-driven TF-anchored panel** (~150k pairs). Anchors centered on TF motif occurrences, with categories defined by the TF identity. Every category makes a *directional prediction* (e.g., NEUROD2 pairs should coordinate in Exc; NRF1 pairs should not coordinate cell-type-specifically anywhere). This is the panel that carries the **mechanistic falsification structure** — without it, the framework reduces to "cell-type-specific coordination exists" rather than the specific claim that TF-mediated 3D folding drives it.

**Tier 2 — Unbiased multi-scale genomic-bin panel** (~130k pairs). Non-overlapping genomic bins at three scales — compartment (1 Mb), domain (250 kb), loop (25 kb) — filtered to top-decile pseudobulk-contact pairs at each scale. No TF labels; categories defined by scale only. This panel serves two distinct purposes:

1. **Generalization check.** Verifies that the cell-type-specific coordination signal is not an artifact of TF anchor selection. If Tier 1 (TF-anchored) shows strong E_coord cell-type separation but Tier 2 (unbiased) shows none, that's a problem.
2. **Multi-scale framing for Fig 4 Panel A.** Each scale corresponds to a different known 3D organization mechanism (A/B compartments at 1 Mb, TADs at 250 kb, loops at 25 kb), allowing the framework to demonstrate coordination at all three scales simultaneously.

**Mapping to Fig 4:**

| Figure 4 panel | Panel content | Tier used |
|---|---|---|
| A | Multi-scale tri-modal concordance (compartment / domain / loop) | Tier 2 |
| C | 3D-conditioned tri-modal coupling at lineage E-P pairs (headline) | Tier 1 |
| D | Per-cell heatmaps at marker loci | Tier 1 (subset) |
| E | CTCF footprint × loop occupancy | Tier 1 (CTCF subset) + separate footprint analysis |

The two tiers are produced by parallel builder functions and concatenated into a single panel DataFrame; the downstream pipeline processes them uniformly. A `tier` column in the panel TSV identifies which tier each pair belongs to.

**Recommended phasing:** if implementing under time pressure, start with **Tier 1** for v1 — it carries the GM12878+IMR90 validation and the headline mechanistic claim. Add Tier 2 in a second sprint once Tier 1 has passed the validation acceptance criteria. This matches the staged timeline in §12.

### 3.1 Anchor definitions

Tier 1 anchors are **fixed-size 5 kb windows centered on features**:

- TF-anchored pairs: centered on TF motif occurrence (FIMO p < 1e-4), optionally intersected with ENCODE ChIP-seq peaks for the relevant cell line/tissue (recommended for GM12878+IMR90 validation since ChIP-seq is available for both).
- co_folded / non_folded pairs: centered on top-/bottom-decile pseudobulk contact frequency bins at the appropriate scale.
- random_distance_matched: random 5 kb intervals at matched distance.

Tier 2 anchors are **genomic bins of the panel's nominal size** (1 Mb / 250 kb / 25 kb), non-overlapping, tiled across the genome (excluding centromeres, telomeres, ENCODE blacklist regions). For the loop scale (25 kb), an optional refinement re-centers each anchor on the pseudobulk-contact center within the bin to give a ±2.5 kb sub-anchor at the actual contact peak — this is helpful at the loop scale where 25 kb is large relative to typical loop precision; not worth the complexity at coarser scales.

The 5 kb size for Tier 1 captures the local TF-bound regulatory neighborhood (footprint + flanking nucleosomes) and contains ~10–30 GCH sites for binarization confidence. Tier 2 deliberately uses larger bins at coarser scales — these are not trying to resolve the precise mechanism, only to verify that coordination is detectable at known scales of 3D organization.

### 3.2 Pair panel file format

A bedpe-style TSV consumed by `panel.py`:

```
pair_id    chrom_i  start_i  end_i  chrom_j  start_j  end_j  distance  tier   category                       distance_band
T1_001     chr1     1000000  1005000  chr1     1500000  1505000  500000   tier1  tf_anchored_NEUROD2            mid
T1_002     chr2     2300000  2305000  chr2     5300000  5305000  3000000  tier1  random_distance_matched        long
T2_001     chr1     8000000  9000000  chr1     12000000 13000000 4000000  tier2  unbiased_compartment           long
T2_002     chr1     8000000  8250000  chr1     9000000  9250000  1000000  tier2  unbiased_domain                mid
T2_003     chr1     8000000  8025000  chr1     8500000  8525000  500000   tier2  unbiased_loop                  short
...
```

Distance bands (independent of tier): `short` (100–500 kb), `mid` (500 kb – 2 Mb), `long` (2 Mb +). The `tier` column lets downstream analyses stratify cleanly.

### 3.3 Tier 1 categories — TF-anchored

The Tier 1 category panel is designed so each row makes a *directional* prediction about which cell type should show coordination, enabling bidirectional falsification tests (not just "the two cell types differ" but "B-lymphoid TFs separate GM12878-biased coordination, fibroblast TFs separate IMR90-biased coordination").

**Categories shared between validation and cortex runs:**

| Category | Definition | Expected behavior |
|----------|------------|-------------------|
| `co_folded` | Pseudobulk contact frequency in top decile at matched distance | Strong cell-type-specific Δ |
| `non_folded` | Pseudobulk contact frequency in bottom decile, distance-matched | Weak Δ (Step 1 negative control) |
| `tf_anchored_CTCF` | Both anchors contain CTCF motif (convergent orientation) | Strong, broadly across cell types (architectural) |
| `tf_anchored_NRF1` | Both anchors NRF1 motif (housekeeping promoter TF) | NOT cell-type-specific — **falsification control** |
| `tf_anchored_SP1` | Both anchors SP1 motif (housekeeping) | NOT cell-type-specific — **falsification control** |
| `random_distance_matched` | Random pairs matched on distance distribution | Near-chance signal |

**Validation-specific (GM12878+IMR90) lineage TF categories:**

| Category | Lineage | Expected behavior |
|----------|---------|-------------------|
| `tf_anchored_SPI1` | B-lymphoid (PU.1) | Strong Δ in GM12878; null in IMR90 |
| `tf_anchored_EBF1` | B-lymphoid | Strong Δ in GM12878; null in IMR90 |
| `tf_anchored_IRF4` | B-lymphoid | Strong Δ in GM12878; null in IMR90 |
| `tf_anchored_TEAD1` | Fibroblast (YAP/TAZ pathway) | Strong Δ in IMR90; null in GM12878 |
| `tf_anchored_SRF` | Fibroblast (serum response) | Strong Δ in IMR90; null in GM12878 |
| `tf_anchored_AP1` (JUN/FOS) | Fibroblast (stress/growth) | Strong Δ in IMR90; null in GM12878 |

This bidirectional panel is critical for validation: a unidirectional panel (only B-lymphoid TFs) would only confirm that GM12878 and IMR90 differ in some way, which is trivially true. The bidirectional panel confirms that *the direction* of cell-type-specific coordination matches *the direction* predicted by TF lineage assignment.

**Cortex-specific lineage TF categories:**

| Category | Lineage | Expected behavior |
|----------|---------|-------------------|
| `tf_anchored_NEUROD2` | Excitatory neuron | Strong Δ in Exc; null elsewhere |
| `tf_anchored_OLIG2` | Oligodendrocyte | Strong Δ in Oli |
| `tf_anchored_SPI1` (PU.1) | Microglia | Strong Δ in Mic |
| `tf_anchored_MEF2C` | Active neurons (broad) | Strong Δ in active neuronal clusters |

**Tier 1 target panel size: ~150,000 pairs total** across categories and distance bands.

### 3.4 Tier 2 categories — unbiased multi-scale

Three scales, each defined by non-overlapping genomic bins at the indicated size. Pair enumeration restricted to **top-decile pseudobulk-contact pairs** at each scale (full bin × bin enumeration would be combinatorially infeasible at the loop scale; top-decile filtering keeps pairs where the population-average contact is real enough to give individual cells any chance of supporting the contact).

| Category | Bin size | Distance range | Approx pairs | Biological interpretation |
|----------|----------|----------------|--------------|---------------------------|
| `unbiased_compartment` | 1 Mb | 5 Mb – 50 Mb (or chromosome length) | ~20k | A/B compartment co-localization |
| `unbiased_domain` | 250 kb | 500 kb – 5 Mb | ~30k | TAD-level coordination |
| `unbiased_loop` | 25 kb | 100 kb – 2 Mb | ~80k | Loop-level / E-P-like scale |

**Why pseudobulk-top-decile filter:** at 25 kb resolution there are ~120k bins genome-wide (human) → ~7×10^9 pairs, computationally and statistically infeasible. The top-decile filter (per distance band, to avoid all top-decile pairs sitting at very short range) keeps the panel tractable while concentrating on pairs where any cell could plausibly support the contact.

**Tier 2 target panel size: ~130,000 pairs total** across scales.

**Note on Tier 2 EB priors:** the empirical-Bayes prior pooling (§5 in math doc) should be by category — so `unbiased_compartment` has one prior shared across all 1 Mb bin pairs, `unbiased_domain` has another, `unbiased_loop` a third. Coordination magnitudes likely differ across scales (compartment-scale coordination is high but driven by A/B structure; loop-scale coordination is sparser but more mechanism-specific), so pooling across scales would inappropriately shrink scale-specific structure.

**Note on existing meth_concord bin-based concordance pipeline:** the existing pipeline (Steps 5–7 in `concordance.py`, `shrinkage.py`, `matrix_ops.py`) computes Pearson r of read-end posteriors per 5 Mb bin per cell. This is *different* from the Tier 2 panel: the existing pipeline aggregates many pair midpoints per bin into a single per-cell concordance value; the Tier 2 panel treats each bin × bin pair as its own observation. They are not interchangeable. The Tier 2 1 Mb compartment scale partially overlaps in spirit with the existing 5 Mb bins but operates at a finer resolution and uses the pair-of-regions framework. Do not replace either with the other; both have a place.

### 3.5 Tier 1 / Tier 2 consistency check

A new acceptance criterion for the GM12878+IMR90 validation (added to §8.3):

> **Tier 2 unbiased coordination must show cell-type structure consistent with Tier 1:** AUC at Tier 2 `unbiased_loop` pairs must be > 0.7 (showing coordination at unbiased loci does separate cell types), AND AUC at Tier 2 `unbiased_loop` pairs must be ≤ AUC at Tier 1 lineage TF pairs (confirming that the targeted panel captures stronger or comparable signal — Tier 1 mechanistic specificity should yield equal or stronger separation than Tier 2 generic coordination).

This protects against two failure modes:
- **Tier 1 strong but Tier 2 null:** signal is specific to selected anchors in a way that doesn't generalize — possibly a motif-scan artifact or panel-selection bias.
- **Tier 2 stronger than Tier 1:** TF-anchored selection is *worse* than random bin selection — suggests motifs are missing actual binding (use ChIP-seq peaks instead) or the TF-mediated mechanism is not what's driving cell-type-specific coordination.

Both failure modes are diagnostic and inform what to fix; neither is necessarily a project-killer, but both must be resolved before proceeding to cortex.

### 3.6 Module sketches (panel.py)

Two builder functions, both producing the same panel DataFrame schema (with `tier` column distinguishing source):

```python
def build_tier1_tf_anchored_panel(
    tf_motif_beds: dict[str, str],     # motif_name -> BED path of motif occurrences
    tf_chip_beds: dict[str, str] | None,  # optional motif × ChIP intersection
    pseudobulk_cooler: str,             # for co_folded / non_folded selection
    chrom_sizes: dict[str, int],
    anchor_size: int = 5_000,
    distance_bands: dict[str, tuple[int, int]] = None,
    n_per_category_per_band: int = 5_000,
    cell_type_context: str = "validation_gm12878_imr90",  # or "cortex"
    output_path: str = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build the Tier 1 TF-anchored panel."""


def build_tier2_multiscale_panel(
    pseudobulk_cooler: str,
    chrom_sizes: dict[str, int],
    blacklist_bed: str | None = None,
    scales: dict[str, dict] = {
        "compartment": {"bin_size": 1_000_000, "distance_range": (5_000_000, 50_000_000)},
        "domain":      {"bin_size": 250_000,   "distance_range": (500_000, 5_000_000)},
        "loop":        {"bin_size": 25_000,    "distance_range": (100_000, 2_000_000)},
    },
    top_decile_filter: bool = True,
    loop_recenter: bool = True,           # re-anchor 25 kb bins on contact peak
    output_path: str = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build the Tier 2 unbiased multi-scale panel."""


def build_panel(
    tier1_config: dict | None,
    tier2_config: dict | None,
    output_path: str,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build the combined Tier 1 + Tier 2 panel.

    If both configs supplied, concatenates with disjoint pair_id namespaces
    (T1_* for tier 1, T2_* for tier 2). Either may be None to run a single
    tier (e.g., Tier 1 only for the initial GM12878+IMR90 validation).
    """
```

All deterministic given `random_state`. Panel cached as parquet keyed on a config hash. The combined panel TSV can be ~280k pairs (Tier 1 + Tier 2); memory and compute remain tractable.

---

## 4. Per-pair-per-cell linked/unlinked aggregation (Step 6p)

### 4.1 Definition (reframed for actual meth_concord data)

For a given pair of regions (i, j) and cell c, every row in `posteriors.parquet` for cell c is one of:

| Class | Condition | Use |
|-------|-----------|-----|
| **Linked (L)** | (pos1 ∈ i AND pos2 ∈ j) OR (pos1 ∈ j AND pos2 ∈ i); distance ≥ `min_distance` | Joint observation of state at i AND j given a long-range contact |
| **Unlinked-i (U_i)** | pos1 ∈ i AND \|pos2 − center(i)\| ≤ `unlinked_max_distance` AND pos2 ∉ j (or symmetric on pos2) | Local marginal state at i |
| **Unlinked-j (U_j)** | pos1 ∈ j AND \|pos2 − center(j)\| ≤ `unlinked_max_distance` AND pos2 ∉ i (or symmetric on pos2) | Local marginal state at j |
| **Other** | Anything else (e.g. one end in i and mate at a long-range site that is not j, or neither end in i ∪ j) | Discarded — not informative for this pair |

**Critical: the short-range constraint on unlinked reads.** The unlinked-i reads must have their mate at *short range* from anchor i (default `unlinked_max_distance = 20 kb`, matching the existing pipeline's `min_distance` floor). This ensures:

1. The unlinked-i reads measure **local** marginal accessibility at i — the same quantity bulk NOMe-seq at i would measure. Without the short-range constraint, "unlinked-i" reads would include reads whose mate is at some random other long-range site, which contaminates the marginal estimate with other long-range contact contexts.
2. The contrast `linked - unlinked` cleanly becomes "long-range contact-conditional concordance minus local-baseline-predicted concordance" — the correct framing for testing Step 2 of the three-step TF model.
3. The unlinked-i reads are exactly the short-range partner of the existing meth_concord pipeline's `--short-range-max` parameter, providing a natural symmetry: long-range reads (distance ≥ 20 kb) become linked observations at panel pairs; short-range reads (distance < 20 kb) become unlinked observations at panel pairs where one end overlaps an anchor.

The distance filter (`min_distance`, default 20 kb) is already enforced on linked reads as in the existing pipeline; further restrict by `n_min` per end (existing `min_gch_sites`, default 3).

### 4.2 Binarization of per-read-end state

The existing pipeline produces continuous `posterior1`, `posterior2` ∈ [0, 1]. For the hierarchical model, binarize:

- **state = "open"** if posterior > τ (default 0.5; the existing pipeline's `--posterior-threshold`).
- **state = "closed"** if posterior ≤ τ.
- Reads with NaN posterior (n=0 ends) are discarded.

For each linked pair:
- **active-concordant** = both ends open
- **symmetric-concordant** = both ends same state (both open OR both closed)

For each unlinked-i read: state at end-in-i is "open" or "closed". Similarly for unlinked-j.

Active concordance is the primary outcome (reflects coordinated regulatory activation, the signature of TF-mediated nucleosome eviction at both anchors); symmetric is supplementary.

### 4.3 Per-pair, per-cell counts

For each (pair_id, cell_id), classify and tally:

| Quantity | Symbol | Definition |
|----------|--------|------------|
| Linked reads, total | $n_{ij}^{c,L}$ | Long-range read pairs with end1∈i, end2∈j (or symmetric) and both ends informative |
| Linked reads, active-concordant | $k_{ij}^{c,L,A}$ | Of those, both ends open |
| Linked reads, symmetric-concordant | $k_{ij}^{c,L,S}$ | Of those, same state |
| Unlinked-i, total | $n_i^c$ | Short-range reads with one end in i (mate within `unlinked_max_distance` of i, mate ∉ j) |
| Unlinked-i, open | $k_i^{c,\text{open}}$ | Of those, the in-i end open |
| Unlinked-j, total | $n_j^c$ | symmetric, mate within `unlinked_max_distance` of j |
| Unlinked-j, open | $k_j^{c,\text{open}}$ | symmetric |

### 4.4 Implementation

`pair_concordance/classify.py:aggregate_pair_counts()`:

```python
def aggregate_pair_counts(
    posteriors_parquet: str,
    cell_name: str,
    panel: pd.DataFrame,                # output of panel.py
    posterior_threshold: float = 0.5,
    min_gch_sites: int = 3,
    min_distance: int = 20_000,         # min distance for linked reads
    unlinked_max_distance: int = 20_000,  # max mate distance for unlinked reads
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

### 8.1 Ground-truth cell-type labels (supplied via sample metadata)

Ground-truth labels are NOT computed inside this pipeline. They are supplied as a column in the sample metadata TSV (the same manifest already used by the existing `nome_hic_concordance` pipeline) and consumed as-is. This decouples ground-truth determination from the pair-concordance analysis, ensuring labels are based on an orthogonal modality and cannot be tautologically influenced by E_coord (which is built from GCH accessibility + 3D contacts).

**Provenance of the labels (for documentation, not part of this pipeline):**

For the GM12878+IMR90 validation, labels were previously determined by:

1. Per-cell HCG methylation at 250 kb bins.
2. UMAP embedding of the cells × bins HCG matrix.
3. Comparison of the resulting clusters against bulk WGBS profiles from GM12878 and IMR90 reference data.

The two-cluster structure was unambiguous (separated by orders of magnitude in cluster distance), so cluster-to-cell-line assignment is reliable. The resulting per-cell label is stored in the sample metadata as `cell_type` (or similar column name; configurable).

**Why this is a valid validation:**

E_coord is constructed from GCH accessibility states and Hi-C contact patterns. The ground truth is from HCG methylation. These modalities are biologically related but not identical — GCH captures Step 2 of TF regulation (nucleosome positioning), HCG captures Step 3 (DNA methylation reprogramming). Asking "does the GCH-and-contact-based E_coord recover the HCG-based clusters?" is therefore a real test of whether contact-conditional coordination measurements carry cell-type information that is consistent with the cell-type structure observed in an independent modality. It is not a tautology, and it does not require any cell-line-specific or species-specific ground-truth procedure.

**Sample metadata format:**

The existing manifest format is extended with an optional `cell_type` column:

```
cell_name    bam_path                  group        cell_type
cell_001     /data/cell_001.bam        mixed_v1     GM12878
cell_002     /data/cell_002.bam        mixed_v1     IMR90
cell_003     /data/cell_003.bam        mixed_v1     GM12878
cell_004     /data/cell_004.bam        mixed_v1     ambiguous
...
```

Cells with `cell_type = ambiguous` (or NaN) are still processed by the pair-concordance pipeline (their Δ values contribute to EB priors and embeddings), but they are excluded from supervised evaluation in §7.1, §7.2, and §8.3.

### 8.2 Pipeline (`pair_concordance/pipelines/validate_gm12878_imr90.py`)

```bash
nome_hic_pair_concordance \
    --posteriors-dir output/context_GCH/04_posteriors/ \
    --pair-panel panels/gm12878_imr90.bedpe \
    --samples samples.tsv \
    --label-column cell_type \
    --output-dir output/pair_concordance_gm12878_imr90/ \
    --eb-pool-by category \
    --n-threads 8 \
    --mode validation
```

Validation mode:

1. Build/load pair panel (with the bidirectional GM12878/IMR90 TF panel from §3.3).
2. Aggregate linked/unlinked counts per cell.
3. Fit EB priors per category.
4. Compute Δ per pair per cell.
5. Build E_contact, E_coord, E_full.
6. Compute classification accuracy vs `cell_type` column in samples.tsv (excluding `ambiguous`).
7. Run AUC per category, matched contact comparison.
8. Generate validation report (JSON + PDF).

### 8.3 Pre-registered acceptance criteria

Going to mouse cortex is gated on passing all six. **Commit to these thresholds before running.** Criteria 1–5 apply to Tier 1 outputs; criterion 6 is the Tier 1 / Tier 2 consistency check from §3.5.

| # | Criterion | Threshold | Rationale |
|---|---|---|---|
| 1 | E_full classification accuracy vs `cell_type` labels (on Tier 1 panel) | ≥ 0.95 | The mix is well-separated by HCG; E_full integrates all modalities and should easily recover this |
| 2 | E_coord classification accuracy vs `cell_type` labels (on Tier 1 panel) | ≥ 0.85 | Coord alone (GCH + contact) should still discriminate two very different cell lines via Step 2 signal |
| 3 | AUC at `random_distance_matched` pairs (E_coord) | ≤ 0.65 | Confirms coord measures pair-specific signal, not noise |
| 4 | AUC at housekeeping pairs (`tf_anchored_NRF1`, `tf_anchored_SP1`) significantly less than AUC at lineage TF pairs (E_coord) | LR test p < 0.05 | Housekeeping ≠ lineage — confirms TF-folding hypothesis structure |
| 5 | **Directional consistency:** B-lymphoid TF pairs (`SPI1`, `EBF1`, `IRF4`) drive separation in the GM12878-biased direction; fibroblast TF pairs (`TEAD1`, `SRF`, `AP1`) drive separation in the IMR90-biased direction | Both directional AUCs > 0.7; cross-direction AUCs near chance | Bidirectional falsification — confirms separation matches predicted TF lineage |
| 6 | **Tier 1 / Tier 2 consistency:** AUC at Tier 2 `unbiased_loop` pairs > 0.7 AND ≤ AUC at Tier 1 lineage TF pairs | Both inequalities hold | Confirms (a) signal generalizes beyond TF-selected anchors, and (b) targeted TF panel captures stronger or comparable signal than unbiased panel |

Criteria 5 and 6 together form the strongest defense of the framework's biological claim:

- Criterion 5 (directional) says: cell-type separation depends on *which* TFs anchor the pairs in a way that matches TF biology.
- Criterion 6 (Tier consistency) says: the framework is not over-tuned to specific TF anchors, but the mechanistic specificity of TF anchors does produce stronger signal than random bins.

Failing criterion 6 while passing 1–5 is the most likely "interesting failure mode" worth dwelling on:

- **Tier 1 strong, Tier 2 null:** signal is artifactually specific to selected anchors. Most likely cause is motif-scan inaccuracy (motif occurrence ≠ actual TF binding). Fix by intersecting motif anchors with ENCODE ChIP-seq peaks.
- **Tier 2 stronger than Tier 1:** unbiased bins separate cell types better than TF-anchored bins. This indicates either motifs are wrong OR the TF-mediated mechanism is not what's driving cell-type coordination at the loci you're sampling. Diagnostic; informs whether to use ChIP-seq-intersected anchors, switch to a different TF panel, or rethink the mechanistic claim.

If any criterion fails: STOP. Debug. Do not proceed to mouse cortex.

**Run priority if implementing in phases:** If Tier 2 is not yet implemented when Tier 1 reaches the validation stage, run Tier 1 alone first (criteria 1–5 only). Tier 1 passing is necessary but not sufficient — Tier 2 must be implemented and criterion 6 must pass before proceeding to mouse cortex.

### 8.4 Likely failure modes and diagnostics

| Symptom | Likely cause | Check |
|---|---|---|
| E_coord ARI ≪ E_contact ARI | Coordination signal too sparse to embed | `median per-cell informative pair count`; need ≥500 |
| All Δ near zero | Independence baseline computed wrong, or `unlinked_max_distance` too small (no unlinked reads survive) | Inspect a few high-coverage pairs manually; tune `unlinked_max_distance` 10–50 kb |
| NRF1 AUC ≈ lineage TF AUC | TF panel construction wrong, OR coord captures only marginal state, OR mixture model classifying everything as same state | Re-run mixture model diagnostic from existing Step 3; check motif scan output |
| Per-pair Δ has heavy tails | EB priors not regularizing enough at low-count pairs | Inspect `category` priors; consider per-band pooling; floor $\nu \geq 5$ |
| Directional consistency fails | Either the lineage-TF motifs aren't where the actual TF binding is (use ChIP-seq peak intersection instead of motif scan), or the TF complex in question doesn't actually mediate Step 1 contacts in these cells | Cross-reference motif-anchored panel with ENCODE ChIP-seq for GM12878 and IMR90 |

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

- **Existing `design/NOME_HIC_CONCORDANCE_PIPELINE_DESIGN.md`** in this repo: documents Steps 1–7 of the current bin-based concordance pipeline. This v2.2 design supplements that document with the pair-concordance subpackage (Steps 5p–9p) and the two-tier panel design.
- **`SCNOMEHIC_FIG4_PLAN_v1.md`** (sibling `scnomehic_analysis` repo): the contact-aware pipeline here implements the statistical core of Fig 4 Panels A (Tier 2 unbiased multi-scale) and C (Tier 1 TF-anchored). Mouse-cortex outputs from this pipeline feed directly into the figure.
- **Companion math doc**: `PAIR_CONCORDANCE_MATH_v1.1.md` formalizes the likelihood, prior specification, posterior derivations, and identifiability conditions. The math doc is tier-agnostic — same model applies to Tier 1 and Tier 2 pairs.

---

## 14. Version history

- **v1**: Generic design before meth_concord source was available; used a hypothetical adapter layer and assumed external data structures.
- **v2**: Rewritten after reviewing actual `meth_concord` codebase. Builds directly on `04_posteriors/{cell_name}.posteriors.parquet` outputs. Reuses Steps 1–4 unchanged. Reframes "linked vs unlinked" as re-aggregation of existing parquet rows by pair-of-regions definition (not new data extraction). New subpackage under `nome_hic_concordance/pair_concordance/` with a second CLI entry point.
- **v2.1:**
  - **§0 (new):** Three-step TF-mediated regulation biological framing (build contacts → exclude nucleosomes → reprogram methylation). Aligns the analysis decomposition with the mechanistic hypothesis being tested.
  - **§4.1:** Unlinked-i and unlinked-j reads now require their mate to be at *short range* from the in-region anchor (`unlinked_max_distance`, default 20 kb). This ensures they measure local marginal accessibility, the correct baseline for testing Step 2 conditional on Step 1.
  - **§4.4:** Added `unlinked_max_distance` parameter to `aggregate_pair_counts()`.
  - **§3.3:** Replaced generic TF panel with a bidirectional panel for GM12878+IMR90 validation: B-lymphoid TFs (SPI1/PU.1, EBF1, IRF4) expected to drive GM12878-biased coordination, fibroblast TFs (TEAD1, SRF, AP1) expected to drive IMR90-biased coordination. Added SP1 alongside NRF1 as a housekeeping falsification control. Cortex TF panel separated cleanly.
  - **§8.1:** Replaced SNP-genotype-based ground truth with **pre-specified cell-type labels supplied via the sample metadata TSV**. The labels were previously determined (outside this pipeline) by HCG-methylation clustering at 250 kb bins compared against bulk WGBS references. Acknowledges that ground truth comes from an orthogonal modality (HCG) while E_coord is built from GCH + contacts.
  - **§8.3:** Added a fifth acceptance criterion testing *directional consistency*.
- **v2.2 (current):**
  - **§3 (restructured):** Introduced **two-tier panel design**:
    - **Tier 1** — hypothesis-driven TF-anchored panel (~150k pairs, ~5 kb anchors), carries the mechanistic falsification structure.
    - **Tier 2** — unbiased multi-scale genomic-bin panel at compartment (1 Mb), domain (250 kb), and loop (25 kb) scales, top-decile-pseudobulk-contact-filtered (~130k pairs total), serves as generalization check and feeds Fig 4 Panel A.
    - Two parallel builder functions in `panel.py` produce a combined panel with a `tier` column.
    - Explicit mapping of Fig 4 panels to tiers (Panel A → Tier 2, Panel C → Tier 1).
    - Recommended phasing: implement Tier 1 first, add Tier 2 in a second sprint.
  - **§3.4 (new):** Tier 2 categories — `unbiased_compartment`, `unbiased_domain`, `unbiased_loop`. Discussion of bin-pair selection (top-decile filter), EB prior pooling per category, and relationship to existing meth_concord bin-based concordance pipeline (different aggregation; both should be kept).
  - **§3.5 (new):** Tier 1 / Tier 2 consistency check defined as a formal acceptance criterion. Identifies the two interesting failure modes: Tier 1 strong + Tier 2 null (anchor-selection artifact) and Tier 2 stronger than Tier 1 (motif misspecification or wrong mechanistic claim).
  - **§3.6:** Updated `panel.py` module sketches to two builder functions plus a combiner.
  - **§8.3:** Added **sixth acceptance criterion** for Tier 1 / Tier 2 consistency. Now six criteria total.

Future revisions:
- v3 after Week 5 sprint: refine based on actual implementation experience.
- v4 after Week 6 GM12878+IMR90 validation: lock or revise based on validation outcomes.
- v5: add HCG (tri-modal Step 3) extension if v2.2 results justify it.
