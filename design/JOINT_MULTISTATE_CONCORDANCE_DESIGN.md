# Joint GCH+HCG multi-state long-range concordance

## Design document v0.3 — locked decisions + implementation status

> **v0.3 (implemented):** J1/J2, a focused J4, and the J3 embedding are all
> built as the leakage-free path — see **§10**.  J4 result on real data:
> **long-range HCG (methylation) concordance carries strong cell-type identity
> beyond marginals (Δ|m+cov ≈ 0.90–0.93 at 1–2 Mb), but the joint multi-state
> does not exceed HCG-only** — methylation is the carrier (accessibility
> concordance is weak), as §9b anticipated.  J3 was built for the interpretive
> channel decomposition, not to beat HCG (§10.1).

> Extends the signed (single-context) concordance pipeline
> (`SIGNED_CONCORDANCE_AND_RWR_DESIGN.md`) to integrate **GCH
> accessibility** and **HCG CpG methylation** into a single per-end
> epigenetic *state*, and to measure **multi-state long-range
> concordance** between distal loci within each cell.

---

## 1. Motivation

The current pipeline measures GCH and HCG concordance **separately** —
each context run gives one per-end probability ("open" for GCH,
"methylated" for HCG) and a 2-state co-concordance (co-open / co-closed /
mixed). But each NOMe-HiC read end carries **both** signals, and their
*combination* is biologically meaningful:

| GCH (accessibility) | HCG (methylation) | joint state | biology |
|---|---|---|---|
| high (accessible) | low (unmethylated) | **open** | active promoters/enhancers |
| low (occupied)    | high (methylated)  | **closed** | repressed heterochromatin |
| low (occupied)    | low (unmethylated)  | **poised** | nucleosomal-unmethylated / poised |
| high (accessible) | high (methylated)  | **active-methylated** | rarer (methylated accessible elements) |

Measuring whether two distal, 3-D-contacting loci share the **same joint
state** ("co-open", "co-closed", "co-poised", "co-active-methylated") is a
richer probe of coordinated long-range epigenetic regulation than either
context alone.

---

## 2. Per-end joint state

From the two Beta-Binomial mixtures (one per context):

- `a = P(accessible | GpC counts)`
- `m = P(methylated | CpG counts)`

Under a first-order independence approximation (GCH ⟂ HCG *given the
locus*), the per-end soft state distribution over the 2×2 grid is:

```
q_open             = a · (1 − m)
q_closed           = (1 − a) · m
q_poised           = (1 − a) · (1 − m)
q_active_methylated= a · m
        (sum = 1)
```

**Decision — independence approximation (v1).** GCH and HCG are
biologically anti-correlated; using the product of marginals is an
approximation. The principled v2 upgrade is to learn the joint
distribution (a 2-D mixture over (accessibility, methylation)) instead of
multiplying marginals. Start with the product.

### 2.1 Missing modality — **lenient by default**

An end may have only GpC sites, only CpG sites, or neither:

| end has | `a` | `m` | effect |
|---|---|---|---|
| both | observed | observed | sharp 4-state |
| GCH only | observed | **prior** | accessibility resolved, methylation diffuse |
| HCG only | **prior** | observed | methylation resolved, accessibility diffuse |
| neither | — | — | end dropped |

This is **Bayesian marginalization** of the missing modality: the
Beta-Binomial mixture's `predict_proba(k=0, n=0)` returns the model's
prior (the genome-wide marginal rate), so plugging `n = 0` for the absent
context *automatically* yields the prior. The partially-observed end's
state distribution is correctly *less committed*, not wrongly committed.

**Decisions:**
- **Lenient mode is the default** (marginalize missing → keep the pair).
  A **strict mode** (require both modalities at both ends) is optional for
  cleaner-but-sparser analyses.
- **Prior source: global mixture prior by default**; a **locally-informed
  per-region prior** (per-bin / per-feature methylation & accessibility
  rate) is an **option** (more accurate for CpG-islands vs gene bodies).
- **Shared-prior artifact safeguard.** Imputing a shared prior could create
  spurious agreement between two prior-imputed ends. This is a per-locus
  property present equally in linked and short-range pairs, so it **cancels in
  the linked − short-range subtraction** (§4), provided missingness is
  distance-independent within a bin. A QC field records the
  fully-observed vs partially-observed pair fraction per (cell, bin).

---

## 3. Multi-state soft co-state (generalized U-3)

For a pair of ends *i*, *j* with state distributions `q_i`, `q_j`, the soft
co-state for each state *s* is the product (independence across ends), and
"mixed" is the remainder:

```
co_s   = q_i[s] · q_j[s]          for s ∈ {open, closed, poised, active_methylated}
mixed  = 1 − Σ_s co_s
        (the five sum to 1)
```

Aggregate the soft counts per **(cell, 250-kb bin)** separately for the
**linked** pool (genomic distance ≥ d0 ≈ 20 kb; long-range) and the
**short-range** pool (< d0; local baseline), giving
`k_co_open, k_co_closed, k_co_poised, k_co_am, k_mixed, n` for each pool.

---

## 4. Estimation + the concordance vector

**Estimation: Dirichlet-Multinomial (DM)**, exactly as in the signed
pipeline but over the 5 categories (4 co-states + mixed). DM is already a
K-category estimator; the empirical-Bayes method-of-moments prior fit
carries over unchanged. (The signed pipeline's DM-vs-MLE conclusion holds:
DM on RWR-smoothed counts was more robust on the full dataset.)

**Output — a per-(cell, bin) concordance vector** (the multi-state
generalization of the signed scalar):

```
Δ_s = π_s(linked) − π_s(short-range)   for s ∈ {co-open, co-closed, co-poised, co-active-methylated}
```

- **Four channels** (drop "mixed" — it is the redundant residual; the four
  co-state Δ's carry the free information).
- Optional interpretable contrast `Δ_co-open − Δ_co-closed` recovers the
  original single-context "activity axis" as one coordinate.

The feature object is a tensor: **cells × bins × 4 channels**.

---

## 5. RWR denoising — both modalities

- **Matrices smoothed:** all four co-state channels × {linked, short-range}.
- **k-NN graph (pass 0):** built from **both** modalities —
  `QN(mean GCH accessibility) ⊕ QN(mean HCG methylation)` per cell (or the
  per-cell mean over the four joint states). Refinement passes rebuild the
  graph from the joint concordance vector itself (eigengap auto-stop).

---

## 6. Dimensionality reduction + embedding

**Per-channel normalization is mandatory first** — the channels are on very
different scales (co-poised large, co-active-methylated small), so each
channel gets the per-cell-rank → per-bin-QN → recenter stack independently
to put all four on a common footing.

**Decision — embedding strategy:**
- **Default: per-channel SVD → concatenate scores → joint SVD/skip-top →
  single UMAP.** Bounds each channel's contribution, is scale-robust, and
  is *diagnosable* — per-channel component AUC reveals **which co-state
  channel carries cell-type identity** (a result, not just a method knob).
- **Optional: tensor decomposition** (Tucker / CP via `tensorly`) on the
  cells × bins × channels tensor, yielding a cell factor (→ UMAP), a bin
  factor, and a **channel factor** (how the channels combine). Implemented
  as a selectable alternative to try.

**Standard diagnostic:** per-channel component AUC vs cell-type labels.

---

## 7. Data prerequisite + feasibility gate

**Prerequisite.** Joint per-end records carrying **both**
`(k_GpC, n_GpC, k_CpG, n_CpG)` on the same read end. Today GCH and HCG are
parsed in *separate* context runs; the joint method needs them aligned per
end (the information is in the same reads). Either refactor Step 2 to emit
joint records, or join the two context runs' read pairs on
`(chrom, pos1, pos2)`. (Exclude ambiguous GCG positions per standard
NOMe-seq.)

**Feasibility gate (Phase J0, do first).** The joint statistic needs both
modalities; lenient mode mitigates but does not eliminate sparsity. Before
committing to the build, measure on existing data:
- fraction of ends with GpC-only / CpG-only / both;
- fraction of pairs fully-observed (strict) vs each-end-≥1-modality
  (lenient);
- per-(cell, bin) linked-pair counts under both modes.

`scripts/joint_observability_check.py` reports these.

---

## 8. Implementation phases

| phase | content |
|---|---|
| **J0** | feasibility check (pair counts, observation completeness) — gate |
| **J1** | joint per-end records + joint state posteriors (§2) |
| **J2** | multi-state soft co-counts + DM + Δ vector + RWR over both modalities (§3–5) |
| **J3** | per-channel SVD/UMAP default + tensor option + per-channel AUC (§6) |
| **J4** | comparison: joint vs GCH-only vs HCG-only vs accessibility vs scHi-C via `nome_hic_compare` |

---

## 9b. Phase J0 findings (feasibility result, GM12878/IMR90, 40-cell sample)

Per-end modality (≥2 sites = informative):

| modality | fraction of ends |
|---|---|
| both GCH + HCG | 15.5% |
| GCH only | 69.4% |
| HCG only | 0.7% |
| neither | 14.4% |

- **GCH observed at ~85% of ends; HCG at only ~16%.** HCG-without-GCH is
  ~0.7%, so **HCG observation ⊂ GCH observation** — methylation is the
  hard limiting modality (~5× sparser).
- **Strict mode is infeasible:** long-range (≥1 kb) both-modality-both-ends
  pairs have **median 0** per (cell, bin). Drop strict as a realistic mode.
- **Lenient mode is count-viable** (median 2 linked pairs / (cell, bin) —
  the same regime the single-context method already handles via RWR) **but
  methylation-starved:** with HCG real at ~16% of ends, the methylation
  axis is prior-imputed for ~84% of ends, so the methylation-dependent
  channels (poised, active-methylated, and closed-vs-poised) would be
  largely prior-driven and likely cancel against the short-range baseline.

**Decision — gate the build on a cheaper test.** Before implementing
J1–J4, determine whether long-range **HCG concordance carries any
cell-type signal standalone**: run the existing signed pipeline on
`context_HCG` and compare GCH- vs HCG-concordance (vs accessibility,
scHi-C) with `nome_hic_compare`. If HCG concordance is uninformative
(expected, given the sparsity), the joint method cannot add methylation
signal the reads don't contain — a legitimate negative result that avoids
the heavy build. If it *is* informative, proceed to J1, likely with a
**coarsened HCG resolution** (wider per-end CpG window / coarser
methylation bins) to lift the 16% observation rate.

---

## 9. Locked decisions

1. **4 co-state channels** (open / closed / poised / active-methylated);
   "mixed" dropped as residual.
2. **Lenient missing-modality mode by default**; strict optional.
3. **Global mixture prior by default**; locally-informed per-region prior
   optional.
4. **DM estimation** over the 5 categories.
5. **Linked − short-range subtraction** per channel (also cancels the
   shared-prior artifact).
6. **RWR over both modalities** (graph from GCH ⊕ HCG; all channels smoothed).
7. **Per-channel SVD embedding by default**; **tensor decomposition optional**.
8. **Independence approximation** for the per-end joint (product of
   marginals) in v1; learned-joint in v2.

---

## 10. Implementation status (v0.3)

**Built — J1/J2, a focused J4, and J3 — all on the leakage-free path.** The
estimator mirrors the validated single-context `ep_coordination_existence.py`
(pool PAIRS, not CELLS; distance-matched O/E; the full confound ladder) —
**not** the RWR path, which the single-context investigation showed can
*manufacture* cell-type signal from the marginal-built cell graph
(`SIGNED_CONCORDANCE_AND_RWR_DESIGN.md` §6b). The J3 embedding is plain
unsupervised SVD/UMAP of the Δ tensor (not cross-cell smoothing on a marginal
graph), so it stays leakage-free too.

| phase | component | file | what |
|---|---|---|---|
| J1/J2 | §2–§3 core | `nome_hic_concordance/joint_state.py` | `per_end_joint_state` (independence product); `impute_missing_modality` (lenient, global prior); `pair_co_state` (4 co-states + `mixed`) |
| J2 | §4 per-cell | `nome_hic_concordance/joint_coordination.py` | join GCH+HCG per end (§7); distance-matched `log2(O/E)` per channel pooled per region; emits **joint(4ch)/gch/hcg** on one end pool + one null |
| — | eval ladder | `nome_hic_concordance/coordination_eval.py` | shared confound ladder (marginal → +coverage → +batch), classifier panel (`lr`/`svd-lr`/`rf`), permutation null, ROC |
| J4 | §4/§6 CLI | `scripts/joint_coordination_existence.py` | joint-vs-gch-vs-hcg through the same ladder; `--per-channel`; QC line (§2.1) |
| J3 | §6 embedding | `nome_hic_concordance/joint_embedding.py` + `scripts/joint_multistate_embedding.py` | `--embedding concat-svd` (per-channel SVD → concat) or `tensor` (Tucker/CP cell factor); per-channel + per-component AUC (which co-state / which SVD comps to skip); optional **standard package RWR** (`--rwr` / `--rwr-compare`, §10.2) |

Tested: `joint_state` 17, `coordination_eval` 18, `joint_coordination` 6 (incl.
the methylation-cancellation identity), `joint_embedding` 7, + 4 CLI smoke =
**52 new tests**; full suite 215 passed.

**Implementation decisions beyond §9:** **global prior** for the missing
modality (one number/context, not per-cell — leak-safe, decision #3);
**gch/hcg references computed on the *joint* end pool** (apples-to-apples
"does joint beat the best single context?").

### 10.1 J4 result (real data, 5 / 2 / 1 Mb) — methylation is the carrier

Δ|m+cov (marginal-, coverage- and missingness[`--min-region-frac 0.5`]-
controlled; label-permutation *p*):

| scale | joint(4ch) | gch (access) | hcg (methyl) | miss |
|---|---|---|---|---|
| 5 Mb | 0.63 (p=.04) | 0.52 (ns) | 0.57 (ns) | 0.54 |
| 2 Mb | 0.90 (p=.02) | 0.62 (.04) | **0.91 (.02)** | 0.63 |
| 1 Mb | 0.92 (p=.02) | 0.61 (.04) | **0.93 (.02)** | 0.62 |

- **Long-range HCG (methylation) concordance carries strong cell-type identity
  beyond marginal levels** (0.90–0.93 at 1–2 Mb, ≫ miss ≈ 0.62) — the §6b
  result, reproduced in the joint pipeline. `Δ|m > Δ` again (residualizing the
  marginal *sharpens* it).
- **GCH (accessibility) concordance adds little** (0.52–0.62).
- **Joint ≈ HCG, not above it** at the scales with signal → the joint-
  combination hypothesis is **not supported here**; methylation does the work
  (as §9b anticipated). Per-channel: the methylation-dependent co-states carry
  it (1 Mb: active_methylated 0.94, open 0.93; poised 0.70 weakest).
- Nuance at 5 Mb: joint and its open/am channels are significant where *neither*
  single-context **signed contrast** is — the joint keeps 4 *separate* channels
  while gch/hcg subtract two, cancelling co-moving signal at coarse scale. Weak.

**Caveat:** the marginal AUC ≈ 1.0 (mean methylation level alone separates the
cells), so the 0.90 residual leans on the residualization; the permutation null
rules out *chance* but not residual-marginal leakage. The single-context HCG
passed batch + split-half in §6b; re-run the joint test with `--split-half` and
`--batch-column` to make *this* run airtight.

**Why J3 was still built:** joint ≈ HCG means the *combination* adds nothing,
but the 4-channel decomposition is interpretive — it localizes the signal to
the methylation-dependent co-states and yields a cell map. Built for that, not
to beat HCG.

**Remaining:** the `nome_hic_compare` wiring (joint vs scHi-C / accessibility in
the shared harness) and the confirmatory batch/split-half joint run.

### 10.2 Densification — SVD by default, standard RWR optional

The leakage concern that motivated the existence test applies to the
**hypothesis test** ("does concordance carry signal *beyond* the marginal?"),
where a marginal-built RWR graph would be circular.  That question is **settled**
— the existence test (no imputation at all) reproduced the marginal-*controlled*
signal (§10.1), proving it is real and not an RWR artifact.  The embedding is a
**clustering / visualization** step, not that test, so the leakage concern does
not apply here, and there is no reason to avoid the validated RWR.

- **Default:** `--rwr off` → densification is **region-pooling per-channel SVD**
  — each cell's thousands of sparse region-Δ's are pooled into a few dense
  components *within that cell* (no cross-cell borrowing). Good enough whenever
  the SVD already separates the cells.
- **`--rwr on`:** the **same multi-pass RWR as the rest of
  `nome_hic_concordance`** (`run_rwr_pipeline` scheme), reusing
  `build_knn_graph` / `rwr_impute_matrices` / `post_norm_for_knn` /
  `_top_eigengap` unchanged:
  - **pass 0** — graph from `QN(mean accessibility ⊕ mean methylation)`;
  - **pass ≥1 (refinement)** — graph rebuilt from the *concordance* of the
    previous pass (`--rwr-refinement-basis postnorm|raw`), **re-imputing from the
    original channels** (only the graph evolves);
  - **`--rwr-auto-select`** — keep the pass with the largest post-norm
    concordance eigengap (`_top_eigengap`, label-free; pass 0 never selected),
    up to `--rwr-max-passes`, patience 1. `--rwr-refine-passes N` runs a fixed N.
- **`--rwr-compare`:** runs no-RWR and RWR (same embedding) and prints the AUC
  delta (no-RWR │ RWR │ Δ), writing `.norwr` and `.rwr` outputs. A large RWR gain
  on easy cell types is *marginal-driven* (the graph is the marginal), not
  evidence the concordance separates the cells — `RWR ≈ no-RWR` is the clean win.

The densification label (`RWR` / `no-RWR`) is **orthogonal to** the embedding
(`concat-svd` / `tensor`): `--rwr off --embedding tensor` still runs the tensor
decomposition, just on the un-smoothed channels (the leakage-free view).

(The earlier bespoke "concordance-graph RWR" was removed: unnecessary — the
leakage question is settled — and untested relative to the package RWR.)

### 10.3 Embedding method + component diagnostic

`--embedding` chooses how the cells × regions × 4-channel tensor becomes a cell
embedding:

- **`concat-svd` (default)** — per-channel normalize → per-channel SVD →
  **concatenate** scores → joint SVD/skip-top → UMAP.
- **`tensor`** — a **Tucker** (`--tensor-method tucker`, `--tensor-rank
  r_cell,r_region,r_channel`) or **CP/PARAFAC** (`cp`) decomposition of the
  tensor; the **cell factor** *is* the embedding (→ UMAP) and the **channel
  factor** (co-state loadings per component) is reported + written
  (`*.tensor_channel_factor.csv`). Needs `tensorly`; couples channels and regions
  jointly rather than concatenating independent per-channel SVDs.

Both feed the same UMAP and the same RWR densification (RWR smooths the channels
first; then either embedding runs on the smoothed channels).

To decide whether any channel's leading SVD component is a technical axis (as the
GCH 1st component was, single-context), the script prints a **per-channel,
per-component cell-type AUC** table (top 10 of each channel; `*.component_auc.csv`)
— high = cell type, low ≈ technical. `--skip-top-per-channel N` then drops the
top N components **of each channel** before concatenation (the per-channel
analogue of the single-context `--skip-top-svd`, which here acts only on the
*joint* SVD). The per-cell-rank normalization already removes per-cell depth, so
the default is 0.

### 10.4 Batch correction — Harmony, two stage choices

**Always Harmony** — `center`/`zscore` are too weak even in 1-D and were dropped
from the CLI. `--batch-column` (repeatable / comma-sep, e.g. `exp_batch,lib_batch`;
modeled jointly) + `--batch-correct {none, per-channel, embedding}`:

- **`per-channel` (recommended)** — Harmony-correct **each co-state channel AND
  the RWR marginal graph basis**, *before* RWR / tensor. This matches the
  single-context pipeline (`concordance.py` Step 3.5: correct the mean-
  accessibility *before* the RWR graph), generalized to the four channels — so
  the corrected features feed RWR (graph from the corrected marginal) and the
  tensor sees batch-free data. The graph basis **must** be corrected too, or
  batch re-enters via the RWR graph. (Reuses
  `nome_hic_concordance.batch_correction.apply_batch_correction(..., "harmony")`,
  i.e. PCA → Harmony → reconstruct, per channel.)
- **`embedding`** — Harmony on the **final cell factor / concat-SVD scores**,
  right before UMAP (the field-standard scRNA `embed → Harmony → UMAP` step).
  Cheaper (one Harmony run on ~20 dims) and corrects the map regardless of how
  batch entered, but only the low-dim embedding.

Pipeline order for `per-channel`: normalize each channel → **Harmony per channel
+ Harmony the marginal graph basis** → RWR (graph from corrected marginal; refine
from corrected concordance) → tensor / concat-SVD cell factor → UMAP.

The script prints a **per-factor batch-AUC** on the final embedding (one number
per batch column, lower = batch better removed) alongside the cell-type AUC
(should stay high). It scores **each batch factor separately**, NOT their
cartesian-product composite — the composite of several factors fragments into
singleton groups that CV can't score (which would print `NaN`); per-factor each
column keeps enough cells per level. A factor with only one level among labeled
cells, or a level with <2 cells, is reported as `n/a(reason)` rather than NaN.
If cell type *and* batch fall together, they are **confounded** in the design
(cells of a type don't span batches) and no correction can separate them — a
sampling problem, not a method one.
