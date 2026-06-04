# Joint GCH+HCG multi-state long-range concordance

## Design document v0.2 — locked decisions + implementation status

> **v0.2 (implemented):** the §2–§4 core and a focused J4 are built as the
> leakage-free *existence test* — see **§10**.  The J0 gate passed (long-range
> HCG concordance carries cell-type information beyond marginal levels; see
> `SIGNED_CONCORDANCE_AND_RWR_DESIGN.md` §6b), so the build proceeded to J1/J2.

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

## 10. Implementation status (v0.2)

**Built — J1/J2 plus a focused J4, as the leakage-free *existence test*.** It
mirrors the validated single-context `ep_coordination_existence.py` (pool
PAIRS, not CELLS; distance-matched O/E; the full confound ladder) — **not** the
RWR embedding path, which the single-context investigation showed can
*manufacture* cell-type signal from the marginal-built cell graph
(`SIGNED_CONCORDANCE_AND_RWR_DESIGN.md` §6b). Starting from the clean estimator
means any joint signal is trustworthy *before* investing in the J3 embedding.

| component | file | what |
|---|---|---|
| §2–§3 core | `nome_hic_concordance/joint_state.py` | `per_end_joint_state` (independence product); `impute_missing_modality` (lenient, global prior); `pair_co_state` (4 co-states + `mixed`); `co_state_agreement` |
| §4 per-cell | `nome_hic_concordance/joint_coordination.py` | join GCH+HCG per end (§7); distance-matched `log2(O/E)` per co-state channel pooled per region; emits **joint(4ch) / gch / hcg** feature sets on one end pool + one null |
| eval ladder | `nome_hic_concordance/coordination_eval.py` | shared confound ladder (marginal → +coverage → +batch), classifier panel (`lr`/`svd-lr`/`rf`), permutation null, ROC — channel-agnostic |
| §4/§6 CLI | `scripts/joint_coordination_existence.py` | runnable joint-vs-gch-vs-hcg test through the same ladder; `--per-channel` = which co-state carries identity (§6 diagnostic); QC line (§2.1) |

Tested: `joint_state` 17, `coordination_eval` 18, `joint_coordination` 6 (incl.
the methylation-cancellation identity — with HCG fully missing the joint
channels collapse to the accessibility ratios), CLI 2 end-to-end smoke.

**Implementation decisions beyond §9:**
- **Global prior** for the missing modality (one number per context across
  cells), not per-cell — so lenient-imputed ends inject no cell-type marginal
  structure (locked decision #3, made concrete and leak-safe).
- **gch/hcg references are computed on the *joint* end pool**, so "does joint
  beat the best single context?" is apples-to-apples (same cells / ends / null
  / CV folds). Absolute single-context numbers still come from
  `ep_coordination_existence.py`.

**Not yet built:** J3 (per-channel SVD/UMAP embedding + tensor-decomposition
option + per-channel component AUC on the *embedding*) and the
`nome_hic_compare` wiring. **Gate J3 on a positive existence result:** if on
real data `joint Δ|m+cov ≈ max(gch, hcg)` (the outcome §9b anticipates, with
HCG real at only ~16 % of ends), the methylation axis adds little beyond the
accessibility signal and the heavy embedding build is not warranted — a
legitimate negative recorded cheaply. If `joint > max(gch, hcg)` above the
permutation null and above the missingness benchmark, proceed to J3 (likely
with coarsened HCG resolution to lift the observation rate).
