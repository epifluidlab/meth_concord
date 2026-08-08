# Signed concordance + RWR imputation + comparison framework

> **Terminology note (v0.4).** Throughout this document the baseline pool
> is called **"unlinked."** That name is a **misnomer**: the pool is the
> set of **short-range** same-bin read pairs (genomic distance
> `< min_distance`) — which are *real short-range contacts*, not
> non-contacts. Read "unlinked" as **"short-range"** everywhere. The CLI
> now exposes it as `--baseline short-range` (with `unlinked` kept as a
> deprecated alias), and adds `--baseline distance-matched`, an O/E null
> built from *random non-contact* end-pairs at the matched distance, which
> — unlike the short-range pool — controls for distance and the bin
> marginal. See the distance-matched baseline section below.
>
> **Baseline update (v0.5).** A third option, `--baseline distance-pooled`,
> joins the two above. It uses the **same** non-contact end-pair null as
> `distance-matched`, but instead of matching each observed contact's
> distance it **pools every null end-pair with separation
> `>= min_distance`** (excluding the observed contacts) into a single mean
> expected per bin. This trades the distance control of
> `distance-matched` for **many more pairs → a much less-noisy expected**
> ("more reads for the estimation"). Mechanically it is a drop-in sibling:
> it fills the same expected (`_DM_KEYS`) pool with `n == #contacts` and
> rate `= the pooled mean`, so the downstream DM / `log2(O/E)` handling and
> the `--concordance-combine auto → log2-ratio` default are identical.
> Use `distance-pooled` when the per-bin distance-matched expected is too
> sparse/noisy and bin-marginal control suffices; use `distance-matched`
> when you must also hold genomic distance fixed.
>
> **Findings (v0.6).** The headline empirical question — *does long-range
> concordance carry cell-type information beyond marginal levels?* — is
> answered **yes**, in two systems, after a full confound ladder (and after
> catching one false positive and one false negative along the way). See the
> new **§6b. Empirical findings** below for the evidence, the ladder, and the
> honest effect sizes; the clean test lives in
> `scripts/ep_coordination_existence.py` (no cross-cell imputation).
>
> **Directional decomposition (v0.7).** The 3-state (open / closed / mixed)
> co-state distribution has exactly **two** degrees of freedom — the marginal
> level `p̄` and the association `Cov` — and two identities pin down what any
> combine mode can possibly measure:
>
> 1. `π_open − π_closed == p̄₁ + p̄₂ − 1`, an **exact per-pair identity** (the
>    `p_i·p_j` terms cancel). So `signed-diff` carries **zero** coordination
>    signal against a marginal-matched O/E null — it is a pure marginal
>    contrast.
> 2. `(O_open − E_open) == (O_closed − E_closed) == Cov`, **identically**:
>    coordination moves mass out of `mixed` into *both* concordant cells by
>    the same amount. So the *excess* cannot be attributed to a direction —
>    a "co-open excess" map and a "co-closed excess" map are the **same map**.
>
> Consequently "co-open vs co-closed" is **not** an independent third axis;
> it *is* the marginal, and no single statistic can be simultaneously
> marginal-free and directional. `concordant-diff` measures the coordination
> **magnitude** (`= 2·Cov`) and is therefore direction-blind by construction
> (`Cov` is invariant under `p → 1−p`).
>
> To get a directional read, the sign must come from **where the concordant
> mass sits**, not from the excess. `--concordance-combine concordant-signed`
> emits
>
> ```
> C = (O_open + O_closed) − (E_open + E_closed)   coordination magnitude (= 2·Cov)
> F = O_open / (O_open + O_closed)                realized direction
> D = C · (2F − 1)                                directional coordination
> ```
>
> `D > 0` = coordinated co-**open**, `D < 0` = coordinated co-**closed**,
> `D ≈ 0` = uncoordinated (mixed dominates) — the single
> co-closed → mixed → co-open scale. **Caveat:** `D ≈ 0` *also* arises for a
> coordinated-but-**balanced** bin (`F ≈ 0.5`, half co-open / half co-closed),
> which is direction-*ambiguous* rather than mixed. `C` is therefore emitted
> alongside `D` to separate the two (mixed → `C ≈ 0`; balanced → `C` large
> with `D ≈ 0`). Reference values from the fixture in
> `tests/test_signed_concordance.py`: coordinated-open `D = +0.151`,
> coordinated-closed `D = −0.151` (**same** `C = 0.304`), uncoordinated
> `D = C = 0`, balanced-coordinated `D = 0` with `C = 0.405`.
>
> Step 5 writes `concordance_{coordination,direction,directional,marginal}`
> for **every** combine mode (they are companion diagnostics, not tied to the
> headline Δ). Step 7 `--component-umaps` embeds `coordination` and
> `marginal` — the two *independent* d.o.f. — so it is visible which axis the
> cell-type structure lives on; a co-open/co-closed UMAP pair would be
> degenerate per identity (2) above.

## Design document v0.3 — locked decisions (RWR pre-Layer 2, QN k-NN basis)

> **Status:** All §6 decisions locked. Layer 2 = Dirichlet-Multinomial
> plug-in (no EM); soft counts = U-3 on both linked and unlinked pools;
> RWR runs **pre-Layer 2** on the soft-count matrices, with the k-NN graph
> built from the **quantile-normalized mean-accessibility matrix** on the
> initial pass and rebuilt from `QN(Δ_signed)` on optional refinement
> passes. Diagnostic results on the user's 288-cell GM12878/IMR90 data
> confirmed the design choices (see commit `5b38b41` and surrounding
> diagnostics). Ready for Phase 1 + RWR implementation.

> **Changelog vs v0.2:** §3 entirely rewritten (RWR moves pre-Layer 2,
> k-NN from QN(mean_accessibility), iterative-pass loop rebuilds k-NN
> from QN(Δ_signed)). §6 decisions #4, #5b (new), #16 reconciled.
> Layer-2 inference is plug-in not EM (added §6 #17). Defaults updated
> from data: `--min-gch-sites 2`, `--min-linked-pairs-per-cell 100`,
> `--min-pairs-per-bin 1` (§6 #14, #15).

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

### 1.3 Why RWR imputation, and why pre-Layer 2

Single-cell concordance at 250 kb is severely sparsity-limited. The
diagnostic on the user's 288-cell dataset (commit `4299701` / report)
shows median `n_linked = 1` per (cell, bin) — i.e. most cells have at
most a single long-range linked pair observed per bin. At that
density:

- The empirical-Bayes prior dominates the posterior almost completely.
  With `α_total ≈ 10` and `n_linked = 1`, the per-(cell, bin) posterior
  is shifted by at most ~9% off the prior — regardless of what the
  pair actually says. Cell-to-cell variation in Δ_signed is essentially
  the prior mean across cells, plus noise.
- Layer-2 inference improvements (DM vs BB, EM vs plug-in) don't help:
  they're all prior-dominated in this regime. EM in particular gives
  no measurable benefit (decision #17).

What does help is **borrowing strength across similar cells**: scHiCluster's
approach for sparse single-cell Hi-C. We build a k-NN graph over cells
using a dense auxiliary feature (mean accessibility, which uses ~100×
more data per (cell, bin) than concordance), then propagate signal
across the graph via random-walk-with-restart. For a cell with
`n_linked = 1` at a bin, RWR blends in evidence from the ~15 nearest
neighbours, raising the effective sample size to a useful range
(~15–30) before Layer-2 ever sees the data.

**Placement matters.** Putting RWR *after* Layer 2 (the v0.2 design)
operates on a matrix that's already prior-dominated — smoothing
prior-mean values together gives more prior-mean values, no signal
recovery. RWR has to run **before** Layer 2 so the EB shrinkage sees
imputed counts dense enough to track per-cell biology.

The integration goal: a default-on pre-Layer-2 step that takes the
soft-count matrices, applies RWR using a QN-mean-accessibility k-NN
graph, and feeds the imputed counts into Layer 2.

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

## 3. Part B — RWR imputation (pre-Layer 2)

### 3.1 Where it fits in the pipeline

RWR is **not** a post-processing pass on the Δ matrix. It runs **before
Layer 2**, on the raw soft-count matrices, so that Layer 2 sees imputed
counts dense enough for meaningful EB shrinkage. With median
`n_linked = 1` per (cell, bin) at 250 kb, this is the only mechanism by
which per-(cell, bin) Δ values can carry signal beyond the global prior
mean.

The expanded Step 5 looks like this:

```
Step 5a  per-cell parquets → per-(cell, bin) soft counts (Layer 1 → soft U-3)
            outputs: k_open, k_closed, k_mixed, n_linked,
                     k_u_open, k_u_closed, k_u_mixed, n_unlinked
            plus:    mean_accessibility per (cell, bin)
                     (dense; uses all informative ends, linked + unlinked)

Step 5b  RWR imputation on the soft-count matrices              ← NEW, default on
            k-NN graph from QN(mean_accessibility) (initial pass)
            optional refinement passes rebuild k-NN from QN(Δ_signed)
            applies same k-NN to ALL soft-count matrices

Step 5c  Layer 2 EB shrinkage on imputed soft counts → cells × bins Δ matrix
            Dirichlet-Multinomial plug-in (no EM)

Step 6   Stein shrinkage          (skipped for betabinom — already EB-shrunk)
Step 7   matrix_ops + UMAP        → final embedding
```

Toggle: `--impute {none, rwr}`, default **`rwr`** (not off). With
`--impute none` the pipeline behaves as in v0.2: soft counts → Layer 2
directly, which under the user's data sparsity gives a Δ matrix that's
essentially the global prior mean. RWR is the unlock, so the default is on.

### 3.2 Algorithm — initial pass

Given (a) the dense `mean_accessibility[N, B]` matrix and (b) the stack
of soft-count matrices `F_raw = {k_open, k_closed, k_mixed, n_linked,
k_u_open, k_u_closed, k_u_mixed, n_unlinked}`, each of shape `[N, B]`:

**1) k-NN graph from QN(mean_accessibility).**

```
A_qn = quantile_normalize_per_bin(mean_accessibility)
       # uses the same _quantile_normalize_per_bin helper as matrix_ops,
       # so QN logic is shared across the pipeline.
       # NaN entries (rare, only when a cell has 0 informative ends in a bin)
       # are masked from the rank computation and imputed to the bin centroid
       # afterward.

D = cosine_similarity(A_qn)        # N × N
for each cell c:
    keep top-k neighbours by D[c, ·]
W = symmetrize(adjacency)           # W = (W + W^T) / 2
P = row_normalize(W)                # transition matrix
```

Why QN(mean_accessibility), not raw mean_accessibility: raw matrix has
highly variable per-bin variance across cells; cosine distance is then
dominated by the noisiest bins. Per-bin QN equalizes bin contributions
to the distance, which is the standard treatment in bulk-RNA-seq and
scATAC-seq preprocessing — and matches the user's validated bin-mean
→ QN → SVD → UMAP recipe.

Why mean_accessibility (not concordance) as the initial k-NN basis: it
is **dense** (uses all informative ends, ~100–1000× more data per
(cell, bin) than concordance), so the k-NN it produces is statistically
reliable. The concordance matrix at this point is too sparse to define
neighbourhoods on.

**2) RWR on the soft-count stack.**

For each matrix `F_i ∈ F_raw` (8 matrices), with the same `P`:

Closed-form solver for `N ≤ 5000`:

```
F_i_imputed = r · (I − (1 − r) · P)⁻¹ · F_i
```

Iterative solver for larger N:

```
F_i_imputed ← F_i
repeat:  F_i_imputed ← (1 − r) · P @ F_i_imputed + r · F_i
until ‖F_i_imputed − F_i_imputed_prev‖_∞ < tol  or  iter ≥ max_iter
```

The same `r · (I − (1−r)·P)⁻¹` factor multiplies every soft-count
matrix, so for closed-form mode we compute it once and apply 8 times.
That's an `N × N × B × 8` operation — well under a minute for
typical N ≈ 1000, B ≈ 12000.

**3) Imputed values are the inputs to Layer 2.**

Layer 2 (§2.3) consumes the imputed `(k_open, k_closed, k_mixed,
n_linked)` for the linked side and `(k_u_open, k_u_closed, k_u_mixed,
n_unlinked)` for the unlinked side. The Dirichlet-Multinomial MoM EB
prior is fit on the imputed counts, and per-(cell, bin) Δ is computed
from the conjugate posterior plug-in formula. Because imputed
`n_linked` is now substantially larger than raw `n_linked = 1`,
per-(cell, bin) posteriors carry meaningful data influence rather
than being prior-dominated.

### 3.3 Iterative refinement passes (optional)

When `--rwr-refine-passes ≥ 1`, after the initial pass we have a
cells × bins `Δ_signed` matrix that's a closer approximation to the
true signal than mean accessibility was. Use it to rebuild the k-NN
for a second pass:

```
Pass 0 (always):
    A_qn ← QN(mean_accessibility)
    k-NN ← cosine k-NN on A_qn
    F_imputed ← RWR(F_raw, k-NN)
    Δ_signed_0 ← Layer-2(F_imputed)

For p = 1 .. --rwr-refine-passes:
    A_qn ← QN(Δ_signed_{p-1})
    k-NN ← cosine k-NN on A_qn
    F_imputed ← RWR(F_raw, k-NN)        ← always from ORIGINAL F_raw
    Δ_signed_p ← Layer-2(F_imputed)

Final Δ matrix = Δ_signed_{final pass}
```

Two important properties:

- **RWR always starts from the original soft counts, never from the
  previous pass's imputed values.** Only the k-NN graph evolves between
  passes. This prevents compound over-smoothing — without this
  discipline, after a few passes the imputed values would all collapse
  to the centroid.

- **All soft-count matrices share the same k-NN graph within a pass.**
  This guarantees the imputed (k_open, k_closed, k_mixed, n_linked)
  remain on the simplex (sum to imputed n_linked) and gives a coherent
  per-(cell, bin) joint distribution downstream.

Default: 0 refinement passes (one-shot RWR). With 0 passes, the
mean-accessibility k-NN is used and the Δ matrix is produced after a
single Layer-2 fit. With ≥ 1 pass, the k-NN evolves to track the
concordance signal itself.

### 3.4 Parameters and defaults

| Parameter | Flag | Default | Notes |
|---|---|---|---|
| Impute method | `--impute` | `rwr` | `rwr` or `none`. With `none`, Layer 2 sees raw soft counts (prior-dominated for sparse data). |
| k for k-NN | `--rwr-k` | 15 | scHiCluster convention; scale to ~√N for very large N |
| restart probability r | `--rwr-restart` | 0.5 | scHiCluster's default |
| max iterations | `--rwr-max-iter` | 30 | Only used in iterative-solver mode |
| convergence tolerance | `--rwr-tol` | 1e-4 | Iterative-solver mode |
| refinement passes | `--rwr-refine-passes` | 0 | 0–2 useful range; each pass adds one Layer-2 fit |
| solver | `--rwr-solver` | `auto` | `direct` (closed-form), `iterative`, or `auto` (direct if N ≤ 5000) |

The k-NN basis is **not** a CLI knob anymore — it's fixed to QN(mean_accessibility)
for pass 0 and QN(Δ_signed) for refinement passes. The v0.2 flag
`--rwr-neighbor-feature {self, accessibility}` is removed.

### 3.5 Outputs

In `output/.../05_concordance/` (alongside the existing concordance outputs):

```
soft_counts_raw.parquet                # long-form: cell, bin, k_open, k_closed, ...,
                                       # n_linked, k_u_*, n_unlinked  (pre-RWR)
soft_counts_imputed.parquet            # same schema, post-RWR
accessibility_mean.parquet             # cells × bins, dense (pre-QN)
cell_knn_graph_pass0.npz               # k-NN adjacency from QN(accessibility)
cell_knn_graph_pass{p}.npz             # only if --rwr-refine-passes ≥ p
rwr_diagnostics.json                   # {N, B, k, r, n_iter_used per pass,
                                       #  converged, pass_count, ...}
rwr_pre_post_distance_histogram.png    # cell-cell distance distribution
                                       # before / after RWR (QC)
```

The downstream `concordance_signed.parquet` is the Layer-2 output of
the final pass and looks identical in shape to the v0.2 design.

### 3.6 Validation diagnostics

Same as v0.2 §3.6, with one addition for iterative refinement:

- **Per-pass convergence:** for `--rwr-refine-passes ≥ 1`, plot
  ‖Δ_signed_p − Δ_signed_{p−1}‖ vs pass number. Should decay; if it
  oscillates, the k-NN-rebuild loop is unstable and we should fall
  back to fewer passes.

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

**(e) Magnitude vs cell-type-information (a separate axis).** All of
(a)–(d) ask whether concordance **varies across cells in a cell-type way**.
They are deliberately **blind to the average magnitude** of concordance:
the embedding/normalization stack (per-cell rank → per-bin QN → recenter)
strips location and scale, so a signal that is **real and strong but
cell-type-invariant** (high mean, low cross-cell variance) is normalized to
near-zero and shows weak kNN/silhouette. "Weak cell-type information" is
therefore **not** the same as "weak long-range concordance."

To test magnitude directly, aggregate the **soft agreement**
(`co-open + co-closed = 1 − mixed`) — *not* the signed open-vs-closed
direction — as a population statistic, stratified by genomic distance, for
three pools that a single `--baseline distance-matched` run already
contains: **linked** contacts, **short-range** local pairs, and the
**distance-matched random** end-pair null. Two magnitude questions follow:

- *Is it real?* per-cell distance-matched **excess** `A_long − A_rand > 0`
  (paired test across cells; `log2(A_long/A_rand) > 0`).
- *How strong?* `A_long` vs `A_short` — does 3-D contact restore
  near-local concordance? (Compare excess-over-random at each distance, as
  the short-range absolute level is inflated by genomic proximity.)

A positive, consistent excess with near-zero clustering is the signature of
a **real but constitutive** long-range concordance (the expected GCH
accessibility story — co-accessible contacts as a general chromatin
property), as opposed to a cell-type-specific one (the HCG methylation
story). `scripts/concordance_magnitude_diagnostic.py` reports the per-cell
excess test and the distance-decay curves (O vs E vs local).

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

The original v0.2 phasing assumed RWR was an optional post-step. With
RWR now pre-Layer 2 and on by default (decisions #4, #5b, #16), the
phases are restructured so that the **first user-facing release**
(Phase 1) ships the complete signed-Δ + RWR pipeline. Splitting Phase 1
and RWR would only produce intermediate releases where Δ is
prior-dominated noise — not useful.

### Phase 1 — Signed concordance + RWR (one combined PR)

The full pre-Layer-2 RWR + Layer-2 DM pipeline. Lands as one logical
unit so the first testable signed-Δ output is meaningful (post-RWR),
not prior-dominated (pre-RWR).

Files / modules:

- **`nome_hic_concordance/accessibility.py`** (new): compute mean
  accessibility per (cell, bin) from per-cell parquets. Used as the
  pass-0 k-NN basis. ~80 lines.
- **`nome_hic_concordance/imputation.py`** (new): k-NN graph builder
  (cosine on a passed-in feature matrix); RWR closed-form / iterative
  solver; multi-pass refinement loop. ~250 lines.
- **`nome_hic_concordance/concordance.py`** (rewrite): U-3 soft
  expected counts for both linked and short-range unlinked pairs;
  `_fit_dirichlet_multinomial_eb_prior` (MoM); `compute_signed_concordance_matrix`;
  keep the one-sided `compute_betabinom_concordance_matrix` path
  working but route it through the same U-3 count machinery so
  `--concordance-statistic both` produces both in one run.
- **`nome_hic_concordance/cli.py`** (changes): new flags
  (`--concordance-statistic`, `--posterior-threshold-{open,closed}`,
  `--min-linked-pairs-per-cell`, `--impute`, `--rwr-{k,restart,refine-passes,...}`),
  update defaults from §6 (#14, #15), alias
  `--concordance-method betabinom` → `--concordance-statistic open`
  with deprecation warning, add cell filter immediately after Step 2,
  save `qc/cells_used.tsv` + `qc/cells_dropped.tsv`.
- **Tests**:
  - `tests/test_signed_concordance.py` — math: U-3 soft-count
    contributions, Dirichlet-Multinomial MoM correctness, sanity-check
    table from §2.3 (active / silenced / chaos / focal-active), the
    full pipeline producing the Δ_signed matrix.
  - `tests/test_imputation.py` — k-NN correctness, RWR convergence and
    reconstruction of synthetic low-rank matrices, multi-pass
    refinement stability.
  - `tests/test_accessibility.py` — mean accessibility per (cell, bin)
    matches a hand-computed reference.
  - `tests/test_betabinom_concordance.py` (regression) — existing
    one-sided path still produces equivalent values when U-3 soft
    counts converge to hard counts under sharp posteriors.

Approximate size: ~700 lines of new code + ~400 lines of tests.

### Phase 2 — scHiCluster compartment sibling command
- New entry point `nome_hic_compartments` in `pyproject.toml`.
- Converter from per-cell parquet → `.scool` (cooler).
- Wrapper around `schicluster.compartment.compartment_call`.
- Optional dep: `pip install nome_hic_concordance[compartments]`.
- Separate sub-design doc when we get to it.

### Phase 3 — Comparison harness
- New entry point `nome_hic_compare` in `pyproject.toml`.
- Implements §4.2 analyses: marginal UMAPs, per-bin variance-partition
  regression, residual UMAP, optional classifier comparison.
- Tests: end-to-end on synthetic matrices with known relationships.

Phase 1 is the only piece needed to ship a working signed-Δ pipeline
on the user's data. Phases 2-3 are methods-comparison infrastructure
for the paper.

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

4. **RWR k-NN graph basis (initial pass) = QN(mean_accessibility).**
   The mean-accessibility matrix is dense (uses all informative ends,
   linked + unlinked), so the k-NN it produces is statistically
   reliable. Per-bin quantile normalization equalizes bin contributions
   to cosine distance, matching the user's validated bin-mean → QN →
   SVD → UMAP recipe.
   *(The v0.2 `--rwr-neighbor-feature {self, accessibility}` flag is
   removed; the k-NN basis is no longer user-selectable for pass 0.)*

5. **RWR refinement passes default to 0** (one-shot RWR with the
   QN(mean_accessibility)-derived k-NN).

5b. **For `--rwr-refine-passes ≥ 1`, subsequent passes rebuild the
    k-NN from QN(Δ_signed) of the previous pass.** RWR always operates
    on the **original** raw soft-count matrices — only the k-NN graph
    evolves between passes. This prevents compound over-smoothing and
    matches scHiCluster's iterative-refinement protocol.

6. **When `--concordance-statistic both`, RWR is applied to both Δ
   matrices using the same k-NN graph.** All soft-count matrices
   within a pass share the same k-NN to keep the imputed counts on the
   simplex (sum to imputed `n_linked`).

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

14. **Low-coverage cell filter at the cell level.** New CLI flag
    `--min-linked-pairs-per-cell N` (default: derived from data via
    `scripts/check_linked_pair_density.py` knee detection; suggested
    floor 100 for typical NOMe-HiC depth). Applied right after Step 2
    (before Step 3 mixture model and downstream steps) — bad cells
    don't contaminate the mixture fit, posterior calls, or the cells ×
    bins matrix. Filtered cells are logged with their `n_linked_total`
    for traceability, and the survivor list is saved as
    `output/.../qc/cells_used.tsv`.

15. **Default `--min-pairs-per-bin` for the signed/open statistics
    drops to 1.** The previous default of 5 was a Pearson-r artefact;
    for Beta-Binomial / Dirichlet-Multinomial shrinkage every (cell,
    bin) with `n_linked >= 1` is a valid observation. Per the §6
    item-14 sparsity discussion, dropping bins at `n=5` removes ~93%
    of the matrix for typical NOMe-HiC data and is too aggressive.
    Backward compat: `--concordance-method pearson` keeps default 5.

16. **RWR moves to pre-Layer 2.** RWR imputation operates on the
    soft-count matrices (`k_open`, `k_closed`, `k_mixed`, `n_linked`,
    `k_u_open`, `k_u_closed`, `k_u_mixed`, `n_unlinked`) using a single
    k-NN graph per pass (see #4 / #5b for which matrix the k-NN is
    built from). Layer 2 then runs on imputed counts. With median
    `n_linked = 1` per (cell, bin), this is the only place where
    per-(cell, bin) Δ can pick up any signal beyond the global prior.
    Default: `--impute rwr` (not off).

17. **EM at Layer 2 dropped.** With median `n_linked = 1` per (cell,
    bin), Layer-2 posteriors are prior-dominated regardless of plug-in
    vs EM. EM offers no measurable benefit while adding significant
    implementation complexity. Layer 2 uses Dirichlet-Multinomial
    plug-in (signed) or Beta-Binomial plug-in (open).

---

## 6b. Empirical findings — does long-range concordance carry cell-type information beyond marginal levels?

**The question.** Does long-range HCG-methylation (or GCH-accessibility)
concordance carry cell-type identity *beyond* what the **marginal** (per-locus
mean) methylation/accessibility already provides — i.e., is the
**coordination** of distal loci informative, not just their individual levels?

### 6b.1 Two methodological traps, both resolved

1. **RWR-imputation false positive (graph leakage).** The pre-Layer-2 RWR
   imputation (§3) smooths the per-(cell, bin) concordance on a cell–cell
   k-NN graph built from `QN(mean accessibility/methylation)`. That **injects
   the marginal's cell-type structure into the concordance matrix**. On
   RWR-imputed concordance the cell-type classifier AUC was high (~0.94–0.98)
   and the per-bin variance-partition residual looked strongly significant
   (residual-kNN z ≈ 89–118). But regenerating concordance with
   `--impute none` collapsed the standalone AUC to chance, and the residual
   structure vanished. **→ the RWR "signal" was the marginals re-encoded by
   the imputation graph, not coordination.** (Caution worth keeping: graph
   imputation on a marginal-derived cell graph can manufacture apparent
   cell-type signal that survives a *linear* confound regression.)

2. **SVD-hidden false negative.** Without imputation the clean test first
   looked **negative** (AUC ≈ 0.5 in cortex). That was an artifact of reducing
   the residual feature matrix with **unsupervised SVD** before classifying:
   after residualizing out the dominant, high-variance marginal, the cell-type
   signal lives in **low-variance directions that SVD-top-k discards** (acute
   in the p ≫ n regime — thousands of regions, hundreds of cells). Classifying
   on the **full** standardized region features (L2-logistic or random forest)
   recovered it — in cell lines (svd-lr 0.36 → rf 0.96) and cortex (≈0.5 →
   0.82–0.84). The lesson: judge "does the residual carry signal?" with a
   classifier that cannot variance-discard, calibrated by a permutation null.

### 6b.2 The clean existence test

`scripts/ep_coordination_existence.py` measures the signal **without
cross-cell imputation**, densifying by pooling *pairs* (clean) not *cells*
(leaky). Per (cell, coarse region) it forms a distance-matched **log2(O/E)**
signed concordance Δ over a chosen distance band and classifies cell type
(OvR-macro AUC for >2 types, label-permutation null). The companion
distance-decay figure plots, vs log-distance (decade ticks, 95% CI, an
"all cells" line): the soft co-state **agreement** (= co-open + co-closed =
P[two ends share state] = the posterior product, **not** log2(O/E)), the
signed **log2(O/E)**, and the **O − E** residual.

### 6b.3 The confound ladder

The cell-type signal was required to survive, **simultaneously**:

| control | flag | removes |
|---|---|---|
| marginal level | `Δ\|m` (per-region residual on mean openness) | linear level |
| nonlinear-marginal + coverage | `Δ\|m+cov` | level², per-region pair count |
| missingness / contact-presence | `--min-region-frac 0.5`; vs the `miss` benchmark | "which regions a cell has data in" (scHi-C-like) |
| short-range methylation domains | `--min-distance 20000` | local CpG-domain co-methylation (1–20 kb) |
| classifier | `--classifier rf` **and** `lr` (full features) | SVD-discarding; linear-only |
| chance | `--n-perm 200` | finite-sample / overfitting inflation |
| region-specific artifact | `--split-half` | a few regions driving it (genome-wide consistency) |
| batch (global per-cell) | `--batch-column exp_batch lib_batch` | library-prep / sequencing batch (additive union) |

`split-half` is needed because batch is a *global per-cell* property that
appears in both chromosome halves — so split-half alone cannot catch it;
`--batch-column` is the dedicated control.

### 6b.4 Result — two systems, both positive

- **IMR90 / GM12878 cell lines (HCG, binary AUC).** Long-range (20 kb–2 Mb)
  `Δ|m+cov` ≈ **0.79 (5 Mb) – 0.89 (1 Mb)**, p = 0.005, ≫ `miss` (~0.67–0.70);
  split-half both halves ≈ 0.70–0.88, p = 0.005. Locus-scale (gone at
  chrom/10–25 Mb region granularity), and **not** short-range domains (survives
  `--min-distance 20000`).

- **Mouse cortex (HCG, multiple cell types, OvR-macro AUC).** `Δ|m+cov` ≈
  **0.82 (5 Mb) – 0.84 (1 Mb)**, p = 0.005, ≫ `miss` (~0.55–0.58); split-half
  both halves ≈ 0.81–0.84, p = 0.005. **Batch did contribute** —
  `Δ|m+cov+batch` (exp+lib, additive) drops to **0.73 / 0.68 / 0.66**
  (5/2/1 Mb) — but stays **significant (p = 0.005) and above `miss`**.

A note that *supports* the signal: across both systems `Δ|m ≥ Δ` (removing
the marginal *sharpens* rather than reduces the AUC), i.e. the coordination
signal was partly *obscured* by residual marginal structure, not produced by
it.

### 6b.5 Conclusion + honest scope

**Long-range HCG-methylation concordance carries cell-type identity beyond
marginal levels**, demonstrated in two systems including primary mouse cortex
with multiple cell types, surviving the full ladder
(marginal → nonlinear-marginal → coverage → missingness → short-range →
two classifiers → permutation null → genome-wide split-half → multi-factor
batch). The **honest effect size is the batch-corrected one** (cortex
~0.66–0.73). The signal is **locus-scale** (1–5 Mb region pooling of 20 kb–2 Mb
pairs) and **genome-wide-distributed**.

Stated honestly:
- The earlier "cortex negative" was a *classifier* (SVD-discarding) artifact,
  not a true negative.
- Multi-type effect sizes are OvR-macro (average separability across types),
  not driven by a single easy type.
- Two belt-and-suspenders controls were **not** run (not blockers): the batch
  **interaction** (`exp×lib` composite vs additive main effects), and a
  **co-observed-only / depth-matched** classification (the last global per-cell
  sequencing-depth loophole). The additive multi-factor batch control is the
  standard one, and the signal survived it.

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
