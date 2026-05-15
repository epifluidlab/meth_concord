# Pair-Concordance Mathematical Companion — v1.1

**Status:** Draft v1.1 (supersedes v1 — adds three-step TF biological framing, short-range constraint on unlinked reads in notation and likelihood, pre-specified ground-truth labels)
**Companion to:** `PAIR_CONCORDANCE_DESIGN_v2.1.md`
**Purpose:** Formalize the statistical model underlying the contact-aware pair-concordance extension. Provides explicit definitions, derivations, and identifiability conditions to serve as (a) the implementation reference for cross-validating code against the math, and (b) the foundation for the manuscript methods section.

All equations in this document map to functions in `nome_hic_concordance/pair_concordance/`. Section cross-references indicate the implementing module.

---

## 0. Biological hypothesis being tested

The statistical framework operationalizes a specific mechanistic hypothesis: TF-mediated long-range gene regulation proceeds through three sequential steps:

- **Step 1 — Contact.** A TF complex brings two distant regulatory regions $i$ and $j$ into 3D proximity. Probability of contact in cell $c$ at pair $(i, j)$: $\pi^c_{ij}$.
- **Step 2 — Nucleosome eviction.** Given contact, the TF complex displaces nucleosomes at both anchors, opening local chromatin. Conditional probability of joint open state given contact: $\theta^{c,L}_{ij}$. Measured by GCH methylation in NOMe-HiC.
- **Step 3 — Methylation reprogramming.** Given contact and accessibility, the regions undergo HCG methylation changes (typically hypomethylation at active elements). Conditional probability of joint hypomethylated state given contact+accessibility: not modeled in v1.1 (deferred to a future tri-modal extension on HCG data).

The pair-concordance framework decomposes observed (contact, joint-accessibility) data into:

- **Step 1 signal:** linked-read count $n^{c,L}_{ij}$ encodes $\pi^c_{ij}$ up to per-cell depth.
- **Step 2 signal:** $\theta^{c,L}_{ij}$ conditional on $n^{c,L}_{ij} \geq 1$, contrasted against the local-marginal-state independence baseline $\theta^{c,U}_{ij} = p^c_i \cdot p^c_j$. The contrast $\Delta^c_{ij} = \theta^{c,L}_{ij} - \theta^{c,U}_{ij}$ isolates Step 2 from Step 1.

This decomposition is the formal target of the hierarchical model below.

---

## 1. Notation

### 1.1 Indices

| Symbol | Meaning | Domain |
|--------|---------|--------|
| $c$ | Cell | $c \in \{1, \ldots, C\}$ |
| $(i, j)$ | Pair of genomic anchor regions | indexed by $p \in \{1, \ldots, P\}$ in the panel |
| $\text{cat}(p)$ | Category label of pair $p$ | $\{$co_folded, non_folded, tf_anchored_*, random_distance_matched$\}$ |
| $b(p)$ | Distance band of pair $p$ | $\{$short, mid, long$\}$ |
| $k(c)$ | Cluster label of cell $c$ | $\{1, \ldots, K\}$ |

For brevity, equations involving pair $p = (i, j)$ are written with subscript $ij$ where unambiguous.

### 1.2 Per-read-end quantities (from existing meth_concord Step 4)

| Symbol | Meaning |
|--------|---------|
| $k^{\text{end}}$ | Number of methylated GCH sites on a single read end |
| $n^{\text{end}}$ | Total informative GCH sites on a single read end |
| $\eta^{\text{end}} = P(\text{accessible} \mid k^{\text{end}}, n^{\text{end}})$ | Posterior accessibility probability, from fitted mixture |
| $\tau$ | Binarization threshold on $\eta^{\text{end}}$ (default 0.5) |
| $s^{\text{end}} = \mathbb{1}[\eta^{\text{end}} > \tau] \in \{0, 1\}$ | Binary accessibility state ("open" = 1, "closed" = 0) |

### 1.3 Per-pair-per-cell counts (from Step 6p)

For each $(p, c)$:

| Symbol | Meaning |
|--------|---------|
| $n^{c,L}_{ij}$ | Number of linked read pairs at long range (≥ $d_{\min}$, default 20 kb), both ends informative |
| $k^{c,L,A}_{ij}$ | Of linked pairs, number with both ends open (active-concordant) |
| $k^{c,L,S}_{ij}$ | Of linked pairs, number with both ends same state (symmetric-concordant) |
| $n^c_i$ | Number of unlinked-i reads: pairs with one end in $i$ AND mate within $d_{\max}^U$ (default 20 kb) of $i$ AND mate not in $j$ |
| $n^c_j$ | Number of unlinked-j reads (symmetric definition) |
| $k^{c,\text{open}}_i, k^{c,\text{open}}_j$ | Of unlinked reads, number with the in-region end open |
| $d_{\min}$ | Minimum long-range distance defining "linked" (default 20 kb, from existing pipeline `min_distance`) |
| $d_{\max}^U$ | Maximum mate distance defining "unlinked" (default 20 kb, the `unlinked_max_distance` parameter) |

**Distance constraints on linked vs unlinked reads are essential.** Linked reads are long-range (distance $\geq d_{\min}$), capturing the joint state of two regions brought into 3D proximity. Unlinked reads at $i$ are short-range pairs where the mate is local to $i$ (within $d_{\max}^U$), so they estimate the cell's *local* marginal accessibility at $i$ — the same quantity bulk NOMe-seq at $i$ would measure. Without the short-range constraint on unlinked reads, the unlinked-i pool would include reads with mates at arbitrary other long-range sites, which would contaminate the marginal estimate with other long-range contact contexts and bias $\hat{p}_i^c$ downward (toward the population-wide accessibility distribution). The contrast linked − unlinked then cleanly becomes "long-range contact-conditional concordance minus local-baseline-predicted concordance" — the correct framing for testing Step 2 of the three-step TF model in §0.

### 1.4 Latent parameters

| Symbol | Meaning |
|--------|---------|
| $\theta^{c,L}_{ij}$ | Linked active-concordance probability at $(i,j)$ in cell $c$ (Step 2 signal conditional on Step 1) |
| $p^c_i, p^c_j$ | Local marginal accessibility probability at $i$, $j$ in cell $c$ |
| $\theta^{c,U}_{ij}$ | Independence-baseline concordance probability (from local marginals) |
| $\Delta^c_{ij}$ | Coordination excess (primary per-cell, per-pair quantity isolating Step 2 from Step 1) |

### 1.5 Embedding-level quantities

| Symbol | Meaning |
|--------|---------|
| $u^c_{ij}$ | E_contact feature value |
| $v^c_{ij}$ | E_coord feature value (precision-weighted $\Delta$) |
| $w^c_{ij}$ | E_coord observation weight (per-pair-per-cell) |
| $\lambda_c$ | Per-cell total linked-read count over the panel |

---

## 2. Generative model

### 2.1 Per-read-end state (existing meth_concord Step 3–4)

The existing two-component beta-binomial mixture defines the read-end-level generative model:

$$z^{\text{end}} \sim \text{Bernoulli}(\pi), \quad p^{\text{end}} \mid z^{\text{end}} = j \sim \text{Beta}(\alpha_j, \beta_j), \quad k^{\text{end}} \mid p^{\text{end}}, n^{\text{end}} \sim \text{Binomial}(n^{\text{end}}, p^{\text{end}})$$

where $z^{\text{end}} = 1$ denotes "accessible", $z^{\text{end}} = 0$ denotes "closed", and $(\pi, \alpha_0, \beta_0, \alpha_1, \beta_1)$ are fitted by EM on the pooled long-range read ends. The Step 4 output $\eta^{\text{end}} = P(z^{\text{end}} = 1 \mid k^{\text{end}}, n^{\text{end}})$ is the per-read-end accessibility posterior.

In the pair-concordance extension, we treat $s^{\text{end}} = \mathbb{1}[\eta^{\text{end}} > \tau]$ as the binary state. This is a thresholding decision that converts the existing soft posterior into a discrete state for the hierarchical model. The threshold $\tau$ is the existing pipeline's `--posterior-threshold`.

**Justification for binarization:** The downstream hierarchical model on linked-pair concordance uses a Bernoulli/beta-binomial likelihood that requires discrete state assignments. A continuous alternative — using $\eta^{\text{end}}_1 \cdot \eta^{\text{end}}_2 + (1-\eta^{\text{end}}_1)(1-\eta^{\text{end}}_2)$ as a soft concordance score — is well-defined but no longer maps to a binomial sampling distribution. The categorical version is statistically cleaner and matches the bulk-NOMe literature convention. Sensitivity to $\tau$ is reported as a robustness check.

### 2.2 Linked-pair likelihood

For each pair $(i,j)$ and cell $c$, conditional on the latent $\theta^{c,L}_{ij}$:

$$k^{c,L,A}_{ij} \mid \theta^{c,L}_{ij}, n^{c,L}_{ij} \sim \text{Binomial}(n^{c,L}_{ij}, \theta^{c,L}_{ij})$$

This treats each linked read pair as an independent Bernoulli trial with success probability $\theta^{c,L}_{ij}$ (= probability both ends are open). Independence holds across read pairs within a cell at a given $(i,j)$ because each read pair is a separate ligation event. Independence across cells holds by the experimental design.

### 2.3 Unlinked-read likelihood

For unlinked-i reads in cell $c$ (one end in $i$, mate within $d_{\max}^U$ of $i$, mate not in $j$):

$$k^{c,\text{open}}_i \mid p^c_i, n^c_i \sim \text{Binomial}(n^c_i, p^c_i)$$

and symmetrically for unlinked-j (mate within $d_{\max}^U$ of $j$, mate not in $i$). The marginals $p^c_i$ and $p^c_j$ are the cell's **local** accessibility probabilities at $i$ and $j$ respectively. They are jointly identifiable from unlinked data because the two endpoints contribute independent observations from separate molecules.

The short-range constraint $d_{\max}^U$ is what makes $p^c_i$ the local marginal. Specifically: a read whose end falls in $i$ and whose mate is at long range to some other site $k \neq j$ would inform a *different* conditional, $P(\text{open at } i \mid \text{contact with } k)$, which mixes Step 2 signal across all $i$'s long-range contacts. Restricting the unlinked-i pool to short-range mates removes this conditioning and recovers the marginal-state quantity we want.

### 2.4 Prior structure

$$\theta^{c,L}_{ij} \sim \text{Beta}(\alpha^L_{\text{cat}(p), b(p)}, \beta^L_{\text{cat}(p), b(p)})$$

— prior pooled across all pairs in the same (category, distance_band) cell, but applied separately per cell. The pooling level can be relaxed to per-category or tightened to per-pair as sensitivity (§13). The default is category × band.

$$p^c_i \sim \text{Beta}(\alpha^M_{\text{cat}(p), b(p)}, \beta^M_{\text{cat}(p), b(p)})$$

— marginal accessibility prior, also pooled by category × band (the M superscript denotes "marginal"). This prior is used only as a category-level fallback when per-cell unlinked counts are too sparse for direct estimation (§5).

Both priors are estimated empirically (§3), not specified subjectively.

---

## 3. Empirical Bayes priors

### 3.1 Linked-concordance prior

For each category-band group $g = (\text{cat}, b)$, collect all $(c, p)$ observations with $p \in g$ and $n^{c,L}_{ij} \geq 1$. Let:

$$\bar{y}_g = \frac{\sum_{(c,p) \in g} k^{c,L,A}_{ij}}{\sum_{(c,p) \in g} n^{c,L}_{ij}}$$

— the pooled mean (weighted by linked count). This estimates $\mathbb{E}[\theta^{c,L}_{ij}]$ across the group.

For the variance, compute the among-observation variance of $y^c_{ij} = k^{c,L,A}_{ij} / n^{c,L}_{ij}$, weighted by $n^{c,L}_{ij}$:

$$S^2_g = \frac{\sum_{(c,p) \in g} n^{c,L}_{ij} (y^c_{ij} - \bar{y}_g)^2}{\sum_{(c,p) \in g} n^{c,L}_{ij}}$$

Under the beta-binomial marginal,

$$\text{Var}(y^c_{ij}) = \frac{\bar{y}_g (1 - \bar{y}_g)}{n^{c,L}_{ij}} \cdot \frac{\alpha^L + \beta^L + n^{c,L}_{ij}}{\alpha^L + \beta^L + 1}$$

The first factor is the binomial component; the second is the beta-binomial overdispersion factor. Setting $S^2_g$ equal to the average of $\text{Var}(y^c_{ij})$ over the group and solving for the prior concentration $\nu^L_g = \alpha^L_g + \beta^L_g$ gives the method-of-moments estimator:

$$\hat{\nu}^L_g = \frac{\bar{y}_g (1 - \bar{y}_g) - S^2_g}{S^2_g - \bar{y}_g (1 - \bar{y}_g) / \bar{n}^L_g}$$

where $\bar{n}^L_g$ is the harmonic mean of $n^{c,L}_{ij}$ in the group. Floor at $\hat{\nu}^L_g \geq \nu_{\min} = 2$ for numerical stability. If the numerator is negative (i.e., observed variance is smaller than binomial-only would predict), set $\hat{\nu}^L_g$ to a large default value (e.g. 100) — this indicates effectively no overdispersion in this group and corresponds to a very informative prior.

Then:

$$\hat{\alpha}^L_g = \bar{y}_g \cdot \hat{\nu}^L_g, \quad \hat{\beta}^L_g = (1 - \bar{y}_g) \cdot \hat{\nu}^L_g$$

Implementation: `empirical_bayes.py:fit_priors_method_of_moments()`. Returns DataFrame indexed by $(\text{cat}, b)$ with $\hat{\alpha}^L_g, \hat{\beta}^L_g$.

### 3.2 Marginal accessibility prior (used only for sparse fallback)

For each category-band group $g$, collect cell-level estimates $\hat{p}^c_i$ from cells with $n^c_i \geq n_{\min}^M$ (default 10). Apply the same MoM procedure on these to derive $\hat{\alpha}^M_g, \hat{\beta}^M_g$. These priors are NOT used when the per-cell unlinked counts are adequate; they only kick in via the fallback rule in §5.2.

---

## 4. Posterior on linked-concordance probability

Given the conjugate beta prior and binomial likelihood, the posterior is closed-form:

$$\theta^{c,L}_{ij} \mid k^{c,L,A}_{ij}, n^{c,L}_{ij} \sim \text{Beta}(\hat{\alpha}^L_g + k^{c,L,A}_{ij}, \; \hat{\beta}^L_g + n^{c,L}_{ij} - k^{c,L,A}_{ij})$$

Mean and variance:

$$\hat{\theta}^{c,L}_{ij} := \mathbb{E}[\theta^{c,L}_{ij} \mid \text{data}] = \frac{\hat{\alpha}^L_g + k^{c,L,A}_{ij}}{\hat{\alpha}^L_g + \hat{\beta}^L_g + n^{c,L}_{ij}}$$

$$\sigma^2(\hat{\theta}^{c,L}_{ij}) = \frac{(\hat{\alpha}^L_g + k^{c,L,A}_{ij})(\hat{\beta}^L_g + n^{c,L}_{ij} - k^{c,L,A}_{ij})}{(\hat{\alpha}^L_g + \hat{\beta}^L_g + n^{c,L}_{ij})^2 (\hat{\alpha}^L_g + \hat{\beta}^L_g + n^{c,L}_{ij} + 1)}$$

Implementation: `hierarchical.py:linked_posterior()`. Vectorized over $(c, p)$ pairs.

**Behavior at extremes:**
- $n^{c,L}_{ij} = 0$: posterior equals prior. $\hat{\theta}^{c,L}_{ij} = \hat{\alpha}^L_g / (\hat{\alpha}^L_g + \hat{\beta}^L_g)$, variance $= \text{Beta variance of prior}$.
- $n^{c,L}_{ij} \to \infty$: posterior concentrates on MLE $k^{c,L,A}/n^{c,L}$, variance $\to 0$.
- The shrinkage strength is controlled by $\hat{\nu}^L_g = \hat{\alpha}^L_g + \hat{\beta}^L_g$ — strong prior (large $\nu$) shrinks low-count posteriors hard toward the group mean.

---

## 5. Independence baseline from unlinked reads

### 5.1 Per-cell estimates when adequate

Recall (§1.3, §2.3) that $n^c_i$ counts only unlinked-i reads — those with one end in $i$ and mate within $d_{\max}^U$ of $i$ (default 20 kb), excluding mates in $j$. This short-range restriction is what makes $\hat{p}^c_i$ the *local* marginal accessibility at $i$, not contaminated by other long-range contact contexts.

When $n^c_i \geq n_{\min}^M$ AND $n^c_j \geq n_{\min}^M$ (default $n_{\min}^M = 5$):

$$\hat{p}^c_i = \frac{k^{c,\text{open}}_i}{n^c_i}, \quad \hat{p}^c_j = \frac{k^{c,\text{open}}_j}{n^c_j}$$

For **active concordance** (both ends open):

$$\hat{\theta}^{c,U,A}_{ij} = \hat{p}^c_i \cdot \hat{p}^c_j$$

For **symmetric concordance** (same state):

$$\hat{\theta}^{c,U,S}_{ij} = \hat{p}^c_i \hat{p}^c_j + (1 - \hat{p}^c_i)(1 - \hat{p}^c_j)$$

Variance via the delta method, exploiting independence of $\hat{p}^c_i$ and $\hat{p}^c_j$ (separate molecules):

$$\text{Var}(\hat{\theta}^{c,U,A}) = (\hat{p}^c_j)^2 \frac{\hat{p}^c_i (1 - \hat{p}^c_i)}{n^c_i} + (\hat{p}^c_i)^2 \frac{\hat{p}^c_j (1 - \hat{p}^c_j)}{n^c_j}$$

Derivation: $\theta^{U,A} = p_i p_j$, so $\nabla \theta = (p_j, p_i)$. With $\text{Var}(\hat{p}_i) = p_i(1-p_i)/n_i$ and the covariance term zero, the delta method gives the expression above.

For symmetric:

$$\text{Var}(\hat{\theta}^{c,U,S}) = (2\hat{p}^c_j - 1)^2 \frac{\hat{p}^c_i (1 - \hat{p}^c_i)}{n^c_i} + (2\hat{p}^c_i - 1)^2 \frac{\hat{p}^c_j (1 - \hat{p}^c_j)}{n^c_j}$$

Derivation: $\theta^{U,S} = p_i p_j + (1-p_i)(1-p_j) = 2 p_i p_j - p_i - p_j + 1$, so $\partial \theta / \partial p_i = 2 p_j - 1$, $\partial \theta / \partial p_j = 2 p_i - 1$.

### 5.2 Category-pooled fallback when sparse

When $n^c_i < n_{\min}^M$ OR $n^c_j < n_{\min}^M$, fall back to the marginal accessibility prior (§3.2). The category-pooled posterior on $p^c_i$ is:

$$p^c_i \mid k^{c,\text{open}}_i, n^c_i \sim \text{Beta}(\hat{\alpha}^M_g + k^{c,\text{open}}_i, \; \hat{\beta}^M_g + n^c_i - k^{c,\text{open}}_i)$$

with mean and variance defined identically to §4 (replacing $\alpha^L \to \alpha^M$ etc.). Substitute the posterior means into $\hat{\theta}^{c,U,A}$ and recompute the delta-method variance using the posterior variances of $p^c_i$, $p^c_j$ in place of the binomial $p_i(1-p_i)/n_i$ terms.

This fallback is mandatory: without it, cells with sparse unlinked observations get nonsensical $\hat{\theta}^{c,U}$ estimates (e.g. 0 or 1 with zero variance from a single read), which would catastrophically bias $\Delta$.

Implementation: `hierarchical.py:unlinked_baseline()`. Includes a `used_fallback` flag in the output per $(c, p)$ for downstream diagnostics.

### 5.3 Why active concordance is the primary outcome

The active version $\theta^{U,A} = p_i p_j$ is monotone in marginal accessibility and has a clean interpretation: "the chance both ends are open by independence alone." It is positively biased when both anchors are constitutively accessible (e.g., housekeeping promoters), but this is correctly handled by $\Delta$: a pair where $p_i, p_j$ are both ≈ 1 will have $\theta^{U,A} \approx 1$, so $\Delta = \theta^L - \theta^{U,A}$ is bounded above by 0 and the embedding feature is near zero (correctly indicating no coordination signal beyond marginal state).

The symmetric version $\theta^{U,S}$ is U-shaped in $(p_i, p_j)$, minimized at $(0.5, 0.5)$ and maximized at the corners $(0,0)$ or $(1,1)$. It captures "both ends in agreement" without requiring active state, which is biologically meaningful for repressed-domain coordination. Report both; use active as primary.

---

## 6. Coordination excess Δ

### 6.1 Definition

$$\Delta^c_{ij} := \hat{\theta}^{c,L}_{ij} - \hat{\theta}^{c,U,A}_{ij}$$

### 6.2 Variance

Treating $\hat{\theta}^{c,L}$ and $\hat{\theta}^{c,U,A}$ as independent (justifiable since they are derived from disjoint sets of reads — see §6.3):

$$\sigma^2(\Delta^c_{ij}) = \sigma^2(\hat{\theta}^{c,L}_{ij}) + \text{Var}(\hat{\theta}^{c,U,A}_{ij})$$

### 6.3 Independence assumption

Linked and unlinked reads at a pair $(i, j)$ are by construction from disjoint molecules in the cell — a read pair is either linked OR contributes to one of the unlinked categories, never both. Therefore the binomial sampling components are independent.

The only source of dependence is shared latent state: a cell whose true $p^c_i$ is high will have systematically higher linked active concordance AND higher unlinked-i open fraction. This is exactly the dependence we want to capture in $\Delta$ — it is not a nuisance to be modeled. The variance expression above treats sampling variability only and is therefore conservative for inferring whether $\Delta \neq 0$ at a given $(c, p)$.

### 6.4 Properties of $\Delta$

- **Range:** $\Delta \in [-1, 1]$ in principle; in practice concentrated near 0 with tails reflecting genuine coordination effects.
- **Sign:** $\Delta > 0$ indicates contact-conditional coordination beyond marginal state. $\Delta < 0$ is rare but possible (anti-coordination, e.g. if linked reads disproportionately catch a heterogeneous transition state).
- **Calibration:** at a pair where linked reads behave as independent observations of the same marginals (no coordination), $\mathbb{E}[\Delta] \to 0$ as $n^L \to \infty$ AND $n^U \to \infty$. Bias from prior is negligible at high coverage and bounded by $|\hat{\alpha}^L/\hat{\nu}^L - \bar{y}_g|$ at low coverage.
- **Posterior variance** dominated by the linked term at low $n^{c,L}$ (prior-dominated) and by the unlinked term when $n^c_i, n^c_j$ are sparse and the fallback widens $\hat{p}^c_i$ uncertainty.

### 6.5 Implementation

`hierarchical.py:compute_delta()`. Inputs: per-cell pair-counts table and EB priors. Outputs: per-cell `{cell}.delta.parquet` with columns `pair_id, theta_L, theta_L_var, theta_U, theta_U_var, used_fallback, delta, delta_var`.

---

## 7. Per-cell coordination vector

### 7.1 Feature definition

For each pair $p$ in the panel and each cell $c$, the E_coord embedding feature is:

$$v^c_p = \begin{cases} \Delta^c_p / \sigma(\Delta^c_p) & \text{if } n^{c,L}_p \geq 1 \text{ AND } \min(n^c_i, n^c_j) \geq n_{\min}^M \\ \text{missing} & \text{otherwise} \end{cases}$$

with an associated observation weight:

$$w^c_p = \begin{cases} 1 / \sigma^2(\Delta^c_p) & \text{if observed} \\ 0 & \text{otherwise} \end{cases}$$

### 7.2 Justification of precision weighting

Under a Gaussian approximation to the $\Delta$ posterior, the precision-weighted feature is the maximum-likelihood estimate of the underlying signal. In an embedding context, precision-weighted z-scores are the right objects to perform PCA on, because variance-stabilization removes the systematic dependence of feature magnitude on per-cell, per-pair coverage. Without precision weighting, high-coverage cells dominate the embedding from coverage variation alone, mimicking biological signal.

Formally: under the per-(c, p) generative model $\Delta^c_p \sim \mathcal{N}(\mu^c_p, \sigma^2(\Delta^c_p))$ with $\mu^c_p$ the unknown true coordination, the standardized $v^c_p = \Delta^c_p / \sigma(\Delta^c_p) \sim \mathcal{N}(\mu^c_p / \sigma(\Delta^c_p), 1)$ has unit variance under the null $\mu^c_p = 0$. Embedding on $v^c_p$ ranks cell deviations from the null by signal-to-noise — the appropriate ranking for cell-clustering on coordination.

### 7.3 Masked-loss PCA

Standard PCA does not handle missingness. We use iterative SVD with a binary mask:

**Algorithm:** Given $V \in \mathbb{R}^{C \times P}$ with mask $M \in \{0, 1\}^{C \times P}$ (here $M_{cp} = \mathbb{1}[w^c_p > 0]$):

1. Initialize $V^{(0)}$ by mean-imputing each column over observed entries.
2. For $t = 1, \ldots, T_{\max}$:
   a. Compute SVD: $V^{(t-1)} = U \Sigma W^\top$, keep top $r$ components.
   b. Reconstruct $\hat{V}^{(t)} = U_r \Sigma_r W_r^\top$.
   c. Replace masked entries: $V^{(t)} = M \odot V^{(0)} + (1 - M) \odot \hat{V}^{(t)}$.
3. Iterate until $\|\hat{V}^{(t)} - \hat{V}^{(t-1)}\|_F / \|\hat{V}^{(t-1)}\|_F < 10^{-4}$.
4. Final principal components: $U_r \Sigma_r$.

This is equivalent to maximizing the masked Frobenius loss $\| M \odot (V - U_r \Sigma_r W_r^\top) \|_F^2$ via alternating minimization. Convergence is rapid (typically <20 iterations for our matrices); implementation in `embed.py:sparse_pca_em()`.

Optional refinement: weight each observation by $w^c_p$ in the loss. This gives weighted-masked PCA, more principled but iteratively heavier. The unweighted masked version is the v1 default; weighted version available via `--pca-mode weighted_masked`.

---

## 8. Three embeddings — formal definitions

### 8.1 E_contact

Per-(cell, pair) feature:

$$u^c_p = \log(1 + n^{c,L}_{ij}) - \log(\lambda_c + 1), \quad \lambda_c = \sum_{p' \in \text{panel}} n^{c,L}_{p'}$$

The `+1` in $\lambda_c$ avoids $\log(0)$ for cells with zero total linked reads (degenerate cells, should be filtered upstream). Standardize per pair across cells; standard PCA → UMAP. This encodes contact frequency normalized for per-cell depth.

This is the **Step 1-only** embedding: it uses only the number of linked reads per pair, not their state. Cell-type information in this embedding comes entirely from the differential 3D contact structure across cell types.

### 8.2 E_coord

Per-(cell, pair) feature $v^c_p$ defined in §7.1, with weights $w^c_p$. Masked-loss PCA (§7.3) → UMAP.

This is the **Step 2-only** embedding: by construction, every contributing observation conditions on $n^{c,L}_p \geq 1$, i.e., conditions on contact being observed. Cell-type information in this embedding cannot be attributed to contact-frequency differences across cell types, because contact presence is held fixed at the per-(c, p) level.

### 8.3 E_full

Per-(cell, pair) feature is the concatenation:

$$f^c_p = \begin{bmatrix} u^c_p \\ v^c_p \\ \eta^c_i \\ \eta^c_j \end{bmatrix}$$

where $\eta^c_i = $ mean of per-read-end posteriors across all reads with at least one end in anchor $i$ in cell $c$ (a marginal accessibility summary). Standardize each component separately across cells; masked-loss PCA → UMAP.

This is the **integrated** embedding — uses all information from contact frequency, coordination given contact, and marginal accessibility at the anchors.

---

## 9. Component contribution tests

### 9.1 Cell-type classification AUC per category

For a fixed (embedding $E$, pair category $\text{cat}$, distance band $b$):

1. Restrict the feature matrix to pairs $p$ with $\text{cat}(p) = \text{cat}, b(p) = b$.
2. PCA (masked if applicable) to top $r = 20$ components.
3. 5-fold stratified CV: for each fold, fit logistic regression $\Pr(k(c) = k^* \mid \text{PC}_1, \ldots, \text{PC}_r)$ for each cluster $k^*$ vs all others. Use one-vs-rest.
4. Compute macro AUC over all $k^*$ and over 5 folds.

Report mean AUC with bootstrap 95% CI (200 bootstraps over cells). Implementation: `component_analysis.py:auc_per_category()`.

### 9.2 Likelihood-ratio test for marginal contribution of E_coord

Restrict to cells with both E_contact and E_coord PCs available (typically all cells).

- $M_0$: logistic regression $k(c) \sim \text{PC}_1^{\text{contact}}, \ldots, \text{PC}_{20}^{\text{contact}}$.
- $M_1$: logistic regression $k(c) \sim \text{PC}_1^{\text{contact}}, \ldots, \text{PC}_{20}^{\text{contact}}, \text{PC}_1^{\text{coord}}, \ldots, \text{PC}_{20}^{\text{coord}}$.

Likelihood-ratio statistic:

$$\Lambda = 2 (\ell(M_1) - \ell(M_0)) \sim \chi^2_{20 \cdot (K-1)}$$

under $H_0$: the E_coord PCs add no information beyond E_contact PCs.

Report $\Lambda$, degrees of freedom, p-value, and the companion $\Delta\text{AUC} = \text{AUC}(M_1) - \text{AUC}(M_0)$ with bootstrap CI. The LR test is the formal hypothesis test; $\Delta\text{AUC}$ is the interpretable effect size.

Implementation: `component_analysis.py:lr_test_marginal_contribution()`.

### 9.3 Matched contact frequency comparison

For each pair of cells $(c, c')$:

$$S^{\text{contact}}_{cc'} = \frac{\langle \mathbf{u}_c, \mathbf{u}_{c'} \rangle}{\|\mathbf{u}_c\|_2 \|\mathbf{u}_{c'}\|_2}, \quad S^{\text{coord}}_{cc'} = \frac{\sum_p w^c_p w^{c'}_p v^c_p v^{c'}_p}{\sqrt{\sum_p w^c_p w^{c'}_p (v^c_p)^2} \sqrt{\sum_p w^c_p w^{c'}_p (v^{c'}_p)^2}}$$

(The weighted-mask cosine in $S^{\text{coord}}$ properly handles missingness.)

Define the binary indicator $T_{cc'} = \mathbb{1}[k(c) = k(c')]$. Fit the regression:

$$S^{\text{coord}}_{cc'} = \beta_0 + \beta_1 S^{\text{contact}}_{cc'} + \beta_2 T_{cc'} + \beta_3 S^{\text{contact}}_{cc'} \cdot T_{cc'} + \epsilon$$

The biological hypothesis predicts $\beta_2 > 0$: even within matched-contact-similarity, same-cluster cells have more similar coordination than different-cluster cells. The interaction $\beta_3$ measures whether this gap widens with contact similarity.

Within-quartile reporting: also fit the regression within the top quartile of $S^{\text{contact}}_{cc'}$. This is the most stringent version of the test — cells with nearly identical contact patterns still differ in coordination by cell type.

Implementation: `component_analysis.py:matched_contact_comparison()`. With ~1500 cells, sub-sample ~100k cell pairs for the regression (full $\binom{1500}{2} \approx 10^6$ is computable but unnecessary).

---

## 10. Subtype resolution (Defense 4)

Conditional on adequate per-subtype sample size (≥100 cells per subtype):

1. Fit multinomial logistic regression on subtype label using top 20 PCs from each embedding separately.
2. Report per-subtype-pair classification accuracy (e.g., L2/3 vs L5).
3. Compute $\Delta\text{accuracy} = \text{acc}(E_{\text{coord}}) - \text{acc}(E_{\text{contact}})$ per subtype contrast.

Headline claim: there exist subtype contrasts where $\Delta\text{accuracy}$ is large and positive — subtypes where contact-only embeddings fail and coordination embeddings succeed. This identifies the regime where the unique scNOMe-HiC capability matters.

If no such contrasts exist for cortex, report it honestly. The component-contribution AUC and the matched-contact-similarity tests (§9) still stand on their own.

---

## 11. Identifiability

### 11.1 What is identified

- $\theta^{c,L}_{ij}$ is identified up to prior shrinkage from linked-read counts. At sufficient $n^{c,L}_{ij}$ the posterior concentrates on the MLE.
- $p^c_i, p^c_j$ are identified from unlinked reads (or category-pooled fallback). Each is the marginal accessibility at the corresponding anchor in the cell.
- $\theta^{c,U,A}_{ij}$ is identified once $p^c_i, p^c_j$ are, by construction.
- $\Delta^c_{ij} = \theta^{c,L} - \theta^{c,U,A}$ is identified as the difference of identified quantities.

### 11.2 What is NOT identified

- **Causal direction of coordination.** $\Delta > 0$ at a pair establishes statistical coupling between contact and concordance but does not establish that contact causes coordination (or vice versa). For the manuscript's biological claims about TF-folding-driven coordination, additional evidence (e.g., perturbation, TF motif enrichment, lineage specificity) is needed and is provided by the falsification design (NRF1 control, lineage-TF specificity).
- **Per-cell coordination at low-coverage pairs.** A pair $(i,j)$ in cell $c$ with $n^{c,L}_{ij} = 0$ has $\hat{\theta}^{c,L}_{ij}$ equal to the prior mean. The pair is dropped from the cell's coordination vector via the masking rule, so this is not an identification problem in the embedding — but it means individual cells with very sparse data contribute weakly. The cell-level embedding location is identified only insofar as the cell has enough informative pairs.
- **Distinguishing coordination from per-cell mean accessibility.** When marginal accessibility at $(i, j)$ approaches 0 or 1 in a cell, $\hat{\theta}^{c,U,A}_{ij}$ approaches 0 or 1, and $\Delta$ is bounded. This is correct behavior but means that highly committed cells (e.g., terminally differentiated cells with constitutively accessible regulatory regions) have inherently smaller $\Delta$ magnitudes than less committed cells. This may produce a per-cell-state confound that should be discussed in the manuscript and stratified in sensitivity analyses (§13).

### 11.3 Identifiability of priors

The category × band EB priors $(\hat{\alpha}^L_g, \hat{\beta}^L_g)$ are identified up to the constraint $\hat{\nu}^L_g \geq 2$. With ~10k–50k observations per group (after multiplying pairs × cells × observed-pairs), MoM estimates are well-determined. The only risk is groups with very few observations (e.g., `tf_anchored_X` with rare TF) — the implementation flags any group with fewer than 100 (cell, pair) observations and refuses to use the prior.

---

## 12. Multiple testing and uncertainty quantification

### 12.1 Per-pair posterior inference (not the primary deliverable)

If individual-pair claims are needed (e.g., "this specific E-P pair shows significant coordination in Exc"), use the Δ posterior with FDR control:

1. Per pair per cluster $k$, compute aggregate $\bar{\Delta}^k_p = \sum_{c: k(c) = k} w^c_p \Delta^c_p / \sum_{c: k(c) = k} w^c_p$ (precision-weighted cluster mean).
2. Standard error via inverse total precision.
3. Test $H_0: \bar{\Delta}^k_p = 0$ vs $H_1: \bar{\Delta}^k_p > 0$ with Wald z-test.
4. BH FDR < 0.05 over all pairs within the panel.

This is provided for completeness; the **primary deliverables (embeddings, AUC per category, LR test, matched contact) do not require per-pair inference** — they aggregate over panels and rely on the inverse-variance weighting to make low-data pairs negligible automatically.

### 12.2 Bootstrap procedures

For AUC confidence intervals: 200 cell-resampling bootstraps. Resample cells with replacement, recompute embedding PCs and CV-AUC. Report 2.5/97.5 percentile interval.

For $\Delta\text{AUC}$ in the LR test: paired bootstrap — same cell resample for $M_0$ and $M_1$, compute paired difference per bootstrap.

For matched-contact regression coefficients: parametric bootstrap from the fitted residual variance (cell-pair observations are not independent so block-bootstrap by cell is more conservative; recommend the parametric default with cell-level cluster-robust SE as a check).

---

## 13. Sensitivity analyses (required in supplementary)

| Parameter | Default | Sensitivity range | Reported as |
|-----------|---------|-------------------|-------------|
| $\tau$ (posterior threshold) | 0.5 | 0.4, 0.5, 0.6 | AUC per category at each $\tau$ |
| Anchor size | 5 kb | 2.5 kb, 5 kb, 10 kb | Same |
| $n_{\min}^M$ (unlinked min count) | 5 | 3, 5, 10 | Fraction of pairs using fallback |
| EB pooling | category × band | category-only; per-pair | Δ distribution and downstream AUC |
| Active vs symmetric concordance | active | symmetric | Both reported in main analysis |
| PCA components $r$ | 20 | 10, 20, 50 | LR test result stability |

The headline conclusions (NRF1 vs lineage AUC contrast; E_coord > E_contact at lineage pairs; matched-contact $\beta_2 > 0$) must hold across the full sensitivity grid for any of those parameters that could be reasonably contested.

---

## 14. Null distributions and negative controls

### 14.1 Within-cell linked/unlinked relabeling

For each cell, randomly relabel a fraction of linked reads as unlinked-i / unlinked-j (preserving per-cell totals), and a corresponding fraction of unlinked reads as linked (preserving pair structure). Recompute $\Delta$. Under $H_0$ that contact carries no coordination-specific information, the relabeled-$\Delta$ distribution should be indistinguishable from the original.

This is the strongest available negative control. Implement once in `pair_concordance/null.py:linked_unlinked_relabel()` and run as a mandatory smoke test before reporting any positive result.

### 14.2 Pair-label shuffle

Within each cell, permute pair IDs (preserving per-cell pair-level counts). Refit EB priors and recompute $\Delta$. AUC per category should drop to near chance — confirms category labels are not introducing structure.

### 14.3 Cell-label shuffle

Permute cluster labels across cells. Refit downstream AUC and LR tests. All test statistics should drop to null distribution. Quick falsifier — if AUC remains high after cell-label shuffle, the analysis pipeline has a leak.

---

## 15. Summary of equations (cross-reference for implementation)

| Quantity | Symbol | Equation | Module |
|----------|--------|----------|--------|
| Read-end state | $s^{\text{end}}$ | $\mathbb{1}[\eta^{\text{end}} > \tau]$ | `classify.py` |
| Linked active-concordance count | $k^{c,L,A}_{ij}$ | $\sum_{r \in L_{ij}^c} s_{r,1} s_{r,2}$ | `classify.py` |
| Marginal accessibility | $\hat{p}^c_i$ | $k^{c,\text{open}}_i / n^c_i$ | `hierarchical.py` |
| Independence baseline | $\hat{\theta}^{c,U,A}_{ij}$ | $\hat{p}^c_i \hat{p}^c_j$ | `hierarchical.py` |
| Unlinked variance | $\text{Var}(\hat{\theta}^{c,U,A})$ | §5.1 expression | `hierarchical.py` |
| EB prior MoM | $\hat{\alpha}^L_g, \hat{\beta}^L_g$ | §3.1 expressions | `empirical_bayes.py` |
| Linked posterior mean | $\hat{\theta}^{c,L}_{ij}$ | $(\hat{\alpha}^L + k^L) / (\hat{\alpha}^L + \hat{\beta}^L + n^L)$ | `hierarchical.py` |
| Linked posterior variance | $\sigma^2(\hat{\theta}^{c,L}_{ij})$ | §4 expression | `hierarchical.py` |
| Coordination excess | $\Delta^c_{ij}$ | $\hat{\theta}^{c,L} - \hat{\theta}^{c,U,A}$ | `hierarchical.py` |
| Delta variance | $\sigma^2(\Delta^c_{ij})$ | $\sigma^2(\hat{\theta}^{c,L}) + \text{Var}(\hat{\theta}^{c,U,A})$ | `hierarchical.py` |
| E_coord feature | $v^c_p$ | $\Delta^c_p / \sigma(\Delta^c_p)$ | `embed.py` |
| E_coord weight | $w^c_p$ | $1 / \sigma^2(\Delta^c_p)$ | `embed.py` |
| E_contact feature | $u^c_p$ | $\log(1 + n^{c,L}_p) - \log(\lambda_c + 1)$ | `embed.py` |
| LR statistic | $\Lambda$ | $2(\ell(M_1) - \ell(M_0))$ | `component_analysis.py` |

---

## 16. Version history

- **v1:** Initial mathematical specification covering generative model, EB priors, posteriors, $\Delta$ definition and variance, embeddings, hypothesis tests, identifiability, sensitivity, and null distributions.
- **v1.1 (current):**
  - **§0 (new):** Three-step TF-mediated regulation biological framing. Maps each step to its measurement modality and clarifies which step the v1.1 framework tests (Step 2 conditional on Step 1, via $\Delta$).
  - **§1.3:** Introduced explicit distance parameters $d_{\min}$ (linked-read minimum, default 20 kb) and $d_{\max}^U$ (unlinked-read maximum mate distance, default 20 kb). Updated definitions of $n^c_i, n^c_j$ to require short-range mates. Added explanatory note on why the short-range constraint is essential.
  - **§1.4:** Renamed $p^c_i, p^c_j$ to "local marginal accessibility" to emphasize they are not population-wide marginals.
  - **§2.3:** Restated unlinked likelihood with explicit short-range mate constraint and explanation of why it gives the right conditional.
  - **§5.1:** Added cross-reference reminder of the short-range constraint at the point of estimation.
  - **(Implicit throughout):** Ground-truth labels for cell-type classification analyses (§9.1, §9.2, etc.) are now sourced from the `cell_type` column of the sample metadata TSV. The labels were previously determined by HCG-methylation clustering at 250 kb bins (an orthogonal modality from the GCH + contact data feeding E_coord), making the validation a real test rather than tautology. No math changes are required — only the data-source documentation.

Future revisions:
- v1.2 after initial implementation: update if any equation differs in practice (e.g. numerical stability fixes).
- v2 if HCG (tri-modal Step 3) extension is implemented: add joint-modality model section.
- Final manuscript revision: trim and reformat for inclusion as Methods supplement.
