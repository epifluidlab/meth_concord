"""Rank-1 test on the 2x2 GCH/HCG cross-covariance between contact ends.

Part B of design/joint_state_rank1_design.md -- the numeric core, pure and
vectorised, so the estimator can be tested against simulation without a BAM.

THE CLAIM.  Under the model

    theta^gch_e = mu^gch + lambda_g * u_e + eta^g_e
    theta^hcg_e = mu^hcg + lambda_h * u_e + eta^h_e

coordination between contacting loci runs through a single shared open/closed
axis u, which makes ``Sigma_AB = rho_u sigma_uA sigma_uB * u_A u_B^T`` -- RANK
1.  Departure from rank 1 means mark-specific coordination the axis does not
explain, and that departure is the intended headline rather than a failure.

WHY THE DETERMINANT IS THE PRIMARY STATISTIC.  For a 2x2, rank <= 1 iff
det = 0.  The whitening matrices are invertible, so rank(R) = rank(Sigma_AB)
exactly: the rank-1 hypothesis does not depend on whitening or on any
measurement-error correction, and Sigma_AB is unbiased as it stands because
sampling noise is independent across the two reads.  rho2 stays as the
interpretable effect size but carries the fragile disattenuated denominators;
when the two disagree, D is the one to trust and the disagreement is itself a
conditioning diagnostic.

    D = 2 det(Sigma_AB) / ||Sigma_AB||_F^2      in [-1, 1], zero iff rank 1

THE TRAP THIS MODULE EXISTS TO AVOID.  Both |D| and rho2 have STRICTLY
POSITIVE expectation under the null at finite n.  Reporting the raw value
makes every bin in the genome read as rank-2, which the design names as the
most likely route to a false headline.  :func:`conditional_rank1_null`
returns the null distribution so the caller reports the EXCESS; nothing here
returns a p-value from an asymptotic approximation, because disattenuated
matrices with heteroscedastic error and non-normal marginals do not satisfy
Bartlett or Lawley.

lambda_h is FREE-SIGNED -- negative in the canonical anticorrelated case,
positive in gene bodies and PMDs -- so there is no single "sign pattern is
correct" boolean here.  Three of the four valid patterns would score as
failures against any such flag.  :func:`sign_pattern` emits the two signs
separately and a class label.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ACC_FIELDS", "new_acc", "accumulate", "merge", "moments",
    "det_statistic", "whiten", "canonical", "sign_pattern",
    "conditional_rank1_null", "PATTERN_CLASSES",
]

#: The 20 additive floats of Section B.2.  Q_AB is NOT symmetric: the anchor
#: and partner ends may weight the marks differently (a promoter anchor
#: contacting distal partners), which is asymmetry rather than rank-2-ness
#: and gets its own diagnostic.
ACC_FIELDS = (
    "W", "W2",
    "S_A_g", "S_A_h", "S_B_g", "S_B_h",
    "Q_AA_gg", "Q_AA_gh", "Q_AA_hh",
    "Q_BB_gg", "Q_BB_gh", "Q_BB_hh",
    "Q_AB_gg", "Q_AB_gh", "Q_AB_hg", "Q_AB_hh",
    "V_A_g", "V_A_h", "V_B_g", "V_B_h",
)

#: (sign lambda_h, sign rho_u) -> the Section 1.1 reading.
PATTERN_CLASSES = {
    (-1, +1): "shared_state",       # ends share open/closed state
    (-1, -1): "opposite_state",     # ends in opposite states
    (+1, +1): "coupled_same",       # accessible-and-methylated coordinated
    (+1, -1): "coupled_opposite",
}


def new_acc() -> np.ndarray:
    """A zeroed accumulator.  All fields are additive, so shards merge by sum."""
    return np.zeros(len(ACC_FIELDS), dtype=np.float64)


def accumulate(acc, theta_A, theta_B, v_A, v_B, weights=None) -> np.ndarray:
    """Add contacts to an accumulator, in place.

    ``theta_*`` are (n, 2) centred (gch, hcg) vectors; ``v_*`` their (n, 2)
    sampling variances.  ``weights`` defaults to the Section B.2 inverse
    total-variance weight; pass explicit weights when they have already been
    capped at the within-stratum 99th percentile.

    The weight deliberately does NOT involve |theta - 0.5|: a posterior near
    0.5 may be uninformative OR a genuine intermediate state, and down-
    weighting it conflates the two.
    """
    theta_A = np.atleast_2d(np.asarray(theta_A, float))
    theta_B = np.atleast_2d(np.asarray(theta_B, float))
    v_A = np.atleast_2d(np.asarray(v_A, float))
    v_B = np.atleast_2d(np.asarray(v_B, float))
    if weights is None:
        weights = 1.0 / (v_A.sum(1) + v_B.sum(1) + 1e-12)
    w = np.asarray(weights, float).ravel()

    ag, ah = theta_A[:, 0], theta_A[:, 1]
    bg, bh = theta_B[:, 0], theta_B[:, 1]
    add = np.array([
        w.sum(), (w ** 2).sum(),
        (w * ag).sum(), (w * ah).sum(), (w * bg).sum(), (w * bh).sum(),
        (w * ag * ag).sum(), (w * ag * ah).sum(), (w * ah * ah).sum(),
        (w * bg * bg).sum(), (w * bg * bh).sum(), (w * bh * bh).sum(),
        (w * ag * bg).sum(), (w * ag * bh).sum(),
        (w * ah * bg).sum(), (w * ah * bh).sum(),
        (w * v_A[:, 0]).sum(), (w * v_A[:, 1]).sum(),
        (w * v_B[:, 0]).sum(), (w * v_B[:, 1]).sum(),
    ])
    acc += add
    return acc


def merge(*accs) -> np.ndarray:
    """Sum accumulators.  Sharding must not change the answer."""
    out = new_acc()
    for a in accs:
        out += np.asarray(a, float)
    return out


def moments(acc) -> dict:
    """Second moments, mean sampling variances and N_eff from an accumulator."""
    a = dict(zip(ACC_FIELDS, np.asarray(acc, float)))
    W = a["W"]
    if W <= 0:
        raise ValueError("empty accumulator: no weight was added")
    mu_A = np.array([a["S_A_g"], a["S_A_h"]]) / W
    mu_B = np.array([a["S_B_g"], a["S_B_h"]]) / W
    S_AA = np.array([[a["Q_AA_gg"], a["Q_AA_gh"]],
                     [a["Q_AA_gh"], a["Q_AA_hh"]]]) / W - np.outer(mu_A, mu_A)
    S_BB = np.array([[a["Q_BB_gg"], a["Q_BB_gh"]],
                     [a["Q_BB_gh"], a["Q_BB_hh"]]]) / W - np.outer(mu_B, mu_B)
    S_AB = np.array([[a["Q_AB_gg"], a["Q_AB_gh"]],
                     [a["Q_AB_hg"], a["Q_AB_hh"]]]) / W - np.outer(mu_A, mu_B)
    V_A = np.array([a["V_A_g"], a["V_A_h"]]) / W
    V_B = np.array([a["V_B_g"], a["V_B_h"]]) / W
    return {"mu_A": mu_A, "mu_B": mu_B, "S_AA": S_AA, "S_BB": S_BB,
            "S_AB": S_AB, "V_A": V_A, "V_B": V_B, "W": W,
            "N_eff": (W ** 2 / a["W2"]) if a["W2"] > 0 else 0.0}


#: Which marks each moment field needs present.  ``gA`` is GCH on the anchor
#: end, ``hB`` is HCG on the partner end.  Used only by the pairwise mode.
FIELD_REQUIRES = {
    "W": (), "W2": (),
    "S_A_g": ("gA",), "S_A_h": ("hA",), "S_B_g": ("gB",), "S_B_h": ("hB",),
    "Q_AA_gg": ("gA",), "Q_AA_gh": ("gA", "hA"), "Q_AA_hh": ("hA",),
    "Q_BB_gg": ("gB",), "Q_BB_gh": ("gB", "hB"), "Q_BB_hh": ("hB",),
    "Q_AB_gg": ("gA", "gB"), "Q_AB_gh": ("gA", "hB"),
    "Q_AB_hg": ("hA", "gB"), "Q_AB_hh": ("hA", "hB"),
    "V_A_g": ("gA",), "V_A_h": ("hA",), "V_B_g": ("gB",), "V_B_h": ("hB",),
}


def new_acc_pairwise() -> np.ndarray:
    """A pairwise-complete accumulator: 20 weighted sums plus 20 weight sums.

    Section B.4's ``moment_mode='pairwise'``.  Each moment entry is averaged
    over the contacts that carry the marks THAT ENTRY needs, so an end with
    GCH but no HCG still contributes to the gg entries -- which with
    ``--min-hcg-sites 1`` is substantially more data than complete-case.

    The cost is that the entries no longer share a sample.  Sigma_AA can come
    back non-PSD, the eigenvalue clipping in :func:`whiten` then fires and is
    flagged, and the two modes can disagree.  That is why the design asks for
    both to be reported rather than for one to be chosen.
    """
    return np.zeros(2 * len(ACC_FIELDS), dtype=np.float64)


def accumulate_pairwise(acc, theta_A, theta_B, v_A, v_B, present_A, present_B,
                        weights=None) -> np.ndarray:
    """Add contacts to a pairwise accumulator, in place.

    ``present_*`` are (n, 2) booleans: is (gch, hcg) observed on that end.
    A value whose mark is absent must never reach a sum, so it is zeroed
    rather than trusted -- an unobserved posterior equals the prior mean, and
    the prior mean is not a measurement.
    """
    theta_A = np.atleast_2d(np.asarray(theta_A, float))
    theta_B = np.atleast_2d(np.asarray(theta_B, float))
    v_A = np.atleast_2d(np.asarray(v_A, float))
    v_B = np.atleast_2d(np.asarray(v_B, float))
    pA = np.atleast_2d(np.asarray(present_A, bool))
    pB = np.atleast_2d(np.asarray(present_B, bool))
    if weights is None:
        weights = 1.0 / (np.where(pA, v_A, 0).sum(1)
                         + np.where(pB, v_B, 0).sum(1) + 1e-12)
    w = np.asarray(weights, float).ravel()

    have = {"gA": pA[:, 0], "hA": pA[:, 1], "gB": pB[:, 0], "hB": pB[:, 1]}
    val = {
        "W": np.ones_like(w), "W2": w,
        "S_A_g": theta_A[:, 0], "S_A_h": theta_A[:, 1],
        "S_B_g": theta_B[:, 0], "S_B_h": theta_B[:, 1],
        "Q_AA_gg": theta_A[:, 0] ** 2,
        "Q_AA_gh": theta_A[:, 0] * theta_A[:, 1],
        "Q_AA_hh": theta_A[:, 1] ** 2,
        "Q_BB_gg": theta_B[:, 0] ** 2,
        "Q_BB_gh": theta_B[:, 0] * theta_B[:, 1],
        "Q_BB_hh": theta_B[:, 1] ** 2,
        "Q_AB_gg": theta_A[:, 0] * theta_B[:, 0],
        "Q_AB_gh": theta_A[:, 0] * theta_B[:, 1],
        "Q_AB_hg": theta_A[:, 1] * theta_B[:, 0],
        "Q_AB_hh": theta_A[:, 1] * theta_B[:, 1],
        "V_A_g": v_A[:, 0], "V_A_h": v_A[:, 1],
        "V_B_g": v_B[:, 0], "V_B_h": v_B[:, 1],
    }
    n_f = len(ACC_FIELDS)
    for i, f in enumerate(ACC_FIELDS):
        need = FIELD_REQUIRES[f]
        ok = np.ones(len(w), bool)
        for m in need:
            ok &= have[m]
        wf = np.where(ok, w, 0.0)
        acc[i] += float((wf * val[f]).sum())
        acc[n_f + i] += float(wf.sum())
    return acc


def moments_pairwise(acc) -> dict:
    """Second moments from a pairwise accumulator.

    Each entry is divided by ITS OWN weight, and each mean is taken over the
    subset that entry saw.  Means from different subsets is what makes this
    approximate; it is the definition of pairwise-complete, not a shortcut.
    """
    a = np.asarray(acc, float)
    n_f = len(ACC_FIELDS)
    s = dict(zip(ACC_FIELDS, a[:n_f]))
    wt = dict(zip(ACC_FIELDS, a[n_f:]))
    if wt["W"] <= 0:
        raise ValueError("empty accumulator: no weight was added")

    def avg(f):
        return s[f] / wt[f] if wt[f] > 0 else np.nan

    mu_A = np.array([avg("S_A_g"), avg("S_A_h")])
    mu_B = np.array([avg("S_B_g"), avg("S_B_h")])
    S_AA = np.array([
        [avg("Q_AA_gg") - mu_A[0] ** 2, avg("Q_AA_gh") - mu_A[0] * mu_A[1]],
        [avg("Q_AA_gh") - mu_A[0] * mu_A[1], avg("Q_AA_hh") - mu_A[1] ** 2]])
    S_BB = np.array([
        [avg("Q_BB_gg") - mu_B[0] ** 2, avg("Q_BB_gh") - mu_B[0] * mu_B[1]],
        [avg("Q_BB_gh") - mu_B[0] * mu_B[1], avg("Q_BB_hh") - mu_B[1] ** 2]])
    S_AB = np.array([
        [avg("Q_AB_gg") - mu_A[0] * mu_B[0], avg("Q_AB_gh") - mu_A[0] * mu_B[1]],
        [avg("Q_AB_hg") - mu_A[1] * mu_B[0], avg("Q_AB_hh") - mu_A[1] * mu_B[1]]])
    V_A = np.array([avg("V_A_g"), avg("V_A_h")])
    V_B = np.array([avg("V_B_g"), avg("V_B_h")])
    W = wt["W"]
    W2 = s["W2"]
    return {"mu_A": mu_A, "mu_B": mu_B, "S_AA": S_AA, "S_BB": S_BB,
            "S_AB": S_AB, "V_A": V_A, "V_B": V_B, "W": W,
            "N_eff": (W ** 2 / W2) if W2 > 0 else 0.0,
            "entry_weights": wt}


def det_statistic(S_AB) -> float:
    """``D = 2 det / ||.||_F^2`` -- zero iff rank 1, and needs no correction.

    Scale-free and invariant to invertible whitening on either side, which is
    what makes it the primary test: it is a statement about Sigma_AB's rank,
    not about any disattenuated denominator.
    """
    M = np.asarray(S_AB, float)
    fro2 = float((M ** 2).sum())
    if fro2 <= 0:
        return 0.0
    return float(2.0 * np.linalg.det(M) / fro2)


def _sqrt_psd(M, floor=1e-6):
    """Symmetric square root -- the inverse of :func:`_inv_sqrt_psd`.

    Needed to map a shuffled null back into theta space.  D is NOT invariant
    under whitening (only its zero is), so a null generated in the whitened
    basis and compared against a D computed on Sigma_AB would be comparing
    two different statistics -- which shows up as wildly conservative
    p-values rather than as an error.
    """
    M = 0.5 * (np.asarray(M, float) + np.asarray(M, float).T)
    vals, vecs = np.linalg.eigh(M)
    vals = np.maximum(vals, floor)
    return (vecs * (vals ** 0.5)) @ vecs.T


def _inv_sqrt_psd(M, floor=1e-6):
    """Symmetric inverse square root, with the clipping flagged not hidden.

    Eigendecomposition rather than Cholesky so the result is orientation
    independent.  With ``--min-hcg-sites 1`` the disattenuated hcg variance is
    a small difference of large numbers and clipping fires often; the caller
    needs to know it happened, so this returns the flag.
    """
    M = 0.5 * (np.asarray(M, float) + np.asarray(M, float).T)
    vals, vecs = np.linalg.eigh(M)
    clipped = bool((vals < floor).any())
    vals = np.maximum(vals, floor)
    return (vecs * (vals ** -0.5)) @ vecs.T, clipped


def whiten(S_AA, S_BB, S_AB, V_A, V_B, floor=1e-6) -> dict:
    """Disattenuate the within-end covariances, then whiten the cross term.

    ``Sigma_AB = S_AB`` uncorrected -- sampling noise is independent across
    the two reads, so the cross-covariance carries no measurement-error bias.
    Only the diagonals need ``- V``, and only the rho scale depends on them.
    """
    S_AA, S_BB = np.asarray(S_AA, float), np.asarray(S_BB, float)
    Sig_AA = S_AA - np.diag(np.asarray(V_A, float))
    Sig_BB = S_BB - np.diag(np.asarray(V_B, float))
    Wa, ca = _inv_sqrt_psd(Sig_AA, floor)
    Wb, cb = _inv_sqrt_psd(Sig_BB, floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_A = np.divide(np.diag(Sig_AA), np.diag(S_AA),
                          out=np.full(2, np.nan), where=np.diag(S_AA) != 0)
        rel_B = np.divide(np.diag(Sig_BB), np.diag(S_BB),
                          out=np.full(2, np.nan), where=np.diag(S_BB) != 0)
    return {"R": Wa @ np.asarray(S_AB, float) @ Wb,
            "Sigma_AA": Sig_AA, "Sigma_BB": Sig_BB,
            "W_A": Wa, "W_B": Wb,
            "C_A": _sqrt_psd(Sig_AA, floor), "C_B": _sqrt_psd(Sig_BB, floor),
            "flag_pd_clipped": bool(ca or cb),
            "reliability_gch": float(rel_A[0]), "reliability_hcg": float(rel_A[1]),
            "reliability_gch_B": float(rel_B[0]),
            "reliability_hcg_B": float(rel_B[1])}


def canonical(R) -> dict:
    """Canonical correlations of a 2x2 whitened cross-covariance.

    For a 2x2 the singular values ARE the canonical correlations, and rho2 is
    exactly the residual Frobenius norm after the best rank-1 approximation.
    """
    R = np.asarray(R, float)
    U, s, Vt = np.linalg.svd(R)
    rho1, rho2 = float(s[0]), float(s[1])
    asym = float(np.linalg.norm(R - R.T) / (np.linalg.norm(R) + 1e-300))
    u1, v1 = U[:, 0], Vt[0]
    cosang = float(np.clip(abs(u1 @ v1), 0.0, 1.0))
    return {"rho1": rho1, "rho2": rho2,
            "rho2_over_rho1": float(rho2 / rho1) if rho1 > 0 else np.nan,
            "U": U, "V": Vt.T, "u1": u1, "v1": v1,
            "asymmetry_frob": asym,
            "asymmetry_angle": float(np.degrees(np.arccos(cosang)))}


def sign_pattern(S_AA, S_AB) -> dict:
    """The two independent signs of Section 1.1, reported separately.

    ``sign_lambda_h`` is the WITHIN-end coupling: with lambda_g > 0 by
    construction (GCH methylation is accessibility), the within-end
    covariance ``lambda_g lambda_h sigma_u^2`` carries lambda_h's sign.
    ``sign_rho_u`` is the BETWEEN-end coordination: ``Sigma_AB[g,g] = rho_u
    sigma sigma lambda_gA lambda_gB`` and both lambda_g are positive.

    No combined "pattern is correct" boolean: all four combinations are
    valid readings, and a flag would score three of them as failures.
    """
    s_lh = int(np.sign(np.asarray(S_AA, float)[0, 1]))
    s_ru = int(np.sign(np.asarray(S_AB, float)[0, 0]))
    return {"sign_lambda_h": s_lh, "sign_rho_u": s_ru,
            "rank1_pattern_class": PATTERN_CLASSES.get((s_lh, s_ru),
                                                       "undetermined")}


def conditional_rank1_null(theta_A, theta_B, v_A, v_B, weights=None,
                           n_boot=500, seed=0, strata=None, floor=1e-6):
    """Section B.6.2 -- H0: rank 1, GIVEN the observed rho1.

    Whiten, rotate onto the canonical basis, and shuffle the SECOND partner
    variate only, keeping (alpha1, beta1, alpha2) paired.  That destroys any
    second-axis association while preserving rho1, which is what makes this
    the null for the rank question; the mate shuffle of B.6.1 destroys rho1
    too and answers a different one.

    Returns the observed D and rho2 with their null distributions.  REPORT
    THE EXCESS: both have strictly positive null expectation at finite n, and
    the raw values make every bin read as rank-2.

    Caveat the design asks to be documented: whitening makes the canonical
    variates uncorrelated, not independent, so this is a conditional
    randomisation test under an uncorrelatedness null.
    """
    rng = np.random.default_rng(seed)
    theta_A = np.atleast_2d(np.asarray(theta_A, float))
    theta_B = np.atleast_2d(np.asarray(theta_B, float))
    acc = accumulate(new_acc(), theta_A, theta_B, v_A, v_B, weights)
    m = moments(acc)
    wh = whiten(m["S_AA"], m["S_BB"], m["S_AB"], m["V_A"], m["V_B"], floor)
    cn = canonical(wh["R"])
    D_obs = det_statistic(m["S_AB"])

    xi_A = (theta_A - m["mu_A"]) @ wh["W_A"]
    xi_B = (theta_B - m["mu_B"]) @ wh["W_B"]
    alpha = xi_A @ cn["U"]
    beta = xi_B @ cn["V"]
    n = alpha.shape[0]
    if strata is None:
        groups = [np.arange(n)]
    else:
        s = np.asarray(strata)
        groups = [np.flatnonzero(s == u) for u in np.unique(s)]

    w = (None if weights is None else np.asarray(weights, float).ravel())
    D_null = np.empty(int(n_boot))
    r2_null = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        b2 = beta[:, 1].copy()
        for g in groups:                      # shuffle WITHIN strata only
            if g.size > 1:
                b2[g] = b2[rng.permutation(g)]
        bperm = np.column_stack([beta[:, 0], b2])
        # Back to THETA space before measuring.  D is computed on Sigma_AB,
        # and it is not whitening-invariant, so the null has to live in the
        # same basis as the observed statistic or the two are incomparable.
        tA = (alpha @ cn["U"].T) @ wh["C_A"] + m["mu_A"]
        tB = (bperm @ cn["V"].T) @ wh["C_B"] + m["mu_B"]
        a2 = accumulate(new_acc(), tA, tB, v_A, v_B, w)
        m2 = moments(a2)
        D_null[b] = det_statistic(m2["S_AB"])
        w2 = whiten(m2["S_AA"], m2["S_BB"], m2["S_AB"], m2["V_A"], m2["V_B"],
                    floor)
        r2_null[b] = canonical(w2["R"])["rho2"]

    # +1 on both sides: a permutation p-value of exactly 0 is not attainable
    # from a finite resample and claiming it would overstate the evidence.
    p_D = float((1 + (np.abs(D_null) >= abs(D_obs)).sum()) / (1 + n_boot))
    p_r2 = float((1 + (r2_null >= cn["rho2"]).sum()) / (1 + n_boot))
    return {
        "D": D_obs, "D_null_mean": float(np.mean(np.abs(D_null))),
        "D_excess": float(abs(D_obs) - np.mean(np.abs(D_null))),
        "p_rank1": p_D,
        "rho1": cn["rho1"], "rho2": cn["rho2"],
        "rho2_null_mean": float(np.mean(r2_null)),
        "rho2_excess": float(cn["rho2"] - np.mean(r2_null)),
        "p_rho2": p_r2,
        "D_null": D_null, "rho2_null": r2_null,
        "flag_pd_clipped": wh["flag_pd_clipped"], "N_eff": m["N_eff"],
    }


def mate_shuffle_null(theta_A, theta_B, v_A, v_B, weights=None,
                      n_boot=200, seed=0, strata=None, floor=1e-6):
    """Section B.6.1 -- H0: no coordination at all.

    Reassign mates within stratum.  This is the WRONG null for the rank-1
    question, because it destroys rho1 along with everything else; it exists
    to answer "is this bin coordinated at all", and as the B.7.5 sanity check
    that rho1 under it sits at zero.

    Returns ``z_mate_shuffle``, the standardised distance of the observed rho1
    from that null.
    """
    rng = np.random.default_rng(seed)
    theta_A = np.atleast_2d(np.asarray(theta_A, float))
    theta_B = np.atleast_2d(np.asarray(theta_B, float))
    v_A = np.atleast_2d(np.asarray(v_A, float))
    v_B = np.atleast_2d(np.asarray(v_B, float))
    w = (None if weights is None else np.asarray(weights, float).ravel())

    m = moments(accumulate(new_acc(), theta_A, theta_B, v_A, v_B, w))
    rho1_obs = canonical(whiten(m["S_AA"], m["S_BB"], m["S_AB"],
                                m["V_A"], m["V_B"], floor)["R"])["rho1"]

    n = theta_A.shape[0]
    groups = ([np.arange(n)] if strata is None
              else [np.flatnonzero(np.asarray(strata) == u)
                    for u in np.unique(np.asarray(strata))])

    null = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        idx = np.arange(n)
        for g in groups:
            if g.size > 1:
                idx[g] = rng.permutation(g)
        m2 = moments(accumulate(new_acc(), theta_A, theta_B[idx],
                                v_A, v_B[idx], w))
        null[b] = canonical(whiten(m2["S_AA"], m2["S_BB"], m2["S_AB"],
                                   m2["V_A"], m2["V_B"], floor)["R"])["rho1"]
    sd = float(np.std(null, ddof=1)) if n_boot > 1 else 0.0
    return {
        "rho1": rho1_obs,
        "rho1_mate_shuffle_mean": float(np.mean(null)),
        "z_mate_shuffle": float((rho1_obs - np.mean(null)) / sd) if sd > 0
                          else np.nan,
        "rho1_mate_shuffle_null": null,
    }


def cell_block_bootstrap(per_cell_accs, n_boot=500, seed=0, floor=1e-6,
                         ci=(2.5, 97.5)):
    """Section B.6.3 -- resample CELLS with replacement, not contacts.

    Reads from one cell share its coverage, its conversion efficiency and its
    cell type, so they are not independent replicates; a contact-level
    bootstrap would give intervals that are far too narrow.  Resampling whole
    per-cell accumulators and re-summing is exact, because every field is
    additive -- no contact is revisited.

    Parameters
    ----------
    per_cell_accs : sequence of accumulator arrays
        One per cell contributing to this (group, bin, stratum).

    Returns
    -------
    dict with percentile intervals for rho1, rho2 and D, and the count of
    resamples that could not be evaluated (a draw may hold too few cells to
    give a positive-weight accumulator).
    """
    accs = [np.asarray(a, float) for a in per_cell_accs]
    n_cells = len(accs)
    out = {"n_cells": n_cells}
    if n_cells < 2:
        # One cell is not a sample of cells.  A bootstrap over it would return
        # a zero-width interval, which reads as certainty rather than as the
        # absence of replication.
        for k in ("rho1", "rho2", "D"):
            out[f"{k}_ci_lo"] = np.nan
            out[f"{k}_ci_hi"] = np.nan
        out["n_boot_failed"] = int(n_boot)
        return out

    rng = np.random.default_rng(seed)
    stack = np.vstack(accs)
    draws = {"rho1": [], "rho2": [], "D": []}
    failed = 0
    for _ in range(int(n_boot)):
        pick = rng.integers(0, n_cells, n_cells)
        acc = stack[pick].sum(axis=0)
        try:
            m = moments(acc)
            cn = canonical(whiten(m["S_AA"], m["S_BB"], m["S_AB"],
                                  m["V_A"], m["V_B"], floor)["R"])
        except (ValueError, np.linalg.LinAlgError):
            failed += 1
            continue
        draws["rho1"].append(cn["rho1"])
        draws["rho2"].append(cn["rho2"])
        draws["D"].append(det_statistic(m["S_AB"]))
    for k, v in draws.items():
        if len(v) >= 2:
            lo, hi = np.percentile(v, ci)
            out[f"{k}_ci_lo"], out[f"{k}_ci_hi"] = float(lo), float(hi)
        else:
            out[f"{k}_ci_lo"] = out[f"{k}_ci_hi"] = np.nan
    out["n_boot_failed"] = failed
    return out


#: Section B.8 bin classes.  II is the headline: coordination the shared
#: open/closed axis does not explain.
BIN_CLASSES = {
    "I": "shared_axis_coordination",
    "II": "mark_specific_beyond_axis",
    "III": "uncoordinated",
    "IV": "anti_assortative",
}


def bin_class(rho1_significant: bool, rank1_rejected: bool,
              sign_rho_u: int) -> str:
    """Section B.8 class from the two tests and the coordination sign.

    IV is checked before II because an anti-assortative bin is a real reading
    of the model (ends in opposite states), not a failure -- and it is still
    rank 1, so calling it II would put an ordinary finding in the headline
    class.
    """
    if not rho1_significant:
        return "III"
    if sign_rho_u < 0:
        return "IV"
    return "II" if rank1_rejected else "I"


def summarize(acc, per_cell_accs=None, theta_A=None, theta_B=None,
              v_A=None, v_B=None, weights=None, strata=None,
              n_boot_null=500, n_boot_mate=200, n_boot_ci=500, seed=0,
              floor=1e-6, alpha_rho1=0.05, alpha_rank1=0.05,
              error_convention="raw", moment_mode="complete",
              gch_norm="location", min_neff=30.0):
    """One Section B.8 row from an accumulator and the contacts behind it.

    ``acc`` carries the moments; the raw contact arrays are needed only for
    the two resampling nulls, and ``per_cell_accs`` only for the cell-block
    interval.  Anything not supplied comes back as NaN rather than being
    silently replaced by a contact-level substitute.
    """
    m = moments(acc)
    wh = whiten(m["S_AA"], m["S_BB"], m["S_AB"], m["V_A"], m["V_B"], floor)
    cn = canonical(wh["R"])
    row = {
        "n_contacts": float(m["W"]), "N_eff": m["N_eff"],
        "D": det_statistic(m["S_AB"]),
        "rho1": cn["rho1"], "rho2": cn["rho2"],
        "rho2_over_rho1": cn["rho2_over_rho1"],
        "asymmetry_frob": cn["asymmetry_frob"],
        "asymmetry_angle": cn["asymmetry_angle"],
        "reliability_gch": wh["reliability_gch"],
        "reliability_hcg": wh["reliability_hcg"],
        "flag_pd_clipped": wh["flag_pd_clipped"],
        "flag_low_neff": bool(m["N_eff"] < min_neff),
        "error_convention": error_convention,
        "moment_mode": moment_mode,
        "gch_norm": gch_norm,
    }
    row.update(sign_pattern(m["S_AA"], m["S_AB"]))

    # w_A = Sigma_AA^(-1/2) u1 and w_B likewise: the per-end directions the
    # shared axis actually loads on, in the original (gch, hcg) basis.
    w_A = wh["W_A"] @ cn["u1"]
    w_B = wh["W_B"] @ cn["v1"]
    row.update({
        "w_A_gch": float(w_A[0]), "w_A_hcg": float(w_A[1]),
        "w_B_gch": float(w_B[0]), "w_B_hcg": float(w_B[1]),
        "phi_A": float(np.degrees(np.arctan2(w_A[1], w_A[0]))),
        "phi_B": float(np.degrees(np.arctan2(w_B[1], w_B[0]))),
        "sigma_AB_gg": float(m["S_AB"][0, 0]),
        "sigma_AB_gh": float(m["S_AB"][0, 1]),
        "sigma_AB_hg": float(m["S_AB"][1, 0]),
        "sigma_AB_hh": float(m["S_AB"][1, 1]),
    })

    have_contacts = theta_A is not None and theta_B is not None
    if have_contacts:
        nul = conditional_rank1_null(theta_A, theta_B, v_A, v_B, weights,
                                     n_boot_null, seed, strata, floor)
        for k in ("D_null_mean", "D_excess", "p_rank1", "rho2_null_mean",
                  "rho2_excess", "p_rho2"):
            row[k] = nul[k]
        ms = mate_shuffle_null(theta_A, theta_B, v_A, v_B, weights,
                               n_boot_mate, seed + 1, strata, floor)
        row["z_mate_shuffle"] = ms["z_mate_shuffle"]
        row["rho1_mate_shuffle_mean"] = ms["rho1_mate_shuffle_mean"]
    else:
        for k in ("D_null_mean", "D_excess", "p_rank1", "rho2_null_mean",
                  "rho2_excess", "p_rho2", "z_mate_shuffle",
                  "rho1_mate_shuffle_mean"):
            row[k] = np.nan

    if per_cell_accs is not None:
        boot = cell_block_bootstrap(per_cell_accs, n_boot_ci, seed + 2, floor)
        row["n_cells"] = boot["n_cells"]
        for k in ("rho1", "rho2", "D"):
            row[f"{k}_ci_lo"] = boot[f"{k}_ci_lo"]
            row[f"{k}_ci_hi"] = boot[f"{k}_ci_hi"]
    else:
        row["n_cells"] = np.nan
        for k in ("rho1", "rho2", "D"):
            row[f"{k}_ci_lo"] = row[f"{k}_ci_hi"] = np.nan

    # rho1 "significant" uses the mate-shuffle z, which is the test for
    # coordination at all; the rank test is a separate p and a separate axis.
    z = row["z_mate_shuffle"]
    rho1_sig = bool(np.isfinite(z) and abs(z) >= 1.959963984540054)
    rank1_rejected = bool(np.isfinite(row["p_rank1"])
                          and row["p_rank1"] < alpha_rank1)
    row["rho1_significant"] = rho1_sig
    row["rank1_rejected"] = rank1_rejected
    row["bin_class"] = bin_class(rho1_sig, rank1_rejected, row["sign_rho_u"])
    row["bin_class_label"] = BIN_CLASSES[row["bin_class"]]
    return row
