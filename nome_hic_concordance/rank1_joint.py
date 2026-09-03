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
