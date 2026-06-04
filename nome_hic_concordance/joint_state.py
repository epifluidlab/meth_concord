"""Joint GCH+HCG multi-state per-end epigenetic state and pairwise co-state.

Implements §2–§3 of ``design/JOINT_MULTISTATE_CONCORDANCE_DESIGN.md``: combine
a per-end **accessibility** posterior ``a = P(accessible | GpC counts)`` with a
per-end **methylation** posterior ``m = P(methylated | CpG counts)`` into a soft
distribution over a 2×2 joint-state grid (the *independence approximation*,
v1 — the product of marginals), and form the pairwise soft **co-state** for two
distal ends of a 3-D contact.

Joint-state grid (``a`` = accessible prob, ``m`` = methylated prob)::

    open               = a · (1 − m)        active promoters / enhancers
    closed             = (1 − a) · m        repressed heterochromatin
    poised             = (1 − a) · (1 − m)  occupied-unmethylated / poised
    active_methylated  = a · m              accessible-methylated (rare)

The four states sum to 1 elementwise by construction.

This module is **pure** (no IO, fully vectorized): the joining of the GCH and
HCG context runs per cell, and the distance-matched O/E estimation, live in the
calling script (``scripts/joint_coordination_existence.py``).  Keeping the math
here makes the independence approximation (§2) and the lenient missing-modality
marginalization (§2.1) — exactly the kind of subtle steps that need it — unit
testable.
"""

from __future__ import annotations

import numpy as np

#: Canonical channel order for the 4 joint co-states.  ``pair_co_state`` returns
#: these four followed by ``mixed`` (the disagreement remainder).
JOINT_STATES = ("open", "closed", "poised", "active_methylated")


def per_end_joint_state(a, m):
    """Soft 4-state distribution per end from accessibility ``a`` and methylation ``m``.

    Parameters
    ----------
    a, m : array-like, in [0, 1]
        ``a = P(accessible | GpC counts)`` and ``m = P(methylated | CpG counts)``
        per end.  Broadcast against each other (so a scalar prior works against
        a vector of observed values).

    Returns
    -------
    (q_open, q_closed, q_poised, q_active_methylated) : tuple of ndarray
        Each the broadcast shape; the four sum to 1 elementwise.  Order matches
        :data:`JOINT_STATES`.

    Notes
    -----
    Independence approximation (design §2, v1): GCH ⟂ HCG *given the locus*, so
    the joint is the product of the two marginals.  The principled v2 upgrade
    learns a 2-D mixture over (accessibility, methylation) instead.
    """
    a = np.asarray(a, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    q_open = a * (1.0 - m)
    q_closed = (1.0 - a) * m
    q_poised = (1.0 - a) * (1.0 - m)
    q_active_methylated = a * m
    return q_open, q_closed, q_poised, q_active_methylated


def impute_missing_modality(a, n_gpc, m, n_cpg, min_gpc, min_cpg,
                            a_prior, m_prior):
    """Lenient missing-modality marginalization (design §2.1).

    An end may carry GpC sites, CpG sites, both, or neither.  Where a modality
    is **uninformative** (fewer than ``min_*`` sites, or a NaN posterior),
    substitute its prior.  This is Bayesian marginalization of the absent
    modality: ``predict_proba(k=0, n=0)`` returns the model's prior (the
    genome-wide marginal rate), so the partially-observed end's joint state is
    correctly *less committed* (diffuse along the missing axis), not wrongly
    committed.

    Use a **global** prior (one number per context, shared across cells) by
    default — locked decision #3 — so prior-imputed ends inject no cell-type
    marginal structure.  ``a_prior`` / ``m_prior`` may also be arrays (a
    locally-informed per-region prior, the optional upgrade) since they are
    broadcast.

    Parameters
    ----------
    a, m : array-like
        Per-end accessibility / methylation posteriors (NaN where the context
        had no sites).
    n_gpc, n_cpg : array-like
        Per-end GpC / CpG site counts (the informativeness gate).
    min_gpc, min_cpg : int
        Minimum sites for a modality to count as observed.
    a_prior, m_prior : float or array-like
        Prior accessibility / methylation rate for imputing the missing axis.

    Returns
    -------
    (a_eff, m_eff, a_obs, m_obs) :
        ``a_eff`` / ``m_eff`` are the effective posteriors (observed value, or
        prior where missing).  ``a_obs`` / ``m_obs`` are boolean masks of which
        ends were genuinely observed — for the §2.1 QC of fully- vs
        partially-observed pair fractions.
    """
    a = np.asarray(a, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    n_gpc = np.asarray(n_gpc)
    n_cpg = np.asarray(n_cpg)
    a_obs = (n_gpc >= min_gpc) & np.isfinite(a)
    m_obs = (n_cpg >= min_cpg) & np.isfinite(m)
    a_eff = np.where(a_obs, a, a_prior)
    m_eff = np.where(m_obs, m, m_prior)
    return a_eff, m_eff, a_obs, m_obs


def pair_co_state(qi, qj):
    """Pairwise soft co-state for two ends with 4-state distributions ``qi``, ``qj``.

    Parameters
    ----------
    qi, qj : sequence of 4 array-like
        ``(q_open, q_closed, q_poised, q_active_methylated)`` for end *i* and
        end *j* of a contact (the output of :func:`per_end_joint_state`).

    Returns
    -------
    (co_open, co_closed, co_poised, co_active_methylated, mixed) : tuple of ndarray
        ``co_s = qi[s] · qj[s]`` is the probability *both* ends are in joint
        state *s* (independence across ends, design §3).  ``mixed = 1 − Σ_s
        co_s`` is the probability the two ends are in *different* joint states.
        The five sum to 1 elementwise.

    Notes
    -----
    ``Σ_s co_s = ⟨qi, qj⟩`` (the soft probability the two ends share state) lies
    in [0, 1] because each ``q`` is a probability vector, so ``mixed`` is always
    ≥ 0 (clipped against float round-off).
    """
    co = tuple(np.asarray(qi[s], dtype=np.float64) * np.asarray(qj[s], dtype=np.float64)
               for s in range(4))
    mixed = np.clip(1.0 - sum(co), 0.0, 1.0)
    return (*co, mixed)


def co_state_agreement(qi, qj):
    """Soft probability the two ends **share** their joint state = ``Σ_s qi[s]·qj[s]``.

    The multi-state generalization of the single-context ``co-open + co-closed``
    agreement used in the distance-decay diagnostic.  Equivalent to
    ``sum(pair_co_state(qi, qj)[:4])`` but avoids forming the channels.
    """
    return sum(np.asarray(qi[s], dtype=np.float64) * np.asarray(qj[s], dtype=np.float64)
               for s in range(4))
