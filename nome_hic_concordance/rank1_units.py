"""Section B.1 units: anchor/partner orientation, strata, and the B.2 weight.

Shared by ``scripts/rank1_joint_state.py`` and ``scripts/rank1_diagnostics.py``
so the diagnostics measure the same units the result is computed from.  A QC
panel built on a second, nearly-identical definition of a contact is a panel
that can pass while the pipeline is wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Section B.1 strata.  Named and fixed rather than log-spaced: the 1-10 kb
#: stratum is FLAGGED (residual self-ligation lives there) rather than dropped,
#: and trans is its own stratum, never folded into the widest cis one.
STRATA = [
    ("1-10kb", 1e3, 1e4, True),
    ("10-100kb", 1e4, 1e5, False),
    ("100kb-1Mb", 1e5, 1e6, False),
    ("1-10Mb", 1e6, 1e7, False),
    (">10Mb", 1e7, np.inf, False),
    ("trans", np.nan, np.nan, False),
]
STRATUM_NAMES = [s[0] for s in STRATA]
FLAGGED_STRATA = {s[0] for s in STRATA if s[3]}



# ------------------------------------------------------------------- units ---

def stratum_of(distance, is_cis) -> np.ndarray:
    """Section B.1 stratum name per contact."""
    d = np.asarray(distance, float)
    cis = np.asarray(is_cis, bool)
    out = np.full(len(d), "", dtype=object)
    out[~cis] = "trans"
    for name, lo, hi, _ in STRATA[:-1]:
        m = cis & (d >= lo) & (d < hi)
        out[m] = name
    return out


def pair_ends(df: pd.DataFrame, bin_size: int) -> pd.DataFrame:
    """One row per (contact, orientation): anchor end A, partner end B.

    Section B.1: every contact appears once per end, so a bin sees every
    contact that touches it with that end as the anchor.  When both ends land
    in the same bin the contact contributes both orientations; they are marked
    so nothing downstream treats them as two independent observations.
    """
    e1 = df[df["end_idx"] == 1].set_index(["cell_id", "pair_idx"])
    e2 = df[df["end_idx"] == 2].set_index(["cell_id", "pair_idx"])
    common = e1.index.intersection(e2.index)
    if not len(common):
        return pd.DataFrame()
    e1, e2 = e1.loc[common], e2.loc[common]

    def side(a, b, tag):
        pos = np.minimum(a["pos_5p"].to_numpy(), a["pos_3p"].to_numpy())
        out = pd.DataFrame({
            "cell_id": a.index.get_level_values(0),
            "pair_idx": a.index.get_level_values(1),
            "anchor_bin": pos // bin_size,
            "orientation": tag,
            "contact_dist": a["contact_dist"].to_numpy(),
            "is_cis": a["is_cis"].to_numpy(),
            "conversion_pass": (a["conversion_pass"].to_numpy()
                                & b["conversion_pass"].to_numpy()),
        })
        for side_tag, frame in (("A", a), ("B", b)):
            for mark in ("gch", "hcg"):
                out[f"{side_tag}_{mark}"] = frame[f"centred_{mark}"].to_numpy()
                out[f"{side_tag}_v_{mark}"] = frame[f"var_{mark}"].to_numpy()
                out[f"{side_tag}_n_{mark}"] = frame[f"n_{mark}"].to_numpy()
        return out

    both = side(e1, e2, 1)
    flip = side(e2, e1, 2)
    out = pd.concat([both, flip], ignore_index=True)
    same = out.groupby(["cell_id", "pair_idx"])["anchor_bin"].transform("nunique")
    out["both_ends_in_bin"] = same == 1
    out["stratum"] = stratum_of(out["contact_dist"], out["is_cis"])
    return out[out["stratum"] != ""].reset_index(drop=True)


def weights_for(u: pd.DataFrame, cap_pct: float) -> np.ndarray:
    """Section B.2 inverse total-variance weight, capped within stratum.

    Deliberately NOT a function of |theta - 0.5|: a posterior near 0.5 may be
    uninformative or a genuine intermediate state, and one weight cannot tell
    the two apart.
    """
    v = (u["A_v_gch"] + u["A_v_hcg"] + u["B_v_gch"] + u["B_v_hcg"]).to_numpy()
    w = 1.0 / (v + 1e-12)
    for s in pd.unique(u["stratum"]):
        m = (u["stratum"] == s).to_numpy()
        if m.sum() > 1:
            w[m] = np.minimum(w[m], np.percentile(w[m], cap_pct))
    return w


