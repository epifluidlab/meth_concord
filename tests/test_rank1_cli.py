"""scripts/rank1_joint_state.py: Part A's posteriors -> the Section B.8 table.

The tests that matter here are the ones about what gets CLAIMED: that class II
is only reached by rejecting rank 1 after FDR, that the BED ranking uses the
excess over the null rather than the raw |D|, and that a contact enters each
bin it touches exactly once per orientation.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "rank1_joint_state.py"


def _mod():
    spec = importlib.util.spec_from_file_location("r1_cli", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["r1_cli"] = m
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ strata ---

def test_trans_is_its_own_stratum():
    m = _mod()
    s = m.stratum_of([-1, 5e3, 5e4, 5e5, 5e6, 5e7],
                     [False, True, True, True, True, True])
    assert s.tolist() == ["trans", "1-10kb", "10-100kb", "100kb-1Mb",
                          "1-10Mb", ">10Mb"]


def test_the_short_range_stratum_is_flagged_not_dropped():
    """Residual self-ligation lives in 1-10 kb.  A.3.6 says handle it here by
    flagging, so the bin stays visible rather than silently vanishing."""
    m = _mod()
    assert "1-10kb" in m.FLAGGED_STRATA
    assert "1-10kb" in m.STRATUM_NAMES


def test_a_contact_below_the_first_edge_is_excluded_not_clamped():
    m = _mod()
    assert m.stratum_of([500.0], [True]).tolist() == [""]


# ------------------------------------------------------------------- units ---

def _ends(n_pairs=40, n_cells=3, seed=0, bin_size=250_000, span=3,
          same_bin=False, rank2=0.0):
    """A tiny Part A posterior frame.

    Every contact is at least 1 kb apart, because anything below the first
    Section B.1 edge is excluded rather than clamped -- a fixture that drifts
    under the floor silently halves itself.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cells):
        for p in range(n_pairs):
            if same_bin:
                p1 = int(rng.integers(0, bin_size // 4))
                p2 = p1 + 2_000 + int(rng.integers(0, bin_size // 4))
            else:
                b1 = int(rng.integers(0, span))
                b2 = int(rng.integers(0, span))
                p1 = b1 * bin_size + int(rng.integers(0, bin_size // 2))
                p2 = b2 * bin_size + int(rng.integers(0, bin_size // 2))
                if abs(p2 - p1) < 1_000:
                    p2 = p1 + 1_500
            u = rng.normal()
            # rank2 adds coordination on the HCG residual that the shared
            # open/closed axis cannot carry -- the class II alternative.
            e = rng.normal()
            for end, pos in ((1, p1), (2, p2)):
                rows.append({
                    "cell_id": f"cell{c}", "pair_idx": p, "end_idx": end,
                    "pos_5p": pos, "pos_3p": pos + 80,
                    "contact_dist": abs(p2 - p1), "is_cis": True,
                    "conversion_pass": True,
                    "centred_gch": u + 0.3 * rng.normal(),
                    "centred_hcg": (-0.8 * u + rank2 * e
                                    + 0.3 * rng.normal()),
                    "var_gch": 0.02, "var_hcg": 0.05,
                    "n_gch": 3, "n_hcg": 1,
                })
    return pd.DataFrame(rows)


def test_each_contact_enters_a_bin_once_per_orientation():
    """B.1: for each contact with an end in b, that end is A and the mate B."""
    m = _mod()
    df = _ends(n_pairs=20, n_cells=1)
    u = m.pair_ends(df, bin_size=250_000)
    assert len(u) == 2 * 20, "one row per end, both orientations"
    for (cell, pair), sub in u.groupby(["cell_id", "pair_idx"]):
        assert sorted(sub["orientation"]) == [1, 2]


def test_both_ends_in_one_bin_is_marked(tmp_path):
    """They are two orientations of ONE contact, not two observations."""
    m = _mod()
    df = _ends(n_pairs=30, n_cells=1, same_bin=True)
    u = m.pair_ends(df, bin_size=250_000)
    assert u["both_ends_in_bin"].all()
    assert u["anchor_bin"].nunique() == 1


def test_the_anchor_and_partner_swap_between_orientations():
    m = _mod()
    df = _ends(n_pairs=5, n_cells=1, span=3)
    u = m.pair_ends(df, bin_size=250_000)
    one = u[u["orientation"] == 1].set_index(["cell_id", "pair_idx"])
    two = u[u["orientation"] == 2].set_index(["cell_id", "pair_idx"])
    common = one.index.intersection(two.index)
    assert np.allclose(one.loc[common, "A_gch"], two.loc[common, "B_gch"])
    assert np.allclose(one.loc[common, "B_hcg"], two.loc[common, "A_hcg"])


def test_weights_are_capped_within_stratum():
    """One very deep contact must not carry a bin on its own."""
    m = _mod()
    df = _ends(n_pairs=200, n_cells=1)
    u = m.pair_ends(df, bin_size=250_000)
    u.loc[u.index[:2], ["A_v_gch", "A_v_hcg", "B_v_gch", "B_v_hcg"]] = 1e-9
    uncapped = 1.0 / (u[["A_v_gch", "A_v_hcg", "B_v_gch",
                         "B_v_hcg"]].sum(1).to_numpy() + 1e-12)
    w = m.weights_for(u, cap_pct=99.0)
    assert w.max() < uncapped.max() / 100, "the outlier weight must be capped"
    for s_ in pd.unique(u["stratum"]):
        mask = (u["stratum"] == s_).to_numpy()
        if mask.sum() > 1:
            assert w[mask].max() == pytest.approx(
                np.percentile(uncapped[mask], 99.0)), (
                "the cap is the within-stratum 99th percentile")


def test_the_weight_ignores_distance_from_a_half():
    """B.2 forbids it: a posterior near 0.5 may be uninformative OR a genuine
    intermediate state, and one weight cannot tell those apart."""
    m = _mod()
    df = _ends(n_pairs=30, n_cells=1)
    u = m.pair_ends(df, bin_size=250_000)
    before = m.weights_for(u, 99.0)
    moved = u.copy()
    for c in ("A_gch", "A_hcg", "B_gch", "B_hcg"):
        moved[c] = 0.5                       # variances untouched
    assert np.allclose(before, m.weights_for(moved, 99.0))


# ------------------------------------------------------------- the B.8 row ---

@pytest.fixture(scope="module")
def run(tmp_path_factory):
    m = _mod()
    d = tmp_path_factory.mktemp("r1")
    df = _ends(n_pairs=140, n_cells=6, seed=3, bin_size=250_000, span=2,
               rank2=1.2)
    shard = d / "post" / "chrom=chr1"
    shard.mkdir(parents=True)
    df.to_parquet(shard / "all.parquet", index=False)

    man = d / "s.tsv"
    man.write_text("name\tbam\tgroup\tcolor\tbatch\n" + "".join(
        f"cell{i}\t-\tG{i % 2}\t#{'aabbcc' if i % 2 == 0 else 'ddeeff'}\tb{i}\n"
        for i in range(6)))

    out = d / "out"
    rc = m.main(["--posteriors", str(d / "post"), "--samples", str(man),
                 "--group-order", "G0,G1", "-o", str(out),
                 "--bin-size", "250000", "--min-contacts", "20",
                 "--n-boot-null", "40", "--n-boot-mate", "40",
                 "--n-boot-ci", "40", "--seed", "1"])
    assert rc == 0
    return {"mod": m, "out": out, "man": str(man), "post": str(d / "post")}


def test_output_has_every_design_b8_column(run):
    d = pd.read_parquet(run["out"] / "04_rank1.parquet")
    want = {
        "group", "chrom", "bin_start", "bin_end", "stratum",
        "n_contacts", "n_cells", "N_eff",
        "D", "D_null_mean", "D_excess", "p_rank1", "q_rank1",
        "rho1", "rho1_ci_lo", "rho1_ci_hi", "z_mate_shuffle",
        "rho2", "rho2_null_mean", "rho2_excess", "rho2_over_rho1",
        "w_A_gch", "w_A_hcg", "w_B_gch", "w_B_hcg", "phi_A", "phi_B",
        "sign_lambda_h", "sign_rho_u", "rank1_pattern_class",
        "asymmetry_frob", "asymmetry_angle",
        "sigma_AB_gg", "sigma_AB_gh", "sigma_AB_hg", "sigma_AB_hh",
        "reliability_gch", "reliability_hcg",
        "flag_pd_clipped", "flag_low_neff", "flag_pairwise_incomplete",
        "flag_short_range", "error_convention", "moment_mode", "gch_norm",
    }
    assert want <= set(d.columns), f"missing: {sorted(want - set(d.columns))}"


def test_class_two_requires_rejecting_rank_one_after_fdr(run):
    """The headline class must survive multiple testing, not just the raw p."""
    d = pd.read_parquet(run["out"] / "04_rank1.parquet")
    ii = d[d["bin_class"] == "II"]
    assert (ii["q_rank1"] < 0.05).all()
    assert (ii["sign_rho_u"] > 0).all(), "a negative rho_u is class IV"
    assert ii["rho1_significant"].all()


def test_fdr_is_computed_within_group(run):
    """Pooling across groups would let a well-covered cell type set the
    threshold for a rare one."""
    d = pd.read_parquet(run["out"] / "04_rank1.parquet")
    for g, sub in d.groupby("group"):
        ok = sub["p_rank1"].notna()
        if ok.sum() < 2:
            continue
        # BH is monotone in p within the set it was computed over.
        s = sub[ok].sort_values("p_rank1")
        assert (s["q_rank1"].to_numpy() ==
                np.maximum.accumulate(s["q_rank1"].to_numpy())).all()


def test_bed_ranks_class_two_by_excess_not_raw_d(run):
    """|D| has a strictly positive null expectation at finite n, so ranking on
    the raw value would put the thinnest bins on top."""
    d = pd.read_parquet(run["out"] / "04_rank1.parquet")
    bed = run["out"] / "classII_mark_specific_beyond_axis.bed"
    assert (d["bin_class"] == "II").any(), (
        "the fixture injects rank-2 structure; if no bin reaches class II the "
        "test that guards the primary figure object never runs")
    rows = [l.split("\t") for l in bed.read_text().splitlines()]
    ii = d[d["bin_class"] == "II"].sort_values("D_excess", ascending=False)
    assert [int(r[1]) for r in rows] == ii["bin_start"].tolist()


def test_every_bed_row_is_a_valid_interval(run):
    for bed in Path(run["out"]).glob("class*.bed"):
        for line in bed.read_text().splitlines():
            f = line.split("\t")
            assert len(f) == 6
            assert int(f[2]) > int(f[1])
            assert 0 <= int(f[4]) <= 1000
            assert "/" not in f[3], "group names must be path-safe in a BED"


def test_qc_tables_cover_the_computable_acceptance_criteria(run):
    qc = run["out"] / "qc"
    for f in ("B72_distance_decay.tsv", "B75_mate_shuffle_rho1.tsv",
              "bin_class_census.tsv"):
        assert (qc / f).exists(), f
    decay = pd.read_csv(qc / "B72_distance_decay.tsv", sep="\t")
    assert {"rho1", "rho2", "D", "D_excess"} <= set(decay.columns)


def test_metadata_records_the_conventions_the_numbers_depend_on(run):
    meta = json.loads((run["out"] / "_rank1_metadata.json").read_text())
    assert meta["moment_mode"] == "complete"
    assert meta["error_convention"] == "raw"
    assert meta["flagged_strata"] == ["1-10kb"]


def test_pairwise_mode_keeps_more_data_than_complete_case(run, tmp_path):
    """B.4: with --min-hcg-sites 1 an end may carry one mark and not the
    other, and complete-case throws the contact away."""
    m = run["mod"]
    src = pd.read_parquet(Path(run["post"]) / "chrom=chr1" / "all.parquet")
    src.loc[src.index[::4], "n_hcg"] = 0        # a quarter of ends lose HCG
    d = tmp_path / "post" / "chrom=chr1"
    d.mkdir(parents=True)
    src.to_parquet(d / "all.parquet", index=False)

    counts = {}
    for mode in ("complete", "pairwise"):
        o = tmp_path / mode
        rc = m.main(["--posteriors", str(tmp_path / "post"), "--samples",
                     run["man"], "--group-order", "G0,G1", "-o", str(o),
                     "--bin-size", "250000", "--min-contacts", "20",
                     "--moment-mode", mode, "--n-boot-null", "20",
                     "--n-boot-mate", "20", "--n-boot-ci", "20"])
        assert rc == 0
        t = pd.read_parquet(o / "04_rank1.parquet")
        counts[mode] = t["n_contacts"].sum()
        assert (t["moment_mode"] == mode).all()
    assert counts["pairwise"] > counts["complete"], counts
