"""Section B.7 acceptance criteria.

These tests check the diagnostics can FAIL.  A QC panel that passes on broken
data is worse than no panel, so most of what follows constructs the specific
defect each criterion exists to catch and asserts the verdict flips.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance import rank1_diagnostics as dx
from nome_hic_concordance.rank1_units import STRATUM_NAMES

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "rank1_diagnostics.py"


def _mod():
    spec = importlib.util.spec_from_file_location("r1_dx", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["r1_dx"] = m
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ B.7.1 ---

def _units(n=4000, seed=0, depth_effect=0.0, conv_effect=0.0, n_sites=None):
    """Contacts with a tunable coverage- or conversion-dependent rho1."""
    rng = np.random.default_rng(seed)
    if n_sites is None:
        n_sites = rng.integers(1, 12, n)
    conv = rng.uniform(0.90, 1.0, n)
    # A coordination strength that moves with depth (or conversion) is exactly
    # the artifact B.7.1 / B.7.3 exist to see.
    strength = np.clip(0.5 + depth_effect * (n_sites - n_sites.mean())
                       / max(np.ptp(n_sites), 1)
                       + conv_effect * (conv - conv.mean()) / max(np.ptp(conv), 1e-9),
                       0.02, 0.98)
    u1 = rng.normal(size=n)
    u2 = strength * u1 + np.sqrt(np.maximum(1 - strength ** 2, 0)) * rng.normal(size=n)
    noise = 0.4
    d = pd.DataFrame({
        "cell_id": rng.integers(0, 5, n).astype(str),
        "pair_idx": np.arange(n),
        "A_gch": u1 + noise * rng.normal(size=n),
        "A_hcg": -0.8 * u1 + noise * rng.normal(size=n),
        "B_gch": u2 + noise * rng.normal(size=n),
        "B_hcg": -0.8 * u2 + noise * rng.normal(size=n),
        "A_v_gch": 0.02, "A_v_hcg": 0.05,
        "B_v_gch": 0.02, "B_v_hcg": 0.05,
        "A_n_gch": n_sites, "A_n_hcg": np.maximum(n_sites // 2, 1),
        "B_n_gch": n_sites, "B_n_hcg": np.maximum(n_sites // 2, 1),
        "stratum": "100kb-1Mb",
    })
    return d, conv


def test_flat_data_gives_a_flat_verdict():
    u, _ = _units(depth_effect=0.0, seed=1)
    t = dx.coverage_flatness(u)
    assert len(t) >= 5
    v = dx.flatness_verdict(t)
    assert v["verdict"] == "flat", v


def test_rho_rising_with_coverage_reads_as_under_corrected():
    """The design's own reading: rising -> under-corrected."""
    u, _ = _units(depth_effect=0.8, seed=2)
    v = dx.flatness_verdict(dx.coverage_flatness(u))
    assert v["verdict"] == "under_corrected", v
    assert v["trend_rho1"] > 0


def test_rho_falling_with_coverage_reads_as_over_corrected():
    u, _ = _units(depth_effect=-0.8, seed=3)
    v = dx.flatness_verdict(dx.coverage_flatness(u))
    assert v["verdict"] == "over_corrected", v


def test_a_d_trend_is_blamed_upstream_not_on_the_convention():
    """D needs no measurement-error correction, so a trend in D cannot be
    fixed by changing error_convention -- the verdict must say so rather than
    recommending a convention that will not help."""
    t = pd.DataFrame({"mean_depth": np.arange(10, dtype=float),
                      "rho1": np.full(10, 0.5), "rho2": np.full(10, 0.1),
                      "D": np.linspace(0.0, 0.6, 10)})
    t.attrs["trend"] = {"rho1": 0.0, "rho2": 0.0, "D": 0.6}
    v = dx.flatness_verdict(t)
    assert v["verdict"] == "upstream_normalization"
    assert "A.3.4" in v["message"]


def test_flatness_stratifies_on_total_depth_across_both_ends():
    """Stratifying on one end alone leaves the other free to vary inside a
    stratum, which blurs the trend the panel exists to see."""
    u, _ = _units(seed=4)
    t = dx.coverage_flatness(u)
    assert t["mean_depth"].is_monotonic_increasing
    assert (t["mean_depth"] > t["mean_n_gch"]).all(), (
        "depth must pool all four site counts, not one end's")


def test_too_few_contacts_says_undetermined_rather_than_guessing():
    u, _ = _units(n=40, seed=5)
    v = dx.flatness_verdict(dx.coverage_flatness(u, min_contacts=50))
    assert v["verdict"] == "undetermined"


# ------------------------------------------------------------------ B.7.3 ---

def test_a_conversion_trend_is_detected():
    """The artifact is molecule-level, so it survives the mate shuffle and the
    B.6.2 null; this panel is the only thing that can see it."""
    u, conv = _units(conv_effect=0.9, seed=6)
    t = dx.conversion_association(u, conv)
    assert len(t) >= 4
    assert abs(t.attrs["trend"]["rho1"]) > dx.FLATNESS_TOL


def test_no_conversion_trend_when_conversion_is_irrelevant():
    u, conv = _units(conv_effect=0.0, seed=7)
    t = dx.conversion_association(u, conv)
    assert abs(t.attrs["trend"]["rho1"]) < dx.FLATNESS_TOL


# ------------------------------------------------------------------ B.7.2 ---

def _rows(n_bins=40, seed=0, rho2_tracks=True):
    rng = np.random.default_rng(seed)
    out = []
    for g in ("G0", "G1"):
        for i, s in enumerate(STRATUM_NAMES[:5]):
            for b in range(n_bins):
                rho1 = 0.8 * np.exp(-0.4 * i) + 0.02 * rng.normal()
                rho2 = (0.3 * rho1 if rho2_tracks
                        else 0.25 + 0.02 * rng.normal())
                out.append({"group": g, "chrom": "chr1",
                            "bin_start": b * 250_000, "stratum": s,
                            "rho1": rho1, "rho2": rho2,
                            "D": 0.05 * rng.normal(),
                            "D_excess": 0.02 * rng.normal(),
                            "rho1_mate_shuffle_mean": 0.01 * rng.normal(),
                            "z_mate_shuffle": 6.0 + rng.normal()})
    return pd.DataFrame(out)


def test_rho2_tracking_rho1_shows_as_a_flat_relative_profile():
    """That flat line at 1 is what a rescaled artifact looks like."""
    d = dx.distance_decay(_rows(rho2_tracks=True), STRATUM_NAMES)
    assert np.allclose(d["rho2_rel"].dropna(), 1.0, atol=0.05)


def test_a_genuinely_different_rho2_profile_departs_from_one():
    d = dx.distance_decay(_rows(rho2_tracks=False), STRATUM_NAMES)
    assert d["rho2_rel"].max() > 1.3, (
        "an rho2 that does not decay with rho1 must show up as a departure")


def test_distance_decay_keeps_the_design_stratum_order():
    d = dx.distance_decay(_rows(), STRATUM_NAMES)
    for g, sub in d.groupby("group", observed=True):
        assert list(sub["stratum"].astype(str)) == STRATUM_NAMES[:5]


# ------------------------------------------------------------------ B.7.5 ---

def test_mate_shuffle_sanity_passes_when_rho1_collapses():
    s = dx.mate_shuffle_sanity(_rows())
    assert s["passes"].all()


def test_mate_shuffle_sanity_fails_when_rho1_survives_the_shuffle():
    """If rho1 survives reassigning mates it is measuring shared coverage or a
    baseline that still carries cell-type structure, not coordination."""
    r = _rows()
    r["rho1_mate_shuffle_mean"] = r["rho1"] * 0.9
    s = dx.mate_shuffle_sanity(r)
    assert not s["passes"].any()


# ------------------------------------------------------------------ B.7.4 ---

def test_p_values_are_uniform_under_simulated_rank_one_truth():
    """Without this the rank-2 calls have no licence."""
    t = dx.null_calibration(n_units=40, n_contacts=400, n_boot=120, seed=11)
    assert t.attrs["ks_uniform_p"] > 0.01, t.attrs
    assert 0.0 <= t.attrs["frac_below_05"] <= 0.20


def test_the_same_machinery_has_power_against_injected_rank_two():
    """A test uniform under the null AND under the alternative is uniform
    because it is measuring nothing."""
    t = dx.null_calibration(n_units=30, n_contacts=400, n_boot=120, seed=12,
                            rank2=0.9)
    assert t.attrs["frac_below_05"] > 0.5, t.attrs["frac_below_05"]


def test_ks_uniform_rejects_an_obviously_non_uniform_sample():
    assert dx._ks_uniform(np.linspace(0, 1, 200)) > 0.5
    assert dx._ks_uniform(np.full(200, 0.9)) < 1e-6


# ------------------------------------------------------- B.7.6 / B.7.7 ------

def _arm(seed, shift=0.0):
    rng = np.random.default_rng(seed)
    n = 60
    return pd.DataFrame({
        "group": "G0", "chrom": "chr1",
        "bin_start": np.arange(n) * 250_000, "stratum": "100kb-1Mb",
        "D": rng.normal(size=n) * 0.1 + shift,
        "D_excess": rng.normal(size=n) * 0.05,
        "rho1": rng.uniform(0.2, 0.8, n), "rho2": rng.uniform(0, 0.3, n),
        "rho2_over_rho1": rng.uniform(0, 0.5, n),
        "N_eff": rng.uniform(50, 500, n), "flag_pd_clipped": False,
    })


def test_ablation_compares_the_same_bin_across_arms():
    """Comparing marginal distributions would hide the thing worth seeing:
    two arms with identical medians can still disagree bin by bin."""
    a, b = _arm(1), _arm(1)
    b["D"] = a["D"] + 0.2                       # every bin moves the same way
    m = dx.compare_ablation({"location": a, "full": b}, "B76")
    s = dx.ablation_summary(m, "D")
    row = s[s["arm"] == "full"].iloc[0]
    assert row["median_shift_vs_first"] == pytest.approx(0.2, abs=1e-9)
    assert row["corr_with_first"] == pytest.approx(1.0, abs=1e-9)


def test_ablation_notices_per_bin_disagreement_behind_equal_medians():
    a = _arm(2)
    b = a.copy()
    sign = np.where(np.arange(len(b)) % 2 == 0, 1.0, -1.0)
    b["D"] = a["D"] + 0.3 * sign               # medians match, bins do not
    m = dx.compare_ablation({"one": a, "two": b}, "B77")
    s = dx.ablation_summary(m, "D")
    row = s[s["arm"] == "two"].iloc[0]
    assert abs(row["median_shift_vs_first"]) < 0.05
    assert row["max_abs_shift_vs_first"] > 0.25, (
        "the per-bin shift must survive when the medians agree")
    assert row["corr_with_first"] < 0.9


def test_an_arm_that_dropped_bins_is_counted_not_silently_aligned():
    """--min-hcg-sites 3 keeps fewer bins than 1; the summary has to say so."""
    a = _arm(3)
    b = a.iloc[:30].copy()
    m = dx.compare_ablation({"hcg1": a, "hcg3": b}, "B77")
    s = dx.ablation_summary(m, "D")
    assert s[s["arm"] == "hcg1"].iloc[0]["n_bins"] == 60
    assert s[s["arm"] == "hcg3"].iloc[0]["n_bins"] == 30
    assert s[s["arm"] == "hcg3"].iloc[0]["n_shared_with_first"] == 30


# --------------------------------------------------------------- end to end ---

@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """The whole panel over a small synthetic Part A output."""
    pytest.importorskip("matplotlib")
    r1_cli = importlib.util.spec_from_file_location(
        "r1_cli_for_dx",
        Path(__file__).resolve().parent.parent / "scripts" / "rank1_joint_state.py")
    r1 = importlib.util.module_from_spec(r1_cli)
    sys.modules["r1_cli_for_dx"] = r1
    r1_cli.loader.exec_module(r1)

    from tests.test_rank1_cli import _ends
    d = tmp_path_factory.mktemp("dx")
    df = _ends(n_pairs=160, n_cells=6, seed=5, bin_size=250_000, span=2,
               rank2=1.0)
    rng = np.random.default_rng(0)
    df["conversion_rate"] = rng.uniform(0.95, 1.0, len(df))
    # Vary the HCG depth so the B.7.7 floor has something to bite on; a
    # constant n_hcg would make every arm above it empty.
    df["n_hcg"] = rng.integers(1, 6, len(df))
    df["n_gch"] = rng.integers(2, 10, len(df))
    df["theta_gch"] = df["centred_gch"] + 0.4
    df["mu_gch"] = 0.4
    df["prior_sd_gch"] = 0.25
    shard = d / "post" / "chrom=chr1"
    shard.mkdir(parents=True)
    df.to_parquet(shard / "all.parquet", index=False)

    man = d / "s.tsv"
    man.write_text("name\tbam\tgroup\tcolor\tbatch\n" + "".join(
        f"cell{i}\t-\tG{i % 2}\t#{'aabbcc' if i % 2 == 0 else 'ddeeff'}\tb{i}\n"
        for i in range(6)))
    out_r1 = d / "r1"
    assert r1.main(["--posteriors", str(d / "post"), "--samples", str(man),
                    "--group-order", "G0,G1", "-o", str(out_r1),
                    "--bin-size", "250000", "--min-contacts", "20",
                    "--n-boot-null", "30", "--n-boot-mate", "30",
                    "--n-boot-ci", "30"]) == 0

    m = _mod()
    qc = d / "qc"
    assert m.main(["--posteriors", str(d / "post"),
                   "--rank1", str(out_r1 / "04_rank1.parquet"),
                   "--samples", str(man), "--group-order", "G0,G1",
                   "-o", str(qc), "--bin-size", "250000",
                   "--null-units", "12", "--null-boot", "40",
                   "--ablation-args",
                   "--min-contacts 20 --n-boot-null 20 --n-boot-mate 20 "
                   "--n-boot-ci 20"]) == 0
    return {"qc": qc, "mod": m}


def test_all_seven_criteria_produce_an_artefact(run):
    qc = run["qc"]
    for f in ("B71_coverage_flatness.tsv", "B72_distance_decay.tsv",
              "B73_conversion_association.tsv", "B75_mate_shuffle.tsv",
              "B74_null_rank-1.tsv", "B74_null_rank-2.tsv",
              "B76_gch_norm_summary.tsv", "B77_min_hcg_sites_summary.tsv"):
        assert (qc / f).exists(), f
    figs = sorted(p.name for p in qc.glob("*.pdf"))
    assert len(figs) >= 5, figs


def test_the_verdicts_are_written_down_not_only_logged(run):
    v = json.loads((run["qc"] / "_b7_verdicts.json").read_text())
    assert v["B71_coverage_flatness"]["verdict"] in (
        "flat", "under_corrected", "over_corrected", "undetermined",
        "upstream_normalization")
    assert "ks_uniform_p" in v["B74_null_calibration"]
    assert v["B74_null_calibration"]["power_vs_rank2"] > 0.3, (
        "the injected-rank-2 arm must have power, or the null arm proves "
        "nothing")
    assert set(v["B76_B77_ablations"]) == {"B76_gch_norm", "B77_min_hcg_sites"}


def test_the_normalization_ablation_actually_ran_three_arms(run):
    s = pd.read_csv(run["qc"] / "B76_gch_norm_summary.tsv", sep="\t")
    assert set(s["arm"]) == {"location", "full", "none"}
    assert (s["n_bins"] > 0).all()


def test_the_min_hcg_ablation_loses_bins_as_the_floor_rises(run):
    """B.7.7's expected behaviour: D stable, coverage shrinking."""
    s = pd.read_csv(run["qc"] / "B77_min_hcg_sites_summary.tsv", sep="\t")
    assert list(s["arm"]) == ["hcg1", "hcg2", "hcg3"]
    assert s.iloc[0]["n_bins"] >= s.iloc[-1]["n_bins"]


def test_pdfs_embed_truetype_not_type3(run):
    """Type 3 text is not editable in Illustrator and several journals reject
    it outright."""
    for pdf in run["qc"].glob("*.pdf"):
        raw = pdf.read_bytes()
        assert b"/Type3" not in raw, pdf.name
        assert b"/FontFile2" in raw or b"/Type0" in raw, pdf.name
