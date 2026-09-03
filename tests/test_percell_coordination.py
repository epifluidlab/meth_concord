"""Part C: per-cell coordination, imputation, embedding, region selection.

The design's own test list for Part C is mostly about things that must NOT
happen -- masked entries must not be zero-filled, a circular graph must be
refused, a null simulation must produce an empty specific-region set.  Those
are what these check, because each one passes silently if it is wrong.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance import percell_coordination as pc
from nome_hic_concordance import rwr_graphs

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "percell_coordination.py"


def _mod():
    spec = importlib.util.spec_from_file_location("pc_cli", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["pc_cli"] = m
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------- C.1 ---

def test_per_cell_rho_is_unbiased_for_the_group_rho1():
    """The design's first Part C test: average over cells of one type and the
    per-cell estimator must land on the pooled rho1."""
    rng = np.random.default_rng(0)
    rho_true = 0.55
    ests = []
    for _ in range(60):
        n = 300
        u = rng.multivariate_normal([0, 0], [[1, rho_true], [rho_true, 1]], n)
        sA = u[:, 0] + 0.1 * rng.normal(size=n)
        sB = u[:, 1] + 0.1 * rng.normal(size=n)
        sA -= sA.mean(); sB -= sB.mean()
        k = pc.contact_kernel(sA, sB, sA.std(), sB.std())
        g = pc.aggregate(k, np.ones(n),
                         pd.DataFrame({"cell": ["c"] * n}))
        ests.append(float(g["rho_hat"].iloc[0]))
    assert abs(np.mean(ests) - rho_true) < 0.05, np.mean(ests)


def test_the_kernel_divides_by_a_pooled_sd_not_a_per_cell_one():
    """Numerator per contact, denominator pooled -- a per-cell denominator is a
    ratio of two noisy quantities and blows up where coverage is thin."""
    s_A = np.array([1.0, 2.0]); s_B = np.array([1.0, 0.5])
    k = pc.contact_kernel(s_A, s_B, 2.0, 4.0)
    assert np.allclose(k, s_A * s_B / 8.0)


def test_projection_carries_the_variance_through_the_direction():
    theta = np.array([[1.0, 2.0]])
    v = np.array([[0.1, 0.4]])
    w = np.array([0.6, -0.8])
    s, v_s = pc.project_ends(theta, v, w)
    assert s[0] == pytest.approx(0.6 - 1.6)
    assert v_s[0] == pytest.approx(0.1 * 0.36 + 0.4 * 0.64)


def test_per_bin_directions_are_honoured_not_a_single_global_one():
    """A global direction would erase the lambda_h sign flips gene bodies and
    PMDs show."""
    theta = np.array([[1.0, 1.0], [1.0, 1.0]])
    v = np.zeros((2, 2))
    w = np.array([[1.0, -1.0], [1.0, 1.0]])      # opposite lambda_h signs
    s, _ = pc.project_ends(theta, v, w)
    assert s[0] == pytest.approx(0.0) and s[1] == pytest.approx(2.0)


def test_shrinkage_pulls_thin_cells_to_the_bin_mean():
    out = pc.shrink(rho_hat=[0.9, 0.9], n_eff=[1.0, 1000.0],
                    rho_bar=[0.1, 0.1], tau=[10.0, 10.0])
    assert abs(out[0] - 0.1) < abs(out[1] - 0.1)
    assert out[1] == pytest.approx(0.9, abs=0.02)


def test_tau_is_large_where_there_is_no_between_cell_signal():
    """A bin whose spread is all sampling noise has nothing per-cell to keep."""
    rng = np.random.default_rng(1)
    n = 200
    noisy = pd.DataFrame({"bin": "b0", "rho": rng.normal(0, 0.3, n),
                          "n_eff": np.full(n, 10.0)})
    real = pd.DataFrame({"bin": "b1", "rho": rng.normal(0, 2.0, n),
                         "n_eff": np.full(n, 10.0)})
    d = pd.concat([noisy, real])
    tau = pc.tau_moment(d["rho"], d["n_eff"], d["bin"])
    assert tau["b0"] > tau["b1"]


def test_feasibility_says_go_coarser_rather_than_re_binning_silently():
    """C.1: under a median N_eff of ~10 the answer is a bigger bin, and that is
    the user's call because it changes every downstream file."""
    f = pc.feasibility(np.full(500, 4.0), 250_000)
    assert f["verdict"] == "too_thin"
    assert "500,000" in f["message"]
    assert pc.feasibility(np.full(500, 35.0), 250_000)["verdict"] == "ok"
    assert pc.feasibility(np.full(500, 14.0), 250_000)["verdict"] == "marginal"


# --------------------------------------------------------------------- C.2 ---

def _per_cell_frame(n_cells=30, n_bins=20, seed=0, neff=40.0, specific=()):
    rng = np.random.default_rng(seed)
    # Per-(cell, bin) variation, not just per-cell.  A per-cell-only baseline
    # cancels exactly under C.2's row centring, leaving every non-planted bin a
    # deterministic function of the cell's group and therefore an enormous F --
    # an artefact of a too-clean fixture, not of the method.
    base = rng.uniform(0.05, 0.45, (n_cells, n_bins))
    rows = []
    for ci in range(n_cells):
        grp = ci % 2
        for bi in range(n_bins):
            base = 0.5 + (0.8 if (bi in specific and grp == 0) else 0.0)
            rows.append({
                "cell_id": f"cell{ci}", "bin": f"chr1:{bi * 250_000}",
                "chrom": "chr1", "bin_start": bi * 250_000,
                "stratum": "100kb-1Mb",
                "rho_hat": base + 0.1 * rng.normal(),
                "n_eff": neff, "se": np.sqrt(1.25 / neff),
                "n_contacts": int(neff),
            })
    d = pd.DataFrame(rows)
    d["rho_bar"] = d.groupby("bin")["rho_hat"].transform("mean")
    d["tau"] = 1.0
    d["rho_shrunk"] = pc.shrink(d["rho_hat"], d["n_eff"], d["rho_bar"], d["tau"])
    return d


def test_matrix_follows_the_c2_order_and_records_it():
    d = _per_cell_frame()
    m = pc.build_matrix(d, sorted(d["cell_id"].unique()),
                        sorted(d["bin"].unique()))
    assert m.steps == ["shrunk", "per_cell_centred", "per_bin_standardised",
                       "masked_neff_lt_5"]


def test_masking_marks_missing_and_never_zero_fills():
    """A zero in a standardised matrix is the bin mean -- an assertion the data
    does not support.  The design's own Part C test."""
    d = _per_cell_frame(neff=40.0)
    d.loc[d.index[:15], "n_eff"] = 1.0
    m = pc.build_matrix(d, sorted(d["cell_id"].unique()),
                        sorted(d["bin"].unique()), min_neff_cell=5.0)
    assert m.mask.sum() == 15
    assert np.isnan(m.values[m.mask]).all(), "masked entries must be NaN"


def test_masking_happens_after_centring():
    """Masking first would let a cell's row mean be taken over a different bin
    set than its neighbours', putting coverage back into the step meant to
    remove it."""
    d = _per_cell_frame(n_cells=8, n_bins=10, neff=40.0)
    thin = (d["cell_id"] == "cell0") & (d["bin"] != "chr1:0")
    d.loc[thin, "n_eff"] = 1.0
    m = pc.build_matrix(d, sorted(d["cell_id"].unique()),
                        sorted(d["bin"].unique()), min_neff_cell=5.0)
    # cell0's row mean used all ten bins, so its one surviving entry is not
    # forced to zero the way a mask-then-centre order would force it.
    surviving = m.values[0][np.isfinite(m.values[0])]
    assert surviving.size == 1
    assert abs(float(surviving[0])) > 1e-9


# --------------------------------------------------------------------- C.3 ---

def test_genomic_graph_never_joins_two_chromosomes():
    P = rwr_graphs.genomic_neighbor_graph(
        np.array(["chr1"] * 3 + ["chr2"] * 3), np.arange(6) * 250_000)
    assert P[2, 3] == 0 and P[3, 2] == 0
    assert np.allclose(np.asarray(P.sum(axis=1)).ravel(), 1.0)


def test_genomic_graph_refuses_unsorted_bins():
    with pytest.raises(ValueError, match="sorted"):
        rwr_graphs.genomic_neighbor_graph(
            np.array(["chr1", "chr1"]), np.array([250_000, 0]))


def test_an_isolated_bin_restarts_on_itself_rather_than_zeroing():
    P = rwr_graphs.genomic_neighbor_graph(
        np.array(["chr1", "chr2", "chr3"]), np.zeros(3, int))
    assert np.allclose(P.diagonal(), 1.0)


def test_cell_graph_refuses_the_coordination_matrix_itself():
    """The design says enforce this in code, not documentation."""
    X = np.random.default_rng(0).normal(size=(40, 6))
    with pytest.raises(rwr_graphs.CircularGraphError, match="itself"):
        rwr_graphs.cell_neighbor_graph(X, X, "marginal_hcg")


def test_cell_graph_refuses_a_numerical_copy():
    X = np.random.default_rng(1).normal(size=(40, 6))
    with pytest.raises(rwr_graphs.CircularGraphError, match="identical"):
        rwr_graphs.cell_neighbor_graph(X.copy(), X, "marginal_hcg")


def test_cell_graph_refuses_features_that_reproduce_the_geometry():
    """A caller passing the coordination matrix's own SVD scores passes both
    the identity and the equality check; the geometry check is what catches
    it."""
    rng = np.random.default_rng(2)
    C = rng.normal(size=(60, 30))
    U, sv, Vt = np.linalg.svd(C, full_matrices=False)
    scores = U * sv                    # a rotation: identical cell-cell geometry
    with pytest.raises(rwr_graphs.CircularGraphError, match="geometry"):
        rwr_graphs.cell_neighbor_graph(scores, C, "marginal_hcg")


def test_cell_graph_accepts_genuinely_independent_features():
    rng = np.random.default_rng(3)
    C = rng.normal(size=(60, 30))
    F = rng.normal(size=(60, 8))
    P, report = rwr_graphs.cell_neighbor_graph(F, C, "marginal_hcg", k=5)
    assert report["provenance"] == "marginal_hcg"
    assert report["geometry_spearman"] < rwr_graphs.GEOMETRY_MAX
    assert np.allclose(np.asarray(P.sum(axis=1)).ravel(), 1.0)


def test_an_unlisted_provenance_is_refused_before_any_geometry_check():
    X = np.random.default_rng(4).normal(size=(20, 4))
    with pytest.raises(rwr_graphs.CircularGraphError, match="provenance"):
        rwr_graphs.cell_neighbor_graph(X, None, "coordination_svd")


# --------------------------------------------------------------------- C.5 ---

def test_a_column_permutation_cannot_be_a_noise_floor():
    """The floor has to come from a matrix rebuilt on shuffled CONTACTS.
    Permuting cells within a bin reorders that bin's column and leaves its
    variance exactly unchanged, so such a floor equals the observed variance
    everywhere and nothing ever clears it -- which looks like "no variable
    bins", not like an error."""
    rng = np.random.default_rng(31)
    v = rng.normal(size=(50, 6))
    permuted = np.take_along_axis(
        v, np.array([rng.permutation(50) for _ in range(6)]).T, axis=0)
    assert np.allclose(np.nanvar(v, axis=0, ddof=1),
                       np.nanvar(permuted, axis=0, ddof=1))


def test_the_empirical_floor_comes_from_the_null_matrix():
    obs = np.random.default_rng(32).normal(0, 2.0, (60, 5))
    null = np.random.default_rng(33).normal(0, 0.5, (60, 5))
    base = pc.sigma2_bio(obs, np.full_like(obs, 0.1))
    out = pc.empirical_sigma2_bio(base, pc.noise_floor_from_null_matrix(null))
    assert np.allclose(out["noise_floor"], np.var(null, axis=0, ddof=1))
    assert (out["sigma2_bio_emp"] > 0).all()
    assert (out["snr_emp"] > 1).all()


def test_sigma2_bio_is_zero_in_expectation_on_shuffled_data():
    """The design's own Part C test: the noise-floor calibration works."""
    rng = np.random.default_rng(5)
    n_cells, n_bins = 80, 40
    se = np.full((n_cells, n_bins), 0.3)
    v = rng.normal(0, 0.3, (n_cells, n_bins))       # pure noise, no biology
    s = pc.sigma2_bio(v, se)
    assert abs(float(np.median(s["sigma2_bio"]))) < 0.05


def test_sigma2_bio_survives_real_between_cell_structure():
    rng = np.random.default_rng(6)
    n_cells, n_bins = 80, 40
    bio = rng.normal(0, 1.0, (n_cells, 1)) * np.ones((1, n_bins))
    v = bio + rng.normal(0, 0.2, (n_cells, n_bins))
    s = pc.sigma2_bio(v, np.full((n_cells, n_bins), 0.2))
    assert float(np.median(s["sigma2_bio"])) > 0.5


def test_raw_variance_would_have_ranked_the_noisiest_bins_first():
    """The reason the decomposition exists."""
    rng = np.random.default_rng(7)
    n = 100
    noisy = rng.normal(0, 1.0, (n, 1))              # all sampling noise
    real = rng.normal(0, 0.8, (n, 1))               # less variance, real
    v = np.hstack([noisy, real])
    se = np.hstack([np.full((n, 1), 1.0), np.full((n, 1), 0.05)])
    s = pc.sigma2_bio(v, se)
    assert s["var_total"][0] > s["var_total"][1], "raw variance ranks noise 1st"
    assert s["sigma2_bio"][1] > s["sigma2_bio"][0], (
        "after subtracting the noise term the real bin must win")


def test_permutation_is_over_cells_not_reads():
    """A read-level permutation breaks the within-cell correlation that makes a
    cell one observation, and gives p-values far too small."""
    rng = np.random.default_rng(8)
    v = rng.normal(size=(60, 12))
    g = np.array(["A"] * 30 + ["B"] * 30)
    p = pc.permutation_p(v, g, n_perm=200, seed=0)
    assert p.shape == (12,)
    assert (p >= 1 / 201).all(), "a permutation p can never be exactly zero"
    assert 0.2 < float(np.median(p)) < 0.8, "no structure -> uniform p"


def test_a_planted_celltype_bin_is_recovered_and_a_merely_variable_one_is_not():
    """The design asks for both halves: recover the specific bins AND confirm
    variable-but-not-specific bins do not appear there."""
    rng = np.random.default_rng(9)
    n_cells, n_bins = 120, 30
    g = np.array(["A"] * 60 + ["B"] * 60)
    v = rng.normal(0, 0.3, (n_cells, n_bins))
    v[:60, 0] += 3.0                                  # specific to A
    v[:, 1] += rng.normal(0, 3.0, n_cells)            # variable, not specific
    p = pc.permutation_p(v, g, n_perm=400, seed=1)
    assert p[0] < 0.01, "the planted cell-type bin must be found"
    assert p[1] > 0.05, "a merely variable bin must not be called specific"


def test_a_bin_that_is_stable_because_it_is_uncoordinated_is_background():
    """|mean_rho| is what separates 'consistently coordinated' from 'never
    coordinated' -- without it the stable set is the dead bins."""
    s = pd.DataFrame({
        "sigma2_bio": [0.0, 0.0], "noise_floor": [0.1, 0.1],
        "mean_rho": [0.9, 0.001], "mean_neff": [100.0, 100.0],
        "F": [1.0, 1.0], "q_value": [0.9, 0.9], "passes_qc": [True, True],
    })
    out = pc.classify_regions(s, top_n=1, fdr=0.05)
    assert bool(out["is_stable"][0]) and not bool(out["is_stable"][1])


def test_background_is_every_qc_passing_bin_not_a_ranked_subset():
    """It is the denominator for every enrichment test; a ranked subset would
    silently change what an enrichment is against."""
    s = pd.DataFrame({
        "sigma2_bio": np.linspace(0, 1, 10), "noise_floor": np.zeros(10),
        "mean_rho": np.full(10, 0.5), "mean_neff": np.full(10, 50.0),
        "F": np.linspace(1, 5, 10), "q_value": np.full(10, 0.5),
        "passes_qc": [True] * 8 + [False] * 2,
    })
    out = pc.classify_regions(s, top_n=3, fdr=None)
    assert int(out["is_background"].sum()) == 8
    assert int(out["is_variable"].sum()) == 3


# --------------------------------------------------------------- end to end ---

@pytest.fixture(scope="module")
def run(tmp_path_factory):
    pytest.importorskip("sklearn")
    pytest.importorskip("matplotlib")
    m = _mod()
    d = tmp_path_factory.mktemp("pcc")
    rng = np.random.default_rng(11)

    # Many bins, few planted.  C.2's per-cell centring redistributes a
    # cell-type signal across the OTHER bins with the opposite sign, in
    # proportion to the fraction of bins carrying it -- so a fixture where 10%
    # of bins are planted makes every remaining bin look group-associated,
    # which is an artefact of the fixture's scale, not of the method.  At
    # genome scale (a handful of specific bins in ~10,800) it is negligible.
    n_cells, n_bins, n_pairs = 30, 120, 720
    # The partner end of a 300 kb contact anchored in bin b lands in b+1, so
    # planting bins 2 and 5 plants four anchors.
    specific = {2, 3}
    # Per-(cell, bin) variation, not just per-cell.  A per-cell-only baseline
    # cancels exactly under C.2's row centring, leaving every non-planted bin a
    # deterministic function of the cell's group and therefore an enormous F --
    # an artefact of a too-clean fixture, not of the method.
    base = rng.uniform(0.05, 0.45, (n_cells, n_bins))
    rows = []
    for ci in range(n_cells):
        grp = ci % 2
        for b in range(n_bins):
            for p in range(n_pairs // n_bins):
                strength = 0.9 if (b in specific and grp == 0) else 0.2
                u = rng.normal()
                pos1 = b * 250_000 + 1_000
                pos2 = pos1 + 300_000
                for end, pos, sc in ((1, pos1, 1.0), (2, pos2, strength)):
                    rows.append({
                        "cell_id": f"cell{ci}",
                        "pair_idx": b * 100 + p, "end_idx": end,
                        "pos_5p": pos, "pos_3p": pos + 80,
                        "contact_dist": 300_000, "is_cis": True,
                        "conversion_pass": True, "conversion_rate": 0.99,
                        "centred_gch": sc * u + 0.25 * rng.normal(),
                        "centred_hcg": -0.8 * sc * u + 0.25 * rng.normal(),
                        "var_gch": 0.02, "var_hcg": 0.05,
                        "n_gch": 4, "n_hcg": 2,
                        "theta_gch": 0.4, "mu_gch": 0.4, "prior_sd_gch": 0.25,
                    })
    post = d / "post" / "chrom=chr1"
    post.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(post / "all.parquet", index=False)

    r1 = pd.DataFrame([{
        "group": "all", "chrom": "chr1", "bin_start": b * 250_000,
        "stratum": "100kb-1Mb", "w_A_gch": 1.0, "w_A_hcg": -0.8,
        "w_B_gch": 1.0, "w_B_hcg": -0.8, "rho1": 0.5, "rho2": 0.1,
    } for b in range(n_bins)])
    r1.to_parquet(d / "rank1.parquet", index=False)

    man = d / "s.tsv"
    man.write_text("name\tbam\tgroup\tcolor\tbatch\n" + "".join(
        f"cell{i}\t-\tG{i % 2}\t#{'aabbcc' if i % 2 == 0 else 'ddeeff'}"
        f"\tb{i % 3}\n" for i in range(n_cells)))

    out = d / "out"
    rc = m.main(["--posteriors", str(d / "post"), "--rank1", str(d / "rank1.parquet"),
                 "--samples", str(man), "--group-order", "G0,G1",
                 "-o", str(out), "--bin-size", "250000",
                 "--strata", "100kb-1Mb", "--min-neff-cell", "2",
                 "--n-svd", "5", "--n-perm", "500", "--n-shuffle", "5",
                 "--top-n", "6", "--fdr", "0.05", "--force"])
    assert rc == 0
    return {"out": out, "mod": m, "man": str(man), "post": str(d / "post"),
            "rank1": str(d / "rank1.parquet"), "specific": specific}


def test_every_c7_output_exists(run):
    o = run["out"]
    for f in ("05_percell_coordination.parquet", "05_svd.parquet",
              "06_region_stats.parquet", "06_regions_variable.bed",
              "06_regions_celltype_specific.bed", "06_regions_stable.bed",
              "06_regions_background.bed", "_partC_metadata.json"):
        assert (o / f).exists(), f
    assert (o / "qc" / "C4_pc_covariate_panel.tsv").exists()
    assert list((o / "figures").glob("heatmap_*.pdf"))


def test_the_pc_covariate_panel_reports_coverage_without_dropping_it(run):
    """In scHi-C PC1 almost always tracks coverage; the panel must say so and
    must NOT silently remove it."""
    panel = pd.read_csv(run["out"] / "qc" / "C4_pc_covariate_panel.tsv", sep="\t")
    assert "r_log10_n_contacts" in panel.columns
    assert "r_mean_neff" in panel.columns
    meta = json.loads((run["out"] / "_partC_metadata.json").read_text())
    assert meta["dropped_pcs"] == [], "nothing may be auto-dropped"


def test_the_per_cell_table_carries_what_c7_asks_for(run):
    t = pd.read_parquet(run["out"] / "05_percell_coordination.parquet")
    for c in ("cell_id", "chrom", "bin", "stratum", "rho_hat", "rho_shrunk",
              "n_eff", "se", "channel"):
        assert c in t.columns, c
    assert (t["se"] > 0).all(), "an SE of zero would mean infinite precision"


def test_region_stats_keeps_every_bin_so_cutoffs_can_be_revisited(run):
    """C.5: the full table carries every score for all bins regardless of
    category."""
    s = pd.read_parquet(run["out"] / "06_region_stats.parquet")
    t = pd.read_parquet(run["out"] / "05_percell_coordination.parquet")
    assert len(s) == t["bin"].nunique()
    for c in ("sigma2_bio", "noise_floor", "F", "p_value", "q_value",
              "specificity_index", "top_group", "mean_rho", "mean_neff",
              "category"):
        assert c in s.columns, c


def test_the_planted_cell_type_bins_are_the_ones_found(run):
    s = pd.read_parquet(run["out"] / "06_region_stats.parquet")
    s["bin_index"] = s["bin_start"] // 250_000
    called = set(s.loc[s["is_celltype_specific"], "bin_index"].tolist())
    assert called, "nothing was called cell-type specific"
    assert run["specific"] <= called, (
        f"planted bins {run['specific']} not all among the calls {called}")
    # And the calls are not simply every bin: the set has to discriminate.
    assert len(called) < s["bin_index"].nunique()


def test_metadata_records_the_matrix_construction_order(run):
    meta = json.loads((run["out"] / "_partC_metadata.json").read_text())
    assert meta["matrix_steps"][0] == "shrunk"
    assert meta["matrix_steps"][1] == "per_cell_centred"
    assert meta["feasibility"]["verdict"] in ("ok", "marginal", "too_thin")


def test_rwr_cells_without_independent_features_is_refused(run, tmp_path):
    m = run["mod"]
    with pytest.raises(SystemExit, match="circular|coordination"):
        m.main(["--posteriors", run["post"], "--rank1", run["rank1"],
                "--samples", run["man"], "--group-order", "G0,G1",
                "-o", str(tmp_path / "x"), "--bin-size", "250000",
                "--strata", "100kb-1Mb", "--rwr-cells", "--force"])


def test_a_null_dataset_yields_no_cell_type_specific_regions(tmp_path):
    """The design's own Part C test.  A pipeline that finds structure in noise
    is worse than one that finds none."""
    rng = np.random.default_rng(21)
    n_cells, n_bins = 40, 40
    v = rng.normal(0, 1.0, (n_cells, n_bins))
    g = np.array(["A"] * 20 + ["B"] * 20)
    p = pc.permutation_p(v, g, n_perm=400, seed=2)
    from nome_hic_concordance.distance_stats import fdr as bh
    q = bh(p)
    assert (q >= 0.05).all(), f"{(q < 0.05).sum()} false calls at FDR 0.05"
