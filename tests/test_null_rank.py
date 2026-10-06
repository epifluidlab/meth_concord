"""Ranking each bin against a null drawn from its own ends.

C and the log odds ratio already subtract the bin's marginal -- the expected
pool is that bin's own read ends, re-paired -- but they cannot make bins
COMPARABLE.  Shrinkage varies with depth (at alpha_0 ~ 100 a bin at n = 10 is
91% prior and one at n = 10,000 is 1%), and the log odds ratio's sensitivity
to a surplus co-open contact is 1/(alpha_open + k_open), roughly five times
larger where k_open -> 0 than in euchromatin on a measured GCH run.  So a
fixed value means different amounts of evidence in a PMD and in an active
domain, which is why heterochromatin reads co-open against any genome-wide
reference: not a displaced zero, an incomparable unit.

The rank fixes the unit by comparing each bin to nulls DRAWN FROM ITS OWN
ENDS, pooled within a depth-by-openness stratum.  What has to be true for
that to be worth reading is checked here: it is calibrated on pure null, it
has power that follows depth rather than ignoring it, and it does not claim
a direction where only a magnitude was planted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance import concordance as cc


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def null_rank_paths(tmp_path, n_cells=12, n_bins=60, bin_size=10_000,
                    seed=0, planted=(), coord=0.75, direction=None):
    """Bins that differ in DEPTH and in MARGINAL openness, null except in
    ``planted``.

    Those two axes are what ``rank_strata`` is built on, so a fixture holding
    them constant could not show whether stratification matters.  Bins cycle
    through four depths and four marginals.

    Unplanted: a pair's two ends are drawn INDEPENDENTLY at the bin's
    marginal, so the truth is C = 0 and L = 0 at every depth and marginal.

    Planted with ``direction=None``: both ends take the SAME state at rate
    ``coord``.  That is pure coordination -- the marginal is untouched, so C
    moves and the DIRECTION does not.

    Planted with ``direction=p``: the concordant pairs are co-open at rate
    ``p`` instead of at the bin's marginal, which moves the direction.
    """
    rng = np.random.default_rng(seed)
    depths = [6, 14, 40, 120]
    margs = [0.12, 0.30, 0.55, 0.80]
    paths = {}
    for c in range(n_cells):
        rec = []
        for b in range(n_bins):
            n_pairs = depths[b % len(depths)]
            pm = margs[(b // len(depths)) % len(margs)]
            base = b * bin_size
            for i in range(n_pairs):
                if b in planted and rng.random() < coord:
                    a = ((rng.random() < direction) if direction is not None
                         else (rng.random() < pm))
                    bb = a
                else:
                    a = rng.random() < pm
                    bb = rng.random() < pm
                p1 = 0.9 if a else 0.1
                p2 = 0.9 if bb else 0.1
                o1 = 200 + (i * 13) % 3000
                o2 = o1 + 1000 + (i * 37) % 5000
                rec.append({"chrom": "chr1", "pos1": base + o1,
                            "pos2": base + o2, "n1": 4, "n2": 4,
                            "posterior1": p1, "posterior2": p2,
                            "distance": o2 - o1})
                # A short-range pair so the legacy unlinked pool is populated.
                # Its far end must be an INDEPENDENT draw at the bin's
                # marginal: giving it 1 - p1 would put an anti-correlated
                # partner for every end into the pool E is re-paired from,
                # which inflates E's mixed mass and makes O - E positive
                # everywhere on what is supposed to be a null fixture.  That
                # measured as a per-stratum median C of +0.004 to +0.057 and
                # a 0.163 false-positive rate before it was fixed.
                p3 = 0.9 if (rng.random() < pm) else 0.1
                rec.append({"chrom": "chr1", "pos1": base + o1,
                            "pos2": base + o1 + 40, "n1": 4, "n2": 4,
                            "posterior1": p1, "posterior2": p3,
                            "distance": 40})
        df = pd.DataFrame(rec)
        for col in ("pos1", "pos2", "distance"):
            df[col] = df[col].astype(np.int64)
        for col in ("n1", "n2"):
            df[col] = df[col].astype(np.int32)
        path = tmp_path / f"c{c}.posteriors.parquet"
        df.to_parquet(path)
        paths[f"c{c}"] = str(path)
    return paths


def _run(tmp_path, planted=(), direction=None, n_draws=8, scope="cell",
         coord=0.75, **kw):
    src = tmp_path / f"src_{len(planted)}_{direction}_{scope}"
    src.mkdir(parents=True, exist_ok=True)
    paths = null_rank_paths(src, planted=set(planted), direction=direction,
                            coord=coord)
    return cc.compute_signed_concordance_matrix(
        paths, bin_size=10_000, min_distance=500, min_gch_sites=2,
        baseline="distance-matched", dm_dist_bin=500, dm_prior="shared",
        output_dir=str(tmp_path / "out"), save_outputs=False, use_rwr=False,
        n_threads=1, eb_min_obs=50, null_draws=n_draws, null_rank=True,
        null_rank_scope=scope, null_rank_min_pool=200, **kw)


P05 = -np.log10(0.05)


def _rate(arr, thr=P05):
    a = np.asarray(arr, float)
    a = a[np.isfinite(a)]
    return float(np.mean(np.abs(a) > thr)) if a.size else float("nan")


# ---------------------------------------------------------------------------
# The pieces, in isolation
# ---------------------------------------------------------------------------


def test_null_keys_leave_the_first_draw_unindexed():
    """A single-draw run has to write exactly what it wrote before this
    option existed, or every consumer of concordance_*_null.parquet breaks."""
    assert cc._null_keys(0) == ("k_null_open", "k_null_closed",
                                "k_null_mixed", "n_null")
    assert cc._null_keys(2) == ("k_null2_open", "k_null2_closed",
                                "k_null2_mixed", "n_null2")
    assert cc._NULL_KEYS == cc._null_keys(0)
    # Every draw's keys are distinct, or they would overwrite each other in
    # the count-matrix dict.
    seen = [k for d in range(6) for k in cc._null_keys(d)]
    assert len(set(seen)) == len(seen)


def test_empirical_p_is_conservative_and_bounded_by_the_pool():
    """A value beyond every draw cannot report a p finer than the pool can
    express, so the +1 correction is the floor and not a rounding detail."""
    pool = np.linspace(-1, 1, 199)
    out = cc._signed_neglog10_p(np.array([5.0, -5.0, 0.0]), pool)
    finest = -np.log10(2.0 / (pool.size + 1))
    assert out[0] == pytest.approx(finest)
    assert out[1] == pytest.approx(-finest)     # same magnitude, other tail
    assert abs(out[2]) < 0.35                   # dead centre -> p near 1


def test_strata_separate_depth_and_openness():
    n = np.array([[2.0, 2.0, 500.0, 500.0]])
    o = np.array([[0.05, 0.9, 0.05, 0.9]])
    s = cc.rank_strata(n, o)
    assert len(set(s.ravel().tolist())) == 4


def test_stratifying_is_what_makes_the_pool_usable():
    """Bins whose null is 20x wider poison a shared pool: the wide ones
    supply the tail and nothing in the narrow class can clear it.

    This is the failure the fixed bands exist to prevent, and it is the same
    one the marker script's permutation null hit (power 0.00 unstratified).
    """
    rng = np.random.default_rng(0)
    n_cells, n_bins, K = 8, 600, 5
    deep = np.arange(n_bins) < 300
    n_raw = np.where(deep, 200.0, 2.0)[None, :].repeat(n_cells, 0)
    openness = np.where(deep, 0.55, 0.12)[None, :].repeat(n_cells, 0)
    sd = np.where(deep, 0.02, 0.4)[None, :].repeat(n_cells, 0)
    draws = [rng.normal(0, sd) for _ in range(K)]
    obs = rng.normal(0, sd)
    obs[:, :10] += 6 * 0.02                      # real, in the narrow class
    keep = np.ones_like(obs, bool)

    st = cc.rank_strata(n_raw, openness)
    r, _ = cc.empirical_rank(obs, draws, st, keep, min_pool=1000)
    flat, _ = cc.empirical_rank(obs, draws, np.zeros_like(st), keep,
                                scope="joint")

    assert _rate(r[:, :10]) > 0.9                # planted, stratified
    assert _rate(flat[:, :10]) < 0.1             # planted, pooled flat
    assert _rate(r[:, 10:300]) < 0.10            # deep nulls stay nominal
    assert _rate(r[:, 300:]) < 0.10              # thin nulls too
    assert _rate(flat[:, 300:]) > _rate(r[:, 300:])


def test_cell_scope_removes_a_per_cell_offset_from_the_score():
    """A global shift -- batch, purity, enzyme efficiency -- must not decide
    which cells look significant.

    The drawn null carries the offset too, so BOTH scopes stay calibrated on
    average.  What separates them is per-cell: under 'joint' the cells at the
    extremes of the offset carry systematically larger scores, and any
    downstream group test then reads the offset as biology.
    """
    rng = np.random.default_rng(1)
    n_cells, n_bins, K = 10, 800, 5
    n_raw = np.full((n_cells, n_bins), 200.0)
    openness = np.full((n_cells, n_bins), 0.5)
    st = cc.rank_strata(n_raw, openness)
    offset = np.linspace(-0.5, 0.5, n_cells)[:, None]
    draws = [rng.normal(0, 0.1, (n_cells, n_bins)) + offset for _ in range(K)]
    obs = rng.normal(0, 0.1, (n_cells, n_bins)) + offset
    keep = np.ones_like(obs, bool)

    spread = {}
    for scope in ("cell", "joint"):
        r, _ = cc.empirical_rank(obs, draws, st, keep, scope=scope,
                                 min_pool=1000)
        assert _rate(r) < 0.10
        spread[scope] = float(np.ptp(np.nanmean(np.abs(r), axis=1)))
    assert spread["cell"] < 0.25 * spread["joint"]


def test_thin_pools_fall_back_to_the_joint_pool_rather_than_guessing():
    rng = np.random.default_rng(2)
    obs = rng.normal(0, 1, (6, 40))
    draws = [rng.normal(0, 1, (6, 40)) for _ in range(3)]
    st = np.zeros_like(obs, dtype=np.int32)
    keep = np.ones_like(obs, bool)
    r, info = cc.empirical_rank(obs, draws, st, keep, scope="cell",
                                min_pool=10_000)
    assert info["n_fallback"] == obs.size
    assert info["median_pool"] == 6 * 40 * 3
    assert np.isfinite(r).all()


def test_untestable_entries_are_dropped_from_the_output_and_the_pool():
    rng = np.random.default_rng(3)
    obs = rng.normal(0, 1, (4, 50))
    draws = [rng.normal(0, 1, (4, 50)) for _ in range(4)]
    st = np.zeros_like(obs, dtype=np.int32)
    keep = np.ones_like(obs, bool)
    keep[:, :20] = False
    r, info = cc.empirical_rank(obs, draws, st, keep, scope="joint")
    assert np.isnan(r[:, :20]).all()
    assert np.isfinite(r[:, 20:]).all()
    # The masked entries are absent from the pool as well, not merely from
    # the output -- otherwise the observed is judged against draws from bins
    # it was not allowed to be compared with.
    assert info["median_pool"] == 4 * 30 * 4


def test_self_calibration_needs_two_draws():
    st = np.zeros((3, 10), dtype=np.int32)
    keep = np.ones((3, 10), bool)
    assert cc.rank_self_calibration([np.zeros((3, 10))], st, keep) is None


# ---------------------------------------------------------------------------
# End to end, through the real pipeline
# ---------------------------------------------------------------------------


def test_pure_null_run_is_calibrated_against_itself_and_conservative(tmp_path):
    """On a fixture with no coordination anywhere: holding out one drawn null
    and scoring it must return roughly uniform p, and the OBSERVED bins must
    not clear more often than nominal."""
    res = _run(tmp_path)
    cal = res["null_rank_calibration"]
    assert set(cal) == {"coordination", "direction_logor"}
    for name, c in cal.items():
        assert c["n"] > 500
        assert c["frac_p05"] < 0.10, (name, c)
        assert c["frac_p01"] < 0.04, (name, c)
        # Every stratum pool must be able to express a p finer than the 0.05
        # the run is judged at, or the check tests nothing.
        assert c["finest_p"] < 0.05
    for name, df in res["null_rank"].items():
        assert _rate(df.to_numpy(float)) < 0.05, name


def test_power_follows_depth_instead_of_ignoring_it(tmp_path):
    """A 6-pair bin cannot support a call and the rank says so; a 120-pair
    bin can.  The raw effect size makes no such distinction -- a thin bin's C
    is mostly prior but still reads as a number."""
    planted = set(range(0, 60, 7))
    res = _run(tmp_path, planted=planted)
    a = res["null_rank"]["coordination"].to_numpy(float)
    idx = np.array([int(b.split(":")[1]) // 10_000
                    for b in res["coordination"].columns])
    is_pl = np.isin(idx, list(planted))
    depth = np.array([[6, 14, 40, 120][b % 4] for b in idx])

    by_depth = {d: _rate(a[:, is_pl & (depth == d)]) for d in (6, 40, 120)}
    assert by_depth[6] < 0.2
    assert by_depth[40] > 0.6
    assert by_depth[120] > 0.6
    assert by_depth[6] < by_depth[40] <= by_depth[120] + 0.15
    # Calls on unplanted bins stay rare, and calls on planted ones point the
    # right way: coordination was ADDED.
    assert _rate(a[:, ~is_pl]) < 0.05
    hit = np.abs(a[:, is_pl]) > P05
    assert float(np.mean(a[:, is_pl][hit] > 0)) > 0.95


def test_pure_coordination_moves_c_and_not_the_direction(tmp_path):
    """The planted signal preserves the marginal, so there is no direction to
    find.  A rank that reported one would be reading its own noise."""
    planted = set(range(0, 60, 7))
    res = _run(tmp_path, planted=planted)
    idx = np.array([int(b.split(":")[1]) // 10_000
                    for b in res["coordination"].columns])
    is_pl = np.isin(idx, list(planted))
    assert _rate(res["null_rank"]["coordination"].to_numpy(float)[:, is_pl]) > 0.3
    assert _rate(res["null_rank"]["direction_logor"].to_numpy(float)[:, is_pl]) < 0.05


def test_the_log_odds_ratio_reads_co_open_from_pure_coordination(tmp_path):
    """L IS NOT AN INDEPENDENT DIRECTION AXIS, and this is why PMDs look
    co-open.

    Coordination moves mass out of `mixed` into BOTH concordant cells by the
    same amount, so it raises S = O_open + O_closed while leaving
    D = O_open - O_closed pinned at the marginal by the exact per-pair
    identity D == pbar_1 + pbar_2 - 1.  With the same D on both sides,

        L = log[(S_O + D)/(S_O - D)] - log[(S_E + D)/(S_E - D)]

    which is antisymmetric about pbar = 0.5 and zero there.  So adding pure
    coordination -- no directional preference whatsoever -- drives L positive
    wherever the marginal is low and negative wherever it is high.
    Heterochromatin is the low-marginal end of the genome, so any
    coordination in it reads as "co-open".

    The rank cannot repair this: the drawn null has C ~ 0, so its L sits at
    0, and the observed bin is correctly reported as displaced.  What the
    rank does fix is the SCALE -- see the correlation check below, which is
    what tells a run that its direction call is really its magnitude call.
    """
    def L(pbar, C):
        D = 2 * pbar - 1
        S_E = pbar ** 2 + (1 - pbar) ** 2
        f = lambda S: np.log((S + D) / 2) - np.log((S - D) / 2)
        return f(S_E + C) - f(S_E)

    assert L(0.10, 0.10) == pytest.approx(+1.7319, abs=1e-3)
    assert L(0.90, 0.10) == pytest.approx(-1.7319, abs=1e-3)
    assert L(0.50, 0.10) == pytest.approx(0.0, abs=1e-12)
    # Monotone in the marginal at fixed, positive C: every low-marginal bin
    # is pushed the same way, which is what makes it a domain-scale artifact
    # rather than scattered noise.
    vals = [L(p, 0.05) for p in (0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85)]
    assert vals == sorted(vals, reverse=True)
    assert vals[0] > 0 > vals[-1]


def test_a_run_reports_how_much_of_its_direction_call_is_magnitude(tmp_path,
                                                                   caplog):
    """Given the identity above, a run has to say whether its L rank is
    tracking its C rank -- otherwise a reader takes coordination for
    direction."""
    with caplog.at_level("INFO"):
        _run(tmp_path, planted=set(range(0, 60, 7)))
    msgs = [r.getMessage() for r in caplog.records]
    assert any("corr(coordination, direction_logor)" in m for m in msgs)


def test_one_draw_still_runs_and_writes_no_calibration(tmp_path):
    res = _run(tmp_path, n_draws=1)
    assert res["null_rank"] is not None
    assert res["null_rank_calibration"] is None


def test_rank_is_emitted_alongside_the_effect_sizes_not_instead(tmp_path):
    """A rank cannot be pooled across a region the way counts can, so the
    region tests and --from-counts profiles still need C and L."""
    out = tmp_path / "out"
    src = tmp_path / "src"
    src.mkdir(parents=True)
    cc.compute_signed_concordance_matrix(
        null_rank_paths(src, n_cells=6, n_bins=24),
        bin_size=10_000, min_distance=500, min_gch_sites=2,
        baseline="distance-matched", dm_dist_bin=500, dm_prior="shared",
        output_dir=str(out), save_outputs=True, use_rwr=False, n_threads=1,
        eb_min_obs=50, null_draws=3, null_rank=True, null_rank_min_pool=100,
    )
    names = {p.name for p in out.glob("*.parquet")}
    for stat in ("coordination", "direction_logor"):
        assert f"concordance_{stat}.parquet" in names
        assert f"concordance_{stat}_rank.parquet" in names
        assert f"concordance_{stat}_null.parquet" in names
    assert (out / "null_rank_calibration.json").exists()
    r = pd.read_parquet(out / "concordance_coordination_rank.parquet")
    c = pd.read_parquet(out / "concordance_coordination.parquet")
    assert list(r.index) == list(c.index)
    assert list(r.columns) == list(c.columns)


def test_null_rank_implies_a_drawn_null(tmp_path):
    """Asking for the rank without --emit-null is never what a caller meant,
    so the library turns it on rather than silently writing nothing."""
    src = tmp_path / "src"
    src.mkdir(parents=True)
    res = cc.compute_signed_concordance_matrix(
        null_rank_paths(src, n_cells=6, n_bins=24),
        bin_size=10_000, min_distance=500, min_gch_sites=2,
        baseline="distance-matched", dm_dist_bin=500, dm_prior="shared",
        output_dir=str(tmp_path / "o"), save_outputs=False, use_rwr=False,
        n_threads=1, eb_min_obs=50, null_draws=3, null_rank=True,
        emit_null=False, null_rank_min_pool=100,
    )
    assert res["null_rank"] is not None


def test_short_range_baseline_warns_instead_of_pretending(tmp_path, caplog):
    """There is no non-contact pool to draw from, so there is no rank."""
    src = tmp_path / "src"
    src.mkdir(parents=True)
    with caplog.at_level("WARNING"):
        res = cc.compute_signed_concordance_matrix(
            null_rank_paths(src, n_cells=4, n_bins=12),
            bin_size=10_000, min_distance=500, min_gch_sites=2,
            baseline="short-range", output_dir=str(tmp_path / "o"),
            save_outputs=False, use_rwr=False, n_threads=1, eb_min_obs=50,
            null_draws=4, null_rank=True,
        )
    assert res["null_rank"] is None
    assert any("no drawn null is available" in r.getMessage()
               for r in caplog.records)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_every_null_flag_reaches_step_five():
    """A flag that parses but is never passed on is the worst kind of bug
    here: the run looks configured and silently uses the default.  Pin the
    parser option and the keyword together."""
    import inspect
    import re
    from nome_hic_concordance import cli

    src = inspect.getsource(cli)
    params = set(inspect.signature(
        cc.compute_signed_concordance_matrix).parameters)
    flags = set(re.findall(r'"--(null-[a-z-]+)"', src))
    assert flags == {"null-draws", "null-graph", "null-rank", "null-z",
                     "null-rank-scope", "null-rank-statistics",
                     "null-rank-min-pairs", "null-rank-min-pool"}
    for flag in flags:
        kw = flag.replace("-", "_")
        assert kw in params, f"--{flag} has no {kw} parameter on step 5"
        # Passed on at BOTH call sites (the plain run and the sweep branch).
        assert len(re.findall(rf"\b{kw}=", src)) >= 2, flag


def test_defaults_leave_the_pipeline_where_it_was():
    """--null-rank is opt-in: the parser's defaults must not turn on a
    K-draw null for every existing run."""
    import inspect
    d = inspect.signature(cc.compute_signed_concordance_matrix).parameters
    assert d["null_rank"].default is False
    assert d["null_draws"].default == 1
    assert d["emit_null"].default is False


def test_the_self_check_splits_its_false_positives_by_sign():
    """A procedure can be calibrated in |score| and lopsided in sign, and for
    the log odds ratio that is the failure mode to watch: coordination at a
    low marginal pushes L one way, so a null carrying any of it calls
    co-open far more often than co-closed."""
    rng = np.random.default_rng(4)
    n_cells, n_bins, K = 6, 400, 6
    st = np.zeros((n_cells, n_bins), dtype=np.int32)
    keep = np.ones((n_cells, n_bins), bool)

    sym = [rng.normal(0, 1, (n_cells, n_bins)) for _ in range(K)]
    cal = cc.rank_self_calibration(sym, st, keep, min_pool=500)
    assert cal["frac_p05_pos"] + cal["frac_p05_neg"] == pytest.approx(
        cal["frac_p05"], abs=1e-12)
    assert abs(cal["frac_p05_pos"] - cal["frac_p05_neg"]) < 0.02

    # A held-out draw that is displaced relative to its siblings: the total
    # rate can stay near nominal while every call lands on one side.
    skew = [rng.normal(0, 1, (n_cells, n_bins)) for _ in range(K)]
    skew[0] = skew[0] + 1.4
    bad = cc.rank_self_calibration(skew, st, keep, min_pool=500)
    assert bad["frac_p05_pos"] > 8 * max(bad["frac_p05_neg"], 1e-4)


def test_a_lopsided_rank_is_called_out(tmp_path, caplog):
    with caplog.at_level("INFO"):
        _run(tmp_path, planted=set(range(0, 60, 7)))
    msgs = [r.getMessage() for r in caplog.records]
    line = [m for m in msgs if "self-check" in m]
    assert line, msgs
    # This fixture's null is symmetric, so nothing should be flagged.
    assert not any("LOPSIDED" in m for m in msgs)


def test_the_calibration_json_carries_a_tail_resolved_quantile_sketch():
    """A Q-Q against uniform lives in the small-p tail; 41 evenly spaced
    levels stop at p = 0.025, coarser than any threshold worth FDR-ing at."""
    lv = cc._Q_LEVELS
    assert lv.min() <= 1e-4
    assert lv.max() == pytest.approx(1.0)
    assert np.all(np.diff(lv) > 0)
    # At least a third of the levels below 0.05, or the tail is unresolved.
    assert float(np.mean(lv < 0.05)) > 0.3


# ---------------------------------------------------------------------------
# --null-z: the same referencing, as an effect size instead of evidence
# ---------------------------------------------------------------------------


def test_z_is_calibrated_on_a_pure_null():
    """Called z, so 2 has to mean two sd.  In particular the numerator
    carries the draw-mean's own variance -- obs - mean_d has variance
    s^2 (1 + 1/K) -- and an uncorrected version comes back wide."""
    rng = np.random.default_rng(7)
    n_cells, n_bins, K = 8, 900, 10
    st = np.zeros((n_cells, n_bins), dtype=np.int32)
    keep = np.ones((n_cells, n_bins), bool)
    draws = [rng.normal(0, 1, (n_cells, n_bins)) for _ in range(K)]
    obs = rng.normal(0, 1, (n_cells, n_bins))
    z, info = cc.null_z_matrix(obs, draws, st, keep)
    v = z[np.isfinite(z)]
    assert abs(float(np.mean(v))) < 0.05
    assert abs(float(np.std(v)) - 1.0) < 0.08, float(np.std(v))
    assert info["n_draws"] == K


def test_z_tracks_the_effect_where_the_rank_saturates():
    """The rank cannot exceed -log10(2/(pool+1)), so it flattens exactly the
    bins that are most clearly real; the z has no ceiling.  That is the
    difference that matters for a volcano axis."""
    rng = np.random.default_rng(8)
    n_cells, n_bins, K = 8, 600, 10
    st = np.zeros((n_cells, n_bins), dtype=np.int32)
    keep = np.ones((n_cells, n_bins), bool)
    draws = [rng.normal(0, 1, (n_cells, n_bins)) for _ in range(K)]
    obs = rng.normal(0, 1, (n_cells, n_bins))
    # Both past the pool's resolution floor -- which is the regime the rank
    # cannot describe and a volcano axis lives in.
    obs[:, :20] += 6.0
    obs[:, 20:40] += 18.0                  # three times as large again
    z, _ = cc.null_z_matrix(obs, draws, st, keep)
    r, _ = cc.empirical_rank(obs, draws, st, keep, scope="joint")

    z_mid, z_big = np.nanmean(z[:, :20]), np.nanmean(z[:, 20:40])
    r_mid, r_big = np.nanmean(r[:, :20]), np.nanmean(r[:, 20:40])
    # z separates the two effect sizes; the rank has run out of room.
    assert z_big > 2.5 * z_mid
    assert r_big < 1.05 * r_mid
    ceiling = -np.log10(2.0 / (n_cells * n_bins * K + 1))
    assert np.nanmax(np.abs(r)) <= ceiling + 1e-9
    assert np.nanmax(np.abs(z)) > ceiling


def test_a_bin_whose_draws_collapsed_borrows_its_stratum_instead_of_exploding():
    """Across millions of cell-bins some sets of draws agree by chance.  An
    unmoderated sd goes to zero there and z goes to infinity."""
    rng = np.random.default_rng(9)
    n_cells, n_bins, K = 6, 400, 10
    st = np.zeros((n_cells, n_bins), dtype=np.int32)
    keep = np.ones((n_cells, n_bins), bool)
    draws = [rng.normal(0, 1, (n_cells, n_bins)) for _ in range(K)]
    for d in draws:                        # bin 0: every draw identical
        d[:, 0] = 0.0
    obs = rng.normal(0, 1, (n_cells, n_bins))
    obs[:, 0] = 0.5
    z, info = cc.null_z_matrix(obs, draws, st, keep)
    assert np.isfinite(z[:, 0]).all()
    assert np.abs(z[:, 0]).max() < 5.0
    assert 0.0 < info["frac_own_variance"] <= 1.0


def test_moderation_leaves_an_informative_bin_on_its_own_scale():
    """Half weight to the bin and half to its stratum at the default
    d0 = K - 1, so a bin with a real spread is not flattened toward its
    neighbours."""
    rng = np.random.default_rng(10)
    n_cells, n_bins, K = 6, 500, 10
    st = np.zeros((n_cells, n_bins), dtype=np.int32)
    keep = np.ones((n_cells, n_bins), bool)
    draws = [rng.normal(0, 1, (n_cells, n_bins)) for _ in range(K)]
    obs = rng.normal(0, 1, (n_cells, n_bins))
    _, info = cc.null_z_matrix(obs, draws, st, keep)
    assert 0.4 < info["frac_own_variance"] < 0.6
    # No moderation at all when the prior is given no weight.
    _, none = cc.null_z_matrix(obs, draws, st, keep, prior_dof=0.0)
    assert none["frac_own_variance"] == pytest.approx(1.0, abs=1e-9)


def test_strata_supply_the_moderation_target_not_one_global_number():
    """A stratum of genuinely wide bins must not be squeezed toward a
    stratum of narrow ones -- that would be the unstratified-pool mistake
    in the denominator."""
    rng = np.random.default_rng(11)
    n_cells, n_bins, K = 6, 800, 10
    wide = np.arange(n_bins) >= 400
    sd = np.where(wide, 5.0, 0.2)[None, :].repeat(n_cells, 0)
    st = np.where(wide, 1, 0).astype(np.int32)[None, :].repeat(n_cells, 0)
    keep = np.ones(sd.shape, bool)
    draws = [rng.normal(0, sd) for _ in range(K)]
    obs = rng.normal(0, sd)
    z, _ = cc.null_z_matrix(obs, draws, st, keep)
    for half in (slice(0, 400), slice(400, None)):
        v = z[:, half]
        assert abs(float(np.nanstd(v)) - 1.0) < 0.12, float(np.nanstd(v))
    # Pooling the two strata into one target breaks exactly that.
    z1, _ = cc.null_z_matrix(obs, draws, np.zeros_like(st), keep)
    assert abs(float(np.nanstd(z1[:, :400])) - 1.0) > 0.3


def test_null_z_is_emitted_as_its_own_matrix(tmp_path):
    res = _run(tmp_path, planted=set(range(0, 60, 7)), n_draws=10)
    assert res["null_z"] is None          # off by default

    out = tmp_path / "zout"
    src = tmp_path / "zsrc"
    src.mkdir(parents=True)
    r = cc.compute_signed_concordance_matrix(
        null_rank_paths(src, n_cells=12, n_bins=60,
                        planted=set(range(0, 60, 7)), coord=0.8),
        bin_size=10_000, min_distance=500, min_gch_sites=2,
        baseline="distance-matched", dm_dist_bin=500, dm_prior="shared",
        output_dir=str(out), save_outputs=True, use_rwr=False, n_threads=1,
        eb_min_obs=50, null_draws=10, null_z=True, null_rank=False)
    assert r["null_z"] is not None
    assert r["null_rank"] is None         # independent switches
    names = {p.name for p in out.glob("*.parquet")}
    assert "concordance_coordination_z.parquet" in names
    assert "concordance_coordination_rank.parquet" not in names
    # And it finds the planted bins with the right sign.
    z = r["null_z"]["coordination"].to_numpy(float)
    idx = np.array([int(b.split(":")[1]) // 10_000
                    for b in r["coordination"].columns])
    pl = np.isin(idx, list(range(0, 60, 7)))
    assert np.nanmean(z[:, pl]) > 2.0
    assert abs(np.nanmean(z[:, ~pl])) < 0.5


def test_null_z_reaches_step_five_from_the_cli():
    import inspect
    import re
    from nome_hic_concordance import cli
    src = inspect.getsource(cli)
    assert '"--null-z"' in src
    assert len(re.findall(r"\bnull_z=", src)) >= 2
    assert "null_z" in inspect.signature(
        cc.compute_signed_concordance_matrix).parameters


def test_the_allocation_is_reported_before_it_happens(tmp_path, caplog):
    """A large cohort dies here as a bare OOM kill with no clue which option
    caused it.  Every drawn-null replicate is FOUR cell-by-bin matrices, so
    --null-draws dominates: at 1,577 cells and 9,548 bins the 12 base
    matrices are 1.4 GB and --null-draws 10 takes it to 6.3 GB, doubled
    again by the imputed copy before RWR allocates anything."""
    src = tmp_path / "src"
    src.mkdir(parents=True)
    with caplog.at_level("INFO"):
        cc.compute_signed_concordance_matrix(
            null_rank_paths(src, n_cells=8, n_bins=40),
            bin_size=10_000, min_distance=500, min_gch_sites=2,
            dm_dist_bin=500, baseline="distance-matched", dm_prior="shared",
            output_dir=str(tmp_path / "o"), save_outputs=False,
            use_rwr=False, n_threads=1, eb_min_obs=50, null_draws=6,
            emit_null=True)
    msgs = [r.getMessage() for r in caplog.records]
    line = [m for m in msgs if "Allocating" in m]
    assert line, msgs
    # 8 soft-count + 4 dm + 6 x 4 null = 36, and it names the null share.
    assert "36 cell-by-bin matrices" in line[0]
    assert "24 of them are 6 drawn-null replicates" in line[0]


def test_the_memory_arithmetic_matches_the_cohort_that_was_killed():
    """1,577 cells x 9,548 bins, --null-draws 10: 52 matrices at 120 MB is
    6.3 GB raw and 12.5 GB with the imputed copy, before RWR."""
    n_cells, n_bins = 1577, 9548
    one_mb = n_cells * n_bins * 8 / 1e6
    assert one_mb == pytest.approx(120, abs=2)
    for k, want_gb in ((0, 1.4), (3, 2.9), (5, 3.9), (10, 6.3)):
        n_mat = 8 + 4 + 4 * k
        assert n_mat * one_mb / 1000 == pytest.approx(want_gb, abs=0.1), k
    # Dropping 10 -> 3 more than halves it, which is the advice given.
    assert (8 + 4 + 4 * 3) / (8 + 4 + 4 * 10) < 0.5


# ---------------------------------------------------------------------------
# The drawn null is never a real contact
# ---------------------------------------------------------------------------


def test_the_drawn_null_never_is_a_real_contact():
    """In a sparse bin a random pairing can land on a contact itself.

    Four ends, one contact (0, 5000), and that contact is the ONLY pair whose
    two ends are both open -- so any co-open mass in the null is the contact
    leaking into it.
    """
    epos = np.array([0, 5_000, 10_000, 20_000], np.int64)
    ep = np.array([1.0, 1.0, 0.0, 0.0])
    keys = cc._contact_keys([0], [5_000])
    for seed in range(60):
        o, c, m, n, hits = cc._null_observed_for_bin(
            epos, ep, 3, 1_000, seed, contact_keys=keys)
        assert o == 0.0, seed
        assert n == 3, "redrawn to full size, not dropped"
    # ...and without the exclusion the contact does get drawn
    leaked = [cc._null_observed_for_bin(epos, ep, 3, 1_000, s)[0] > 0
              for s in range(60)]
    assert any(leaked)
    # a draw refuses a contact exactly when the old one would have used it
    hit = [cc._null_observed_for_bin(epos, ep, 3, 1_000, s,
                                     contact_keys=keys)[4] for s in range(60)]
    assert [h > 0 for h in hit] == leaked


def test_the_hits_measure_how_often_a_random_pairing_is_a_contact():
    """2 of the 12 ordered pairings are the contact: a rate of 1/6."""
    epos = np.array([0, 5_000, 10_000, 20_000], np.int64)
    ep = np.full(4, 0.5)
    keys = cc._contact_keys([0], [5_000])
    runs = [cc._null_observed_for_bin(epos, ep, 3, 1_000, s,
                                      contact_keys=keys) for s in range(400)]
    hits = sum(r[4] for r in runs)
    kept = sum(r[3] for r in runs)
    assert kept == 1200
    assert abs(hits / (hits + kept) - 1 / 6) < 0.04


def test_contact_keys_are_orientation_free():
    assert np.array_equal(cc._contact_keys([5, 900], [100, 2]),
                          cc._contact_keys([100, 2], [5, 900]))
    keys = cc._contact_keys([5, 900], [100, 2])
    assert cc._is_contact([100, 2, 5], [5, 900, 2], keys).tolist() == [
        True, True, False]
    assert not cc._is_contact([1], [2], None).any()


def test_a_run_reports_how_often_the_null_hit_a_contact(tmp_path, caplog):
    with caplog.at_level("INFO"):
        _run(tmp_path, n_draws=2, emit_null=True)
    msgs = [r.getMessage() for r in caplog.records
            if r.getMessage().startswith("Drawn null:")]
    assert sum("own read pairs" in m for m in msgs) == 1
    # ...and how much of the null it could not draw at all
    short = [m for m in msgs if "pairs wanted" in m]
    assert len(short) == 1 and "cell-bins with contacts" in short[0]


def test_a_bin_too_thin_to_split_asks_for_a_null_and_gets_none(tmp_path):
    """One contact and nothing else: two read ends, no half to draw from."""
    path = tmp_path / "c.parquet"
    pd.DataFrame({
        "chrom": ["chr1"], "pos1": [100], "pos2": [5_100],
        "n1": [5], "n2": [5], "distance": [5_000],
        "posterior1": [0.9], "posterior2": [0.1],
    }).to_parquet(path)
    _, counts = cc._compute_cell_dm_expected(
        str(path), "c", bin_size=10_000, min_distance=500, min_gch_sites=2,
        mode="pooled", null_split=True, null_draws=3)
    rec = counts["chr1:0"]
    assert rec["n_null_wanted"] == 3.0
    assert cc._null_keys(0)[3] not in rec


# ---------------------------------------------------------------------------
# The null runs the whole calculation on its own pairs
# ---------------------------------------------------------------------------


def _rwr_run(tmp_path, null_graph, passes, n_draws=2, **kw):
    src = tmp_path / f"src_{null_graph}_{passes}"
    src.mkdir(parents=True, exist_ok=True)
    paths = null_rank_paths(src, planted=set(range(0, 60, 4)))
    return cc.compute_signed_concordance_matrix(
        paths, bin_size=10_000, min_distance=500, min_gch_sites=2,
        baseline="distance-pooled", dm_prior="shared",
        output_dir=str(tmp_path / f"out_{null_graph}_{passes}"),
        save_outputs=False, use_rwr=True, rwr_k=4,
        rwr_n_refine_passes=passes, rwr_refinement_basis="postnorm",
        n_threads=1, eb_min_obs=50, emit_null=True, null_draws=n_draws,
        null_graph=null_graph, **kw)


@pytest.fixture
def rwr_calls(monkeypatch):
    """Every RWR run step 5 starts, with the counts it was handed."""
    from nome_hic_concordance import imputation
    calls, real = [], imputation.run_rwr_pipeline

    def spy(soft_count_matrices, *a, **kw):
        calls.append(dict(soft_count_matrices))
        return real(soft_count_matrices, *a, **kw)

    monkeypatch.setattr(imputation, "run_rwr_pipeline", spy)
    return calls


def test_each_null_draw_runs_its_own_rwr(tmp_path, rwr_calls):
    res = _rwr_run(tmp_path, "own", passes=1)
    null_ks = {k for d in range(2) for k in cc._null_keys(d)}
    assert len(rwr_calls) == 3, "the observed, then one run per draw"
    assert not null_ks & set(rwr_calls[0])
    for d, counts in enumerate(rwr_calls[1:]):
        # the draw's pairs stand in for the contacts...
        assert np.array_equal(counts["k_open"],
                              res["raw_counts"][cc._null_keys(d)[0]])
        assert np.array_equal(counts["n_linked"],
                              res["raw_counts"][cc._null_keys(d)[3]])
        # ...and E is imputed again, over the null's own graph
        assert np.array_equal(counts["k_dm_open"],
                              res["raw_counts"]["k_dm_open"])
        assert not null_ks & set(counts)
    assert len(res["null_rwr_diagnostics"]) == 2


def test_a_shared_null_rides_on_the_observed_run(tmp_path, rwr_calls):
    _rwr_run(tmp_path, "shared", passes=1)
    assert len(rwr_calls) == 1
    assert set(cc._null_keys(0)) <= set(rwr_calls[0])


def test_the_observed_does_not_depend_on_where_the_null_is_imputed(tmp_path):
    own = _rwr_run(tmp_path, "own", passes=1)
    shared = _rwr_run(tmp_path, "shared", passes=1)
    for name in ("coordination", "direction_logor", "marginal"):
        assert np.array_equal(own[name].to_numpy(), shared[name].to_numpy(),
                              equal_nan=True), name
    # the run reports the observed's priors, not the last null's
    assert own["priors"]["linked"]["alpha"] == (
        shared["priors"]["linked"]["alpha"])


def test_without_refinement_both_nulls_use_the_same_graph(tmp_path):
    """Pass 0 is built from accessibility, which no pairing changes."""
    own = _rwr_run(tmp_path, "own", passes=0)
    shared = _rwr_run(tmp_path, "shared", passes=0)
    a = own["null_components"]["coordination"].to_numpy()
    b = shared["null_components"]["coordination"].to_numpy()
    assert np.allclose(a, b, equal_nan=True)


def test_with_refinement_the_null_builds_its_own_graph(tmp_path):
    """A refinement pass rebuilds the graph from the concordance it is
    imputing.  On the observed's graph the null is smoothed over
    neighbourhoods the OBSERVED chose; on its own, over its own."""
    own = _rwr_run(tmp_path, "own", passes=1)
    shared = _rwr_run(tmp_path, "shared", passes=1)
    a = own["null_components"]["coordination"].to_numpy()
    b = shared["null_components"]["coordination"].to_numpy()
    assert not np.allclose(a, b, equal_nan=True)


def test_the_null_graph_is_checked():
    with pytest.raises(ValueError, match="null_graph"):
        cc.compute_signed_concordance_matrix({}, null_graph="observed")


# ---------------------------------------------------------------------------
# A cell graph built from coordination alone
# ---------------------------------------------------------------------------


def test_refinement_builds_its_graph_from_graph_fn(monkeypatch):
    """layer2_fn makes the returned statistic; graph_fn, when given, is what
    every refinement graph is built from."""
    from nome_hic_concordance import imputation as imp
    rng = np.random.default_rng(0)
    n, nb = 30, 20
    cols = [f"chr1:{j}" for j in range(nb)]
    feats, basis = [], []
    real_build = imp.build_knn_graph

    def spy_build(features, k=15, metric="cosine"):
        feats.append(np.array(features))
        return real_build(features, k=k, metric=metric)

    def graph(imputed):
        basis.append(imputed["x"] * 2.0)
        return pd.DataFrame(basis[-1], columns=cols)

    monkeypatch.setattr(imp, "build_knn_graph", spy_build)
    imp.run_rwr_pipeline(
        {"x": rng.random((n, nb))}, pd.DataFrame(rng.random((n, nb))),
        lambda imputed: pd.DataFrame(np.zeros((n, nb)), columns=cols),
        n_refine_passes=2, k=5, refinement_basis="raw", graph_fn=graph)
    assert len(feats) == 3 and len(basis) == 3
    for p in (1, 2):
        assert np.allclose(feats[p], imp.quantile_normalize_for_knn(basis[p - 1]))


@pytest.fixture
def rwr_kwargs(monkeypatch):
    """Every RWR run step 5 starts, with everything it was handed."""
    from nome_hic_concordance import imputation
    calls, real = [], imputation.run_rwr_pipeline

    def spy(soft_count_matrices, *a, **kw):
        calls.append({"counts": dict(soft_count_matrices), **kw})
        return real(soft_count_matrices, *a, **kw)

    monkeypatch.setattr(imputation, "run_rwr_pipeline", spy)
    return calls


def test_a_coordination_graph_starts_from_the_observed_C(tmp_path, rwr_kwargs):
    _rwr_run(tmp_path, "own", passes=1, rwr_init="coordination")
    assert len(rwr_kwargs) == 3
    seen = []
    for call in rwr_kwargs:
        assert call["graph_fn"] is not None
        init = call["initial_feature_matrix"].to_numpy()
        n = call["counts"]
        # a cell-bin with no contact is MISSING, not a 0 that says nothing
        has = (n["n_linked"] > 0) & (n["n_dm"] > 0)
        assert np.array_equal(np.isfinite(init), has)
        seen.append(init)
    # the null starts from its own pairs, on the observed's cell-bins
    obs, null = seen[0], seen[1]
    assert not (np.isfinite(null) & ~np.isfinite(obs)).any()
    both = np.isfinite(obs) & np.isfinite(null)
    assert both.mean() > 0.5 and not np.allclose(obs[both], null[both])


def test_by_default_the_graph_starts_from_accessibility(tmp_path, rwr_kwargs):
    res = _rwr_run(tmp_path, "own", passes=0)
    assert rwr_kwargs[0]["graph_fn"] is None
    assert np.allclose(
        rwr_kwargs[0]["initial_feature_matrix"].to_numpy(),
        res["accessibility"].to_numpy(), equal_nan=True)


def test_rwr_init_is_checked():
    with pytest.raises(ValueError, match="rwr_init"):
        cc.compute_signed_concordance_matrix({}, rwr_init="methylation")


def _level_only_types(tmp_path, n_types=3, per_type=12, n_bins=60,
                      bin_size=10_000, depth=8, seed=0):
    """Cell types that differ ONLY in methylation level.  Every pair's two
    ends are drawn independently, so there is no coordination anywhere."""
    rng = np.random.default_rng(seed)
    marg = rng.uniform(0.1, 0.9, (n_types, n_bins))
    paths, types = {}, []
    for t in range(n_types):
        for c in range(per_type):
            rec = []
            for b in range(n_bins):
                pm, base = marg[t, b], b * bin_size
                for _ in range(rng.poisson(depth)):
                    a, bb = rng.random() < pm, rng.random() < pm
                    o1 = int(rng.integers(100, 3000))
                    o2 = o1 + int(rng.integers(1000, 6000))
                    rec.append((base + o1, base + o2, 0.9 if a else 0.1,
                                0.9 if bb else 0.1, o2 - o1))
                    p3 = 0.9 if rng.random() < pm else 0.1
                    rec.append((base + o1, base + o1 + 40, 0.9 if a else 0.1,
                                p3, 40))
            df = pd.DataFrame(rec, columns=["pos1", "pos2", "posterior1",
                                            "posterior2", "distance"])
            df.insert(0, "chrom", "chr1")
            df["n1"] = np.int32(4)
            df["n2"] = np.int32(4)
            name = f"t{t}_c{c}"
            df.to_parquet(tmp_path / f"{name}.parquet")
            paths[name] = str(tmp_path / f"{name}.parquet")
            types.append(t)
    return paths, np.array(types)


@pytest.mark.parametrize("seed", [0, 1])
def test_a_coordination_graph_gives_level_only_types_nothing(tmp_path, seed):
    """The leak, and the cure.  Over a graph built from methylation, C with
    no coordination in it separates cell types that differ only in level;
    over a graph built from C itself it does not."""
    import tempfile
    from sklearn.decomposition import TruncatedSVD
    from nome_hic_concordance.comparison import _prep_matrix
    from nome_hic_concordance.metrics import knn_chance, knn_purity

    paths, types = _level_only_types(tmp_path, seed=seed)

    def purity(init):
        res = cc.compute_signed_concordance_matrix(
            paths, bin_size=10_000, min_distance=500, min_gch_sites=2,
            baseline="distance-pooled", dm_prior="shared",
            output_dir=str(tmp_path / "out"), save_outputs=False,
            use_rwr=True, rwr_k=8, rwr_n_refine_passes=0, rwr_init=init,
            n_threads=1)
        with tempfile.TemporaryDirectory() as d:
            Z = _prep_matrix(res["coordination"], "x", d, "quantile", 0.0,
                             "rank", True).to_numpy()
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
        return knn_purity(TruncatedSVD(5, random_state=0).fit_transform(Z),
                          types)

    chance = knn_chance(types)
    assert purity("accessibility") > chance + 0.15
    assert purity("coordination") < chance + 0.1
