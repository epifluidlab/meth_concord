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
    assert flags == {"null-draws", "null-rank", "null-rank-scope",
                     "null-rank-statistics", "null-rank-min-pairs",
                     "null-rank-min-pool"}
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
