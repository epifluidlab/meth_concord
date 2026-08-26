"""The floor under C that no baseline can remove.

O and E are both Dirichlet-Multinomial posterior means
(alpha_i + k_i) / (alpha_0 + n), and n_dm == n_linked by construction, so the
two share a denominator and

    C = [ (a_linked_conc - a_expected_conc) + (k_conc - k_dm_conc) ] / (a_0 + n)

The first term is data-free.  Fitting a separate EB prior per pool sets each
pool's shrinkage target to its own global mean, and those means differ by the
genome-wide average coordination -- so a bin with no information is handed the
global average instead of zero.  Post-RWR the imputed n is usually < 1, which
makes the posterior almost all prior, so that term is most of C.

On a mouse cortex run: gap 0.5938 - 0.5825 = 0.0113, median C = 0.011293.
"""

from __future__ import annotations

import numpy as np
import pytest

from nome_hic_concordance import concordance as cc


def _post(alpha, k, n):
    """The posterior concordant mass this pool reports."""
    a0 = sum(alpha.values())
    return (alpha["open"] + alpha["closed"] + k) / (a0 + n)


def test_separate_priors_leave_a_data_free_floor_under_c():
    """Identical observed and expected counts -- perfect agreement, so the
    honest answer is C = 0 -- still score positive when the two pools were
    shrunk toward different targets."""
    a_l = {"open": 30.0, "closed": 29.38, "mixed": 40.62}     # conc 0.5938
    a_d = {"open": 30.0, "closed": 28.25, "mixed": 41.75}     # conc 0.5825
    for n in (0.0, 0.5, 5.0):
        k = 0.5938 * n                     # both pools see the same data
        c = _post(a_l, k, n) - _post(a_d, k, n)
        assert c > 0
    assert _post(a_l, 0.0, 0.0) - _post(a_d, 0.0, 0.0) == pytest.approx(
        0.0113, abs=1e-9)


def test_a_shared_prior_removes_it_exactly_at_every_n():
    a = {"open": 30.0, "closed": 29.38, "mixed": 40.62}
    for n in (0.0, 0.5, 5.0, 500.0):
        k = 0.4 * n
        assert _post(a, k, n) - _post(a, k, n) == 0.0


def test_the_floor_is_largest_where_the_imputed_n_is_smallest():
    """RWR hands back fractional n, typically < 1, so the posterior is nearly
    all prior and the gap is nearly all of C."""
    a_l = {"open": 30.0, "closed": 29.38, "mixed": 40.62}
    a_d = {"open": 30.0, "closed": 28.25, "mixed": 41.75}
    floors = [_post(a_l, 0.0, n) - _post(a_d, 0.0, n)
              for n in (0.1, 1.0, 10.0, 100.0)]
    assert floors == sorted(floors, reverse=True)
    assert floors[0] / floors[-1] > 1.9


def test_the_gap_is_reported_whichever_prior_was_used(tmp_path):
    from tests.test_signed_concordance import _focal_active_paths
    for mode in ("separate", "shared"):
        out = tmp_path / mode
        src = tmp_path / f"src_{mode}"
        src.mkdir(parents=True, exist_ok=True)
        cc.compute_signed_concordance_matrix(
            _focal_active_paths(src, n_cells=2),
            bin_size=1000, min_distance=100, min_gch_sites=2,
            baseline="distance-matched", dm_dist_bin=100, dm_prior=mode,
            output_dir=str(out), save_outputs=True, use_rwr=False,
        )
        import json
        pri = json.load(open(out / "dirichlet_multinomial_prior.json"))
        assert "prior_gap" in pri
        assert pri["prior_gap"]["dm_prior"] == mode
        assert "shared" in pri if mode == "shared" else True


def test_sharing_makes_the_two_priors_identical(tmp_path):
    import json
    from tests.test_signed_concordance import _focal_active_paths
    out = tmp_path / "o"
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    cc.compute_signed_concordance_matrix(
        _focal_active_paths(src, n_cells=2),
        bin_size=1000, min_distance=100, min_gch_sites=2,
        baseline="distance-matched", dm_dist_bin=100, dm_prior="shared",
        output_dir=str(out), save_outputs=True, use_rwr=False,
    )
    pri = json.load(open(out / "dirichlet_multinomial_prior.json"))
    assert pri["linked"]["alpha"] == pri["shared"]["alpha"]
    assert pri["dm_expected"]["alpha"] == pri["shared"]["alpha"]


def test_a_bad_dm_prior_is_an_error(tmp_path):
    from tests.test_signed_concordance import _focal_active_paths
    with pytest.raises(ValueError, match="dm_prior"):
        cc.compute_signed_concordance_matrix(
            _focal_active_paths(tmp_path, n_cells=2),
            bin_size=1000, min_distance=100, min_gch_sites=2,
            baseline="distance-matched", dm_prior="pooled",
            save_outputs=False, use_rwr=False,
        )


def test_the_same_gap_floors_the_log_odds_direction():
    """L = log(O_o/O_c) - log(E_o/E_c) cancels the shared denominator rather
    than sharing it, so the prior enters as the log-odds ratio of the two
    priors' own open:closed splits.  Same defect, same n=0 limit."""
    a_l = {"open": 30.00, "closed": 29.38, "mixed": 40.62}
    a_d = {"open": 26.00, "closed": 32.25, "mixed": 41.75}
    lor = lambda a, ko, kc: np.log((a["open"] + ko) / (a["closed"] + kc))
    at_zero = lor(a_l, 0, 0) - lor(a_d, 0, 0)
    assert at_zero == pytest.approx(
        np.log(30.00 / 29.38) - np.log(26.00 / 32.25))
    assert abs(at_zero) > 0.2                       # far from the null
    # one prior, and the same limit is exactly zero
    assert lor(a_l, 0, 0) - lor(a_l, 0, 0) == 0.0


def test_the_floors_are_arithmetic_on_the_priors_handed_in():
    """The numbers from a real separate-prior run, so the formula is pinned
    to something measured rather than to itself."""
    a_l = {"open": 10.182, "closed": 49.196, "mixed": 40.622}
    a_d = {"open": 11.281, "closed": 46.967, "mixed": 41.752}
    c, l = cc._prior_floors(a_l, a_d)
    assert c == pytest.approx(0.0113, abs=5e-4)
    assert l == pytest.approx(-0.1489, abs=5e-4)
    assert cc._prior_floors(a_l, a_l) == (0.0, 0.0)


def test_the_floor_is_computed_from_the_priors_that_were_used(tmp_path):
    """The report used to be written BEFORE the shared fit replaced the
    priors, so a --dm-prior shared run published the floor it had just
    removed -- 0.017841 and -0.191811 on a real run -- and read as though
    the fix had not applied.  The rates were right and only the JSON was
    wrong, which is the worst of the two to have.

    Asserted as CONSISTENCY between the reported priors and the reported
    floor.  The earlier version checked the floor against 0 on a fixture too
    small for the EB fit, where every prior falls back to Dirichlet(1, 1, 1)
    and every gap is 0 however the code is ordered -- so it passed on the
    broken code."""
    import json
    from tests.test_signed_concordance import _focal_active_paths
    for mode in ("separate", "shared"):
        src, out = tmp_path / f"s_{mode}", tmp_path / f"o_{mode}"
        src.mkdir(parents=True, exist_ok=True)
        cc.compute_signed_concordance_matrix(
            _focal_active_paths(src, n_cells=3),
            bin_size=1000, min_distance=100, min_gch_sites=2,
            baseline="distance-matched", dm_dist_bin=100, dm_prior=mode,
            # Force the real method-of-moments fit.  At the default
            # min_obs the fixture falls back to Dirichlet(1, 1, 1) for every
            # pool, which makes every gap 0 whatever the code does.
            eb_min_obs=1,
            output_dir=str(out), save_outputs=True, use_rwr=False,
        )
        pri = json.load(open(out / "dirichlet_multinomial_prior.json"))
        gap = pri["prior_gap"]
        exp_c, exp_l = cc._prior_floors(pri["linked"]["alpha"],
                                        pri["dm_expected"]["alpha"])
        assert gap["implied_C_floor_at_n0"] == pytest.approx(exp_c)
        assert gap["implied_logor_floor_at_n0"] == pytest.approx(exp_l)
        assert gap["dm_prior"] == mode
        if mode == "shared":
            # what was used, and separately what was removed
            assert pri["linked"]["alpha"] == pri["shared"]["alpha"]
            assert pri["dm_expected"]["alpha"] == pri["shared"]["alpha"]
            assert "separate_fit" in pri
            assert "separate_fit_C_floor" in gap
            assert "separate_fit" in pri
            assert gap["implied_C_floor_at_n0"] == 0.0
            assert gap["implied_logor_floor_at_n0"] == 0.0


def test_the_report_is_written_after_the_prior_is_replaced():
    """The bug was pure ordering, and no fixture here can catch it
    behaviourally: the linked and expected pools come out with identical
    concordant mass (0.9050 vs 0.9050), so the gap is 0 whichever priors the
    report is taken from.  That is precisely how the two tests written with
    the feature passed on the broken code -- at the default min_obs every
    pool falls back to Dirichlet(1, 1, 1) as well.  So the invariant is
    pinned where it lives, in the order of the two writes."""
    import inspect
    src = inspect.getsource(cc.compute_signed_concordance_matrix)
    i_share = src.index('fitted_priors["shared"] =')
    i_used = src.index('fitted_priors["linked"] = {"alpha": alpha_sh')
    i_gap = src.index('fitted_priors["prior_gap"] =')
    assert i_share < i_gap, "the shared fit must precede the gap report"
    assert i_used < i_gap, "'linked' must name the prior actually used"


def test_the_logor_floor_is_reported_too(tmp_path):
    import json
    from tests.test_signed_concordance import _focal_active_paths
    src, out = tmp_path / "src", tmp_path / "o"
    src.mkdir(parents=True, exist_ok=True)
    cc.compute_signed_concordance_matrix(
        _focal_active_paths(src, n_cells=2),
        bin_size=1000, min_distance=100, min_gch_sites=2,
        baseline="distance-matched", dm_dist_bin=100, dm_prior="shared",
        output_dir=str(out), save_outputs=True, use_rwr=False,
    )
    gap = json.load(open(out / "dirichlet_multinomial_prior.json"))["prior_gap"]
    assert gap["implied_logor_floor_at_n0"] == pytest.approx(0.0, abs=1e-12)
    assert gap["implied_C_floor_at_n0"] == pytest.approx(0.0, abs=1e-12)
