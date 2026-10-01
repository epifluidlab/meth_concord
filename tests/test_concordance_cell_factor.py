"""concordance_cell_factor.py: telling a depth stripe from cell-type biology.

Two planted matrices on one fixture.  In the first, a cell's concordance is
a genome-wide bin pattern scaled by how much data the cell has -- the shape
of a heatmap whose columns are solid stripes.  Depth also differs between
cell types, so that factor separates cell types too, which is exactly why a
cell-type ranking selects it.  In the second, each cell type has its own bin
profile and depth plays no part.  The script has to call the first technical
and the second not, and the within-cell-type correlation is what tells them
apart: across all cells both factors track depth, because depth differs by
cell type.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "concordance_cell_factor.py")


def _mod():
    spec = importlib.util.spec_from_file_location("cell_factor", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["cell_factor"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def run(tmp_path):
    rng = np.random.default_rng(0)
    types = np.repeat(["T1", "T2", "T3"], 60)
    n, nb = len(types), 300
    cells = [f"c{i}" for i in range(n)]
    bins = [f"chr{1 + j // 150}:{(j % 150) * 250_000}" for j in range(nb)]
    # Depth differs by cell type AND varies within it.
    type_depth = {"T1": 1.0, "T2": 0.4, "T3": 0.15}
    depth = np.array([type_depth[t] for t in types]) * rng.lognormal(
        0, 0.6, n)
    cover = rng.uniform(0.02, 0.3, nb)
    pairs = rng.poisson(40 * np.outer(depth, cover))
    d = tmp_path / "05_concordance"
    d.mkdir()
    frame = dict(index=cells, columns=bins)
    pd.DataFrame(pairs, **frame).to_parquet(d / "pair_counts_per_bin.parquet")
    level = np.clip(0.75 + rng.normal(0, 0.05, (n, nb)), 0.01, 0.99)
    pd.DataFrame(level, **frame).to_parquet(d / "accessibility_mean.parquet")

    # Technical: one bin pattern, cells differ in how cleanly they show it.
    g = rng.normal(0.05, 0.05, nb)
    snr = depth / (depth + 0.3)
    tech = np.outer(snr, g) + rng.normal(0, 0.03, (n, nb))
    pd.DataFrame(tech, **frame).to_parquet(d / "concordance_tech.parquet")

    # Biological: each cell type its own profile; depth plays no part.
    prof = {t: rng.normal(0, 0.05, nb) for t in ("T1", "T2", "T3")}
    bio = (np.stack([prof[t] for t in types])
           + rng.normal(0, 0.03, (n, nb)))
    pd.DataFrame(bio, **frame).to_parquet(d / "concordance_bio.parquet")

    sp = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells, "cell_type": types,
                  "exp_batch": np.tile(["A", "B"], n // 2)}).to_csv(
        sp, sep="\t", index=False)
    return d, sp, tmp_path


def _summary(run, *names):
    d, sp, tmp = run
    m = _mod()
    argv = []
    for nm in names:
        argv += ["--matrix", f"{nm}={d / f'concordance_{nm}.parquet'}"]
    argv += ["--samples", str(sp), "--cell-type-column", "cell_type",
             "--batch-column", "exp_batch", "-o", str(tmp / "out" / "cf"),
             "--plot-format", "png"]
    assert m.main(argv) == 0
    return pd.read_csv(tmp / "out" / "cf.summary.tsv", sep="\t")


def _get(s, name, kind, cov):
    v = s[(s["matrix"] == name) & (s["kind"] == kind)
          & (s["covariate"] == cov)]["value"]
    return float(v.iloc[0])


def test_a_depth_stripe_tracks_depth_within_cell_type(run):
    s = _summary(run, "tech", "bio")
    assert abs(_get(s, "tech", "rho_within_type", "log10_pairs")) > 0.5
    assert abs(_get(s, "tech", "rho_within_type", "observed_frac")) > 0.4


def test_cell_type_biology_does_not(run):
    s = _summary(run, "tech", "bio")
    assert abs(_get(s, "bio", "rho_within_type", "log10_pairs")) < 0.2
    # ...although across ALL cells it does, since depth differs by type:
    # this is why the within-type number is the one to read.
    assert abs(_get(s, "bio", "rho_all", "log10_pairs")) > 0.25
    assert _get(s, "bio", "eta2", "cell_type") > 0.8


def test_outputs_are_written(run):
    d, sp, tmp = run
    _summary(run, "tech")
    for suffix in (".cells.tsv", ".bins.tsv", ".summary.tsv", ".png"):
        assert (tmp / "out" / f"cf{suffix}").exists(), suffix
    cells = pd.read_csv(tmp / "out" / "cf.cells.tsv", sep="\t")
    assert {"PC1", "log10_pairs", "cell_type", "exp_batch"} <= set(cells)


def test_pc1_is_oriented_to_the_mean_value(run):
    """The SVD sign is arbitrary; the plots read 'up = more concordant'."""
    d, sp, tmp = run
    _summary(run, "tech")
    cells = pd.read_csv(tmp / "out" / "cf.cells.tsv", sep="\t")
    assert cells["PC1"].corr(cells["mean_value"], method="spearman") > 0


def test_missing_sidecars_leave_covariates_empty_not_a_crash(run):
    d, sp, tmp = run
    (d / "pair_counts_per_bin.parquet").unlink()
    s = _summary(run, "bio")
    assert np.isnan(_get(s, "bio", "rho_within_type", "log10_pairs"))
    assert np.isfinite(_get(s, "bio", "eta2", "cell_type"))


def test_the_mle_edge_is_counted():
    """|C| > 0.5 is the imputed-n == 0 edge, not a coordination estimate."""
    m = _mod()
    raw = pd.DataFrame([[0.01, -0.62, 0.02, np.nan], [0.0, 0.03, 0.01, 0.02]],
                       index=["a", "b"], columns=list("wxyz"))
    cov = m.cell_covariates(raw, None, None)
    assert cov.loc["a", "edge_frac"] == pytest.approx(1 / 3)
    assert cov.loc["b", "edge_frac"] == 0
    # an exact 0 is "nothing reached this cell", counted separately
    assert cov.loc["a", "zero_frac"] == 0
    assert cov.loc["b", "zero_frac"] == pytest.approx(1 / 4)


# ------------------------------------------- level tracking, bin by bin ---

@pytest.fixture
def level_run(tmp_path):
    """Eight cell types whose methylation differs bin by bin, as it does.

    'levelL' is L computed from each cell's own level in each bin at ONE
    coordination everywhere, so its cell-type blocks are methylation and
    nothing else.  'bioC' has cell-type profiles unrelated to the level.
    """
    rng = np.random.default_rng(1)
    types = np.repeat([f"T{i}" for i in range(8)], 25)
    n, nb = len(types), 300
    cells = [f"c{i}" for i in range(n)]
    bins = [f"chr1:{j * 250_000}" for j in range(nb)]
    t_idx = np.searchsorted(np.unique(types), types)
    eff = rng.normal(0, 0.06, (8, nb))                 # DMR-like
    p = np.clip(0.62 + eff[t_idx] + rng.normal(0, 0.03, (n, nb)), 0.3, 0.95)
    d = tmp_path / "05_concordance"
    d.mkdir()
    frame = dict(index=cells, columns=bins)
    pd.DataFrame(p, **frame).to_parquet(d / "accessibility_mean.parquet")
    r = 0.1
    L = r * (1 - 2 * p) / (p * (1 - p)) + rng.normal(0, 0.02, (n, nb))
    pd.DataFrame(L, **frame).to_parquet(d / "concordance_levelL.parquet")
    prof = rng.normal(0, 0.03, (8, nb))
    bio = prof[t_idx] + rng.normal(0, 0.02, (n, nb))
    pd.DataFrame(bio, **frame).to_parquet(d / "concordance_bioC.parquet")
    sp = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells, "cell_type": types}).to_csv(
        sp, sep="\t", index=False)
    return d, sp, tmp_path


def _level_summary(level_run):
    d, sp, tmp = level_run
    m = _mod()
    argv = ["--matrix", f"levelL={d / 'concordance_levelL.parquet'}",
            "--matrix", f"bioC={d / 'concordance_bioC.parquet'}",
            "--samples", str(sp), "--cell-type-column", "cell_type",
            "--top-bins", "100", "-o", str(tmp / "out" / "cf"),
            "--plot-format", "png"]
    assert m.main(argv) == 0
    return pd.read_csv(tmp / "out" / "cf.summary.tsv", sep="\t")


def test_blocks_that_are_methylation_are_called_so(level_run):
    s = _level_summary(level_run)
    same = _get(s, "levelL", "level_tracking_top", "same_abs_r")
    other = _get(s, "levelL", "level_tracking_top", "other_bins_abs_r")
    assert same > 0.8
    assert same > other + 0.3
    # L falls as methylation rises (for p > 0.5 and positive coordination)
    assert _get(s, "levelL", "level_tracking_top", "same_r") < -0.8


def test_blocks_unrelated_to_level_sit_at_chance(level_run):
    s = _level_summary(level_run)
    same = _get(s, "bioC", "level_tracking_top", "same_abs_r")
    chance = _get(s, "bioC", "level_tracking_top", "permuted_types_abs_r")
    assert abs(same - chance) < 0.15


def test_level_test_needs_four_cell_types(run):
    """Three cell-type means cannot carry a correlation worth reporting."""
    s = _summary(run, "bio")
    assert not s["kind"].str.startswith("level_tracking").any()
