"""scripts/imputation_benchmark.py: filling sparse coordination, judged
against its own drawn null.

The synthetic counts are step 5's ``counts/`` layout.  Cell types always
differ in methylation level; whether they also differ in COORDINATION, and
at what scale, is what each test varies.  A method earns its keep only by
the gap between the real counts and the nulls.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "imputation_benchmark",
    Path(__file__).resolve().parents[1] / "scripts" / "imputation_benchmark.py")
ib = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ib)


def synth(out, n_types=3, per_type=20, n_bins=120, bin_size=1_000_000,
          depth=1.0, coord=0.0, pattern="domain", block=10, n_null=2,
          seed=0):
    """Counts for cell types that differ in level everywhere and, with
    ``coord`` > 0, in coordination: each type has its own 10-bin domains
    (``pattern='domain'``) or its own single bins (``'bin'``)."""
    rng = np.random.default_rng(seed)
    out = Path(out)
    (out / "counts").mkdir(parents=True, exist_ok=True)
    types = np.repeat(np.arange(n_types), per_type)
    cells = [f"t{t}_c{i}" for t in range(n_types) for i in range(per_type)]
    half = n_bins // 2
    bins = [f"chr{1 + j // half}:{(j % half) * bin_size}"
            for j in range(n_bins)]
    p = rng.uniform(0.1, 0.9, (n_types, n_bins))[types]
    N = len(cells)
    n = rng.poisson(depth, (N, n_bins))
    j = np.arange(n_bins) % half
    unit = j // block if pattern == "domain" else j
    k = np.where((unit[None, :] % n_types) == types[:, None], coord, 0.0)

    def draw(po, pc):
        pr = np.stack([po, pc, 1 - po - pc], -1)
        o = np.zeros((N, n_bins, 3))
        for a, b in zip(*np.nonzero(n)):
            o[a, b] = rng.multinomial(n[a, b], pr[a, b])
        return o

    real = draw(k * p + (1 - k) * p ** 2, k * (1 - p) + (1 - k) * (1 - p) ** 2)
    frames = {"k_open": real[..., 0], "k_closed": real[..., 1],
              "k_mixed": real[..., 2], "n_linked": n,
              "k_dm_open": n * p ** 2, "k_dm_closed": n * (1 - p) ** 2,
              "k_dm_mixed": n * 2 * p * (1 - p), "n_dm": n}
    for d in range(n_null):
        sfx = "" if d == 0 else str(d)
        nd = draw(p ** 2, (1 - p) ** 2)
        frames.update({f"k_null{sfx}_open": nd[..., 0],
                       f"k_null{sfx}_closed": nd[..., 1],
                       f"k_null{sfx}_mixed": nd[..., 2],
                       f"n_null{sfx}": n})
    for key, arr in frames.items():
        pd.DataFrame(np.asarray(arr, float), index=cells,
                     columns=bins).to_parquet(out / "counts" / f"{key}.parquet")
    acc = np.clip(p + rng.normal(0, 0.05, p.shape), 0, 1)
    pd.DataFrame(acc, index=cells, columns=bins).to_parquet(
        out / "accessibility.parquet")
    pd.DataFrame({"cell_name": cells, "group": [f"T{t}" for t in types],
                  "exp_batch": [f"b{i % 2}" for i in range(N)]}).to_csv(
        out / "samples.tsv", sep="\t", index=False)
    return out


def bench(d, methods, **extra):
    argv = ["--counts-dir", str(d / "counts"), "--samples",
            str(d / "samples.tsv"), "--accessibility",
            str(d / "accessibility.parquet"), "--methods", methods,
            "--n-svd", "5", "--rwr-k", "10", "--min-group-size", "5",
            "--output-dir", str(d / "out")]
    for flag, val in extra.items():
        argv += [f"--{flag.replace('_', '-')}", str(val)]
    assert ib.main(argv) == 0
    return pd.read_csv(d / "out" / "imputation_benchmark_summary.tsv",
                       sep="\t", index_col=0)


# ---------------------------------------------------------------- pieces --


def test_coordination_leaves_an_unobserved_cell_bin_missing():
    z = pd.DataFrame([[0.0, 2.0]], index=["c"], columns=["chr1:0", "chr1:1"])
    cnt = {"n_linked": z, "n_dm": z, "k_open": z * 0.5, "k_closed": z * 0.5,
           "k_dm_open": z * 0.25, "k_dm_closed": z * 0.25}
    c = ib.coordination(cnt)
    assert np.isnan(c.iloc[0, 0]) and c.iloc[0, 1] == pytest.approx(0.5)


def test_pooling_sums_counts_within_a_region_and_a_cell():
    bins = ["chr1:0", "chr1:1000000", "chr1:2000000", "chr2:0"]
    g = ib.region_groups(bins, ib.parse_size("2mb"))
    assert list(g) == ["chr1:0", "chr1:0", "chr1:2000000", "chr2:0"]
    df = pd.DataFrame([[1.0, 2.0, 4.0, 8.0]], index=["c"], columns=bins)
    pooled = ib.pool({"n_linked": df}, g)["n_linked"]
    assert pooled.loc["c"].to_dict() == {"chr1:0": 3.0, "chr1:2000000": 4.0,
                                         "chr2:0": 8.0}


def test_every_null_draw_on_disk_is_found(tmp_path):
    d = synth(tmp_path, n_null=3, per_type=3, n_bins=20)
    assert ib.available_draws(d / "counts") == [0, 1, 2]


def test_the_low_rank_fit_fills_every_cell_bin(tmp_path):
    d = synth(tmp_path, per_type=6, n_bins=40, coord=0.5)
    real, _ = ib.load_counts(d / "counts", [0])
    Z, info = ib.lowrank(real, rank=3, max_iter=200)
    assert Z.shape == real["n_linked"].shape
    assert np.isfinite(Z.to_numpy()).all() and info["n_obs"] > 0


# ------------------------------------------------------------- behaviour --


def test_with_no_coordination_only_the_methylation_graph_separates(tmp_path):
    """Cell types that differ only in level.  Pooling and the low-rank fit
    find nothing, on the real counts or the nulls; RWR over the methylation
    graph separates them anyway -- and its NULL does too, which is the
    leak."""
    s = bench(synth(tmp_path, coord=0.0), "pool:10mb,lowrank,rwr-acc")
    chance = 1 / 3
    for meth in ("pool:10mb", "lowrank"):
        assert s.loc[meth, "knn_purity_real"] < chance + 0.15, meth
        assert s.loc[meth, "knn_purity_null_mean"] < chance + 0.15, meth
    assert s.loc["rwr-acc", "knn_purity_null_mean"] > 0.8


def test_domain_scale_coordination_clears_the_null(tmp_path):
    s = bench(synth(tmp_path, coord=0.5, pattern="domain"),
              "pool:10mb,lowrank,rwr-c")
    for meth in ("pool:10mb", "lowrank", "rwr-c"):
        assert s.loc[meth, "auc_gap"] > 0.2, meth
        assert s.loc[meth, "knn_purity_gap"] > 0.15, meth


def test_pooling_loses_coordination_that_changes_bin_to_bin(tmp_path):
    """Pooling assumes the difference is regional.  When each type is
    coordinated in its own single bins, a region holds all types' bins and
    the difference averages away; the low-rank fit keeps it."""
    s = bench(synth(tmp_path, coord=0.5, pattern="bin"), "pool:10mb,lowrank")
    assert s.loc["pool:10mb", "auc_gap"] < 0.15
    assert s.loc["lowrank", "auc_gap"] > 0.2


def test_the_null_is_summarised_over_its_draws(tmp_path):
    s = bench(synth(tmp_path, per_type=8, n_bins=40, n_null=3), "pool:10mb")
    assert s.loc["pool:10mb", "n_null_draws"] == 3
    assert np.isfinite(s.loc["pool:10mb", "auc_null_sd"])
    one = bench(synth(tmp_path / "one", per_type=8, n_bins=40, n_null=3),
                "pool:10mb", null_draws="1")
    assert one.loc["pool:10mb", "n_null_draws"] == 1
