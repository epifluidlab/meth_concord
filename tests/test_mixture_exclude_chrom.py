"""chrM must not enter the beta-binomial fit.

The model describes a mixture of accessible and inaccessible states, which is
a statement about nucleosomes.  chrM has none, so its read ends are not draws
from that mixture at all.  Letting them in moves the fitted components, and
because every posterior is computed under those components, it moves the
posterior of every NUCLEAR site too -- turning "add a reference contig" into
"silently change every existing result".

The reference still gets posteriors; it is only kept out of the FIT.  That is
what makes it comparable to the bins it is used to correct.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance import mixture_model as mm
from nome_hic_concordance.cli import _mixture_exclude_chroms


def _parquet(path, n_nuclear=4000, n_mito=4000, seed=0):
    """Nuclear ends from a two-component mixture; chrM ends nearly all
    methylated, as a nucleosome-free template would be."""
    # Independent RNG streams per contig, so the NUCLEAR rows are identical
    # whether or not chrM is present.  Sharing one stream would make the
    # fixtures differ in the very rows the tests claim are untouched.
    nrng = np.random.default_rng(seed)
    mrng = np.random.default_rng(seed + 9973)
    n_sites = 6

    def _block(chrom, n, p, rng):
        if n == 0:
            return None
        k = (np.concatenate([rng.binomial(n_sites, 0.05, n // 2),
                             rng.binomial(n_sites, 0.55, n - n // 2)])
             if p is None else rng.binomial(n_sites, p, n))
        return pd.DataFrame({
            "chrom": chrom,
            "pos1": rng.integers(0, 10 ** 7, n),
            "pos2": rng.integers(0, 10 ** 7, n),
            "k1": k, "n1": n_sites, "k2": k[::-1], "n2": n_sites,
            "distance": rng.integers(1, 5000, n),
        })

    blocks = [b for b in (_block("chr1", n_nuclear, None, nrng),
                          _block("chrM", n_mito, 0.92, mrng)) if b is not None]
    pd.concat(blocks, ignore_index=True).to_parquet(path)
    return str(path)


def _subsample(paths, **kw):
    kw.setdefault("max_subsample", 0)
    kw.setdefault("short_range_max", 1_000)
    kw.setdefault("min_distance", 20_000)
    return mm.stream_subsample_from_parquets(paths, **kw)[0]


def test_without_the_exclusion_chrM_ends_enter_the_pool(tmp_path):
    """The failure mode, stated plainly, so the fix has something to fix."""
    paths = {"c0": _parquet(tmp_path / "c0.parquet")}
    k, n = _subsample(paths)["all"]
    # chrM is 92% methylated; its presence drags the pooled mean up
    assert k.mean() / n.mean() > 0.4


def test_excluding_chrM_leaves_only_the_nuclear_ends(tmp_path):
    paths = {"c0": _parquet(tmp_path / "c0.parquet")}
    k_all, _ = _subsample(paths)["all"]
    k_nuc, _ = _subsample(paths, exclude_chroms={"M"})["all"]
    assert len(k_nuc) < len(k_all)
    assert k_nuc.mean() < k_all.mean()


def test_the_nuclear_pool_is_identical_to_a_run_without_chrM(tmp_path):
    """The guarantee that makes this safe to run on existing data: adding
    chrM and excluding it from the fit must leave the fit seeing exactly what
    it saw before chrM existed."""
    with_mito = {"c0": _parquet(tmp_path / "a.parquet", n_mito=4000, seed=3)}
    without = {"c0": _parquet(tmp_path / "b.parquet", n_mito=0, seed=3)}
    a_k, a_n = _subsample(with_mito, exclude_chroms={"M"})["all"]
    b_k, b_n = _subsample(without)["all"]
    np.testing.assert_array_equal(np.sort(a_k), np.sort(b_k))
    np.testing.assert_array_equal(np.sort(a_n), np.sort(b_n))


def test_chrom_naming_variants_all_exclude_the_same_contig(tmp_path):
    from nome_hic_concordance.utils import chrom_key
    paths = {"c0": _parquet(tmp_path / "c0.parquet")}
    sizes = {len(_subsample(paths, exclude_chroms={chrom_key(nm)})["all"][0])
             for nm in ("chrM", "M", "MT")}
    assert len(sizes) == 1


def test_excluding_a_contig_that_is_absent_changes_nothing(tmp_path):
    paths = {"c0": _parquet(tmp_path / "c0.parquet")}
    base, _ = _subsample(paths)["all"]
    same, _ = _subsample(paths, exclude_chroms={"22"})["all"]
    np.testing.assert_array_equal(np.sort(base), np.sort(same))


def test_a_fitted_model_moves_when_chrM_is_left_in(tmp_path):
    """Not just the pool -- the parameters someone reads off the fit."""
    paths = {"c0": _parquet(tmp_path / "c0.parquet", seed=5)}
    contaminated = mm.fit_and_compare_from_parquets(
        paths, output_dir=str(tmp_path / "bad"), max_subsample=0,
        n_restarts=1, max_iter=50)["models"]["all"]
    clean = mm.fit_and_compare_from_parquets(
        paths, output_dir=str(tmp_path / "good"), max_subsample=0,
        n_restarts=1, max_iter=50, exclude_chroms={"M"})["models"]["all"]
    a = contaminated.to_dict() if hasattr(contaminated, "to_dict") else {}
    b = clean.to_dict() if hasattr(clean, "to_dict") else {}
    assert a and b, "model should expose its parameters"
    assert a != b, "chrM in the fit must visibly move the components"


def _args(**kw):
    kw.setdefault("include_chrom", None)
    kw.setdefault("mixture_exclude_chrom", None)
    return argparse.Namespace(**kw)


def test_including_chrM_excludes_it_from_the_fit_without_a_second_flag():
    assert _mixture_exclude_chroms(_args(include_chrom=["chrM"])) == {"M"}


def test_a_run_that_adds_no_contig_fits_on_everything_as_before():
    assert _mixture_exclude_chroms(_args()) == set()


def test_fitting_on_everything_remains_possible_explicitly():
    assert _mixture_exclude_chroms(
        _args(include_chrom=["chrM"],
              mixture_exclude_chrom=["none"])) == set()


def test_the_subsample_draw_is_unchanged_too_not_just_the_pool(tmp_path):
    """Re-running step 3 on data that now contains chrM must reproduce the
    ORIGINAL fit, or 'safe to re-run' is not true.  The pool being equal is
    not sufficient: the subsample is drawn from a seeded RandomState consumed
    cell by cell, so the draw is only identical if chrM was filtered before
    the sampler ever saw it."""
    with_mito = {f"c{i}": _parquet(tmp_path / f"a{i}.parquet", n_mito=3000,
                                   seed=10 + i) for i in range(3)}
    without = {f"c{i}": _parquet(tmp_path / f"b{i}.parquet", n_mito=0,
                                 seed=10 + i) for i in range(3)}
    a = _subsample(with_mito, max_subsample=2000, exclude_chroms={"M"})
    b = _subsample(without, max_subsample=2000)
    for subset in ("all", "short", "long"):
        np.testing.assert_array_equal(a[subset][0], b[subset][0])
        np.testing.assert_array_equal(a[subset][1], b[subset][1])
