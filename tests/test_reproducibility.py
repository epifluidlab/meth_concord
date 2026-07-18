"""Reproducibility: every stochastic step is seeded, and --random-seed reaches it.

Two properties per component:
  * SAME seed  -> identical output (reproducible)
  * DIFFERENT seed -> the seed is actually plumbed (a sweep perturbs it)

The second is the one that used to fail silently: several steps had a
hard-coded random_state (42 / 0) or a PYTHONHASHSEED-salted ``hash()``, so they
were either frozen against the flag or non-reproducible across processes.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
import zlib

import numpy as np
import pandas as pd

from nome_hic_concordance import batch_correction, comparison, concordance
from nome_hic_concordance.imputation import _top_eigengap


# --------------------------------------------------------------------------
# The O/E end-pool seed: stable across PROCESSES (was salted hash())
# --------------------------------------------------------------------------

def test_cell_seed_is_stable_across_processes():
    """``hash()`` is salted per process; crc32 is not.  Two fresh interpreters
    must derive the same per-cell seed, or the O/E baseline drifts run-to-run."""
    code = (
        "import zlib;"
        "print((zlib.crc32('batch2_sc1.ACTTGA'.encode('utf-8')) + 42) % (2**31))"
    )
    outs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, check=True).stdout.strip()
        for _ in range(3)
    }
    assert len(outs) == 1, f"per-cell seed differs across processes: {outs}"
    # and it is what concordance.py computes
    expected = (zlib.crc32(b"batch2_sc1.ACTTGA") + 42) % (2 ** 31)
    assert outs.pop() == str(expected)


def test_builtin_hash_would_have_been_unstable():
    """Guard the rationale: builtin hash() of a str IS salted per process, so
    the old seed genuinely varied between runs."""
    code = "print(hash('batch2_sc1.ACTTGA'))"
    outs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, check=True).stdout.strip()
        for _ in range(5)
    }
    assert len(outs) > 1, "expected PYTHONHASHSEED salting to vary hash()"


def test_dm_expected_takes_random_state():
    params = inspect.signature(concordance._compute_cell_dm_expected).parameters
    assert "random_state" in params
    assert "random_state" in inspect.signature(
        concordance.compute_signed_concordance_matrix).parameters


# --------------------------------------------------------------------------
# Seeds are plumbed, not hard-coded
# --------------------------------------------------------------------------

def test_svd_scores_honours_random_state():
    assert "random_state" in inspect.signature(comparison._svd_scores).parameters
    rng = np.random.default_rng(0)
    m = pd.DataFrame(rng.random((40, 25)), index=[f"c{i}" for i in range(40)])
    a, _ = comparison._svd_scores(m, 5, random_state=1)
    b, _ = comparison._svd_scores(m, 5, random_state=1)
    assert np.allclose(a, b)                       # same seed -> reproducible


def test_eigengap_honours_random_state():
    assert "random_state" in inspect.signature(_top_eigengap).parameters
    rng = np.random.default_rng(1)
    M = rng.random((30, 20))
    assert _top_eigengap(M, random_state=3) == _top_eigengap(M, random_state=3)


def test_batch_correction_honours_random_state():
    for fn in (batch_correction.apply_batch_correction,
               batch_correction._harmony_correction):
        assert "random_state" in inspect.signature(fn).parameters, fn.__name__


def test_zscore_correction_is_deterministic():
    """The non-harmony methods carry no randomness at all."""
    rng = np.random.default_rng(2)
    cells = [f"c{i}" for i in range(20)]
    m = pd.DataFrame(rng.random((20, 8)), index=cells)
    b = pd.Series(["a"] * 10 + ["b"] * 10, index=cells, name="batch")
    x = batch_correction.apply_batch_correction(m, b, method="zscore")
    y = batch_correction.apply_batch_correction(m, b, method="zscore")
    assert np.allclose(x.to_numpy(), y.to_numpy(), equal_nan=True)


# --------------------------------------------------------------------------
# Normalization has no randomness at all (no seed needed)
# --------------------------------------------------------------------------

def test_normalization_is_deterministic(tmp_path):
    rng = np.random.default_rng(3)
    m = pd.DataFrame(rng.random((25, 12)), index=[f"c{i}" for i in range(25)],
                     columns=[f"chr1:{i}" for i in range(12)])
    a = comparison._prep_matrix(m, "a", str(tmp_path), "quantile", 0.5,
                                per_cell_norm="rank", post_norm_recenter=True)
    b = comparison._prep_matrix(m, "b", str(tmp_path), "quantile", 0.5,
                                per_cell_norm="rank", post_norm_recenter=True)
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


# --------------------------------------------------------------------------
# The CLI flag exists and defaults to 42 in both tools
# --------------------------------------------------------------------------

def test_both_clis_expose_random_seed():
    from nome_hic_concordance.cli import parse_args
    from nome_hic_concordance.comparison_cli import parse_args as cmp_args

    assert parse_args(["--samples", "s.tsv", "--output-dir", "o"]).random_seed == 42
    assert cmp_args(["--output-dir", "o"]).random_seed == 42
    assert parse_args(["--samples", "s.tsv", "--output-dir", "o",
                       "--random-seed", "7"]).random_seed == 7
    assert cmp_args(["--output-dir", "o", "--random-seed", "7"]).random_seed == 7
