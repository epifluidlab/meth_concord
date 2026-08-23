"""Per-contrast up / down / ns BEDs, and why the background matters.

An enrichment run needs a background.  Using every bin in the genome makes
the comparison partly a measure of TESTABLE versus UNTESTABLE sequence:
bins that never had a statistic are not a random sample, they sit in poor
mappability, repeats and pericentromeric DNA, and those carry exactly the
annotations an enrichment is looking for.  The fix is a background of bins
that WERE measured and did not change.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "celltype_marker_bins.py")

N_DEAD, N_UP, N_DOWN, N_BINS = 10, 12, 12, 70


def _matrix(seed, n_per=14):
    """CT0 against CT1/CT2, with dead bins, up bins and down bins planted."""
    rng = np.random.default_rng(seed)
    cts = ["CT0"] * n_per + ["CT1"] * n_per + ["CT2"] * n_per
    n_cells = len(cts)
    X = rng.normal(0.40, 0.03, (n_cells, N_BINS))
    inarm = np.array([c == "CT0" for c in cts])
    up = slice(N_DEAD, N_DEAD + N_UP)
    down = slice(N_DEAD + N_UP, N_DEAD + N_UP + N_DOWN)
    X[np.ix_(inarm, np.arange(N_BINS)[up])] += 0.30
    X[np.ix_(inarm, np.arange(N_BINS)[down])] -= 0.30
    # never testable: no cell has a value at all
    X[:, :N_DEAD] = np.nan
    return X, cts


def _run(tmp_path, extra=None, seed=0):
    X, cts = _matrix(seed)
    samples = [f"cell{i}" for i in range(len(cts))]
    bins = [f"chr1:{i * 250000}" for i in range(N_BINS)]
    out = tmp_path / "out"
    args = [sys.executable, str(_SCRIPT), "--samples", str(tmp_path / "s.tsv"),
            "--cell-type-column", "cell_type", "--batch-column", "sample",
            "--n-bins", "20", "--n-permutations", "20",
            "--output-dir", str(out), "--plot-format", "png",
            "--contrast", "mening_vs_mono=CT0",
            "--min-cells-per-group", "3"]
    for name in ("gch", "hcg"):
        mat = tmp_path / f"{name}.parquet"
        pd.DataFrame(X, index=samples, columns=bins).to_parquet(mat)
        args += ["--matrix", f"{name}={mat}"]
    pd.DataFrame({"cell_name": samples, "bam_path": ["x"] * len(cts),
                  "cell_type": cts,
                  "sample": ["b1" if i % 2 else "b2"
                             for i in range(len(cts))]}
                 ).to_csv(tmp_path / "s.tsv", sep="\t", index=False)
    r = subprocess.run(args + (extra or []), capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    return out, r


def _bed(out, name, suffix):
    p = out / f"bin_scores_{name}.contrast_mening_vs_mono.{suffix}.bed"
    assert p.exists(), f"missing {p.name}"
    return pd.read_csv(p, sep="\t")


def _starts(df):
    return set(df["start"] // 250000)


def test_all_three_files_are_written_for_both_modalities(tmp_path):
    """The request was explicitly for HCG as well as GCH."""
    out, _ = _run(tmp_path)
    for name in ("gch", "hcg"):
        for suffix in ("up", "down", "ns"):
            assert _bed(out, name, suffix) is not None


def test_up_and_down_carry_the_sign_their_names_claim(tmp_path):
    """diff is mean_in - mean_out, so up means higher in the contrast group."""
    out, _ = _run(tmp_path)
    up = _bed(out, "gch", "up")
    down = _bed(out, "gch", "down")
    col = "contrast_mening_vs_mono_diff"
    assert len(up) and len(down)
    assert (up[col] > 0).all()
    assert (down[col] < 0).all()


def test_the_planted_bins_land_in_the_right_file(tmp_path):
    out, _ = _run(tmp_path)
    planted_up = set(range(N_DEAD, N_DEAD + N_UP))
    planted_down = set(range(N_DEAD + N_UP, N_DEAD + N_UP + N_DOWN))
    found_up = _starts(_bed(out, "gch", "up"))
    found_down = _starts(_bed(out, "gch", "down"))
    # every planted bin is recovered, and none is filed under the opposite
    # sign.  Not asserted as a strict subset: at FDR 0.05 over 70 bins a
    # couple of noise bins are expected to pass, and pinning their absence
    # would make the test a random-seed check rather than a direction check.
    assert planted_up <= found_up
    assert planted_down <= found_down
    assert not (planted_up & found_down)
    assert not (planted_down & found_up)


def test_the_ns_background_excludes_bins_that_were_never_testable(tmp_path):
    """The whole point.  A bin with no statistic was never measured, and
    untestable bins are not a random sample of the genome."""
    out, _ = _run(tmp_path)
    dead = set(range(N_DEAD))
    for name in ("gch", "hcg"):
        assert not (_starts(_bed(out, name, "ns")) & dead)


def test_the_all_bins_file_does_contain_them_which_is_why_it_is_wrong(tmp_path):
    """Stated as a test so the reason for the new files cannot quietly
    stop being true."""
    out, _ = _run(tmp_path)
    allb = pd.read_csv(out / "bin_scores_gch.all.bed", sep="\t")
    assert set(allb["start"] // 250000) & set(range(N_DEAD))


def test_foreground_and_background_are_disjoint(tmp_path):
    out, _ = _run(tmp_path)
    for name in ("gch", "hcg"):
        ns = _starts(_bed(out, name, "ns"))
        assert not (ns & _starts(_bed(out, name, "up")))
        assert not (ns & _starts(_bed(out, name, "down")))


def test_up_and_down_together_are_the_selected_set(tmp_path):
    out, _ = _run(tmp_path)
    sel = pd.read_csv(
        out / "bin_scores_gch.contrast_mening_vs_mono.bed", sep="\t")
    both = _starts(_bed(out, "gch", "up")) | _starts(_bed(out, "gch", "down"))
    assert both == set(sel["start"] // 250000)


def test_the_run_says_what_it_wrote_and_how_to_use_it(tmp_path):
    out, r = _run(tmp_path)
    assert "up /" in r.stderr and "down /" in r.stderr and "ns" in r.stderr
    assert "never testable" in r.stderr
    assert "--background-bed" in r.stderr


def test_a_tiny_background_is_flagged(tmp_path):
    """An enrichment against a handful of bins has no resolution, and that
    is worth saying rather than letting the odds ratio look authoritative."""
    out, r = _run(tmp_path, extra=["--contrast-fdr", "0.9999"])
    assert "almost no resolution" in r.stderr


# ------------------------------------------------------------- the universe --

def test_tested_bed_is_written_for_both_modalities(tmp_path):
    out, _ = _run(tmp_path)
    for name in ("gch", "hcg"):
        assert len(_bed(out, name, "tested")) > 0


def test_tested_excludes_the_bins_that_never_had_a_statistic(tmp_path):
    """Its job is to be the universe of what the contrast COULD have called,
    which is the right pool to draw a matched random control from -- matching
    against all genomic bins would sample sequence no contrast could report."""
    out, _ = _run(tmp_path)
    for name in ("gch", "hcg"):
        assert not (_starts(_bed(out, name, "tested")) & set(range(N_DEAD)))


def test_tested_contains_up_down_and_ns(tmp_path):
    out, _ = _run(tmp_path)
    for name in ("gch", "hcg"):
        t = _starts(_bed(out, name, "tested"))
        for suffix in ("up", "down", "ns"):
            assert _starts(_bed(out, name, suffix)) <= t, suffix


def test_tested_may_exceed_up_down_ns_by_the_filtered_bins(tmp_path):
    """A superset, deliberately: bins significant at FDR but dropped by a
    later filter belong to the universe even though they are in neither the
    foreground nor the background."""
    out, _ = _run(tmp_path, extra=["--contrast-min-diff", "0.25"])
    t = _starts(_bed(out, "gch", "tested"))
    three = set().union(*(_starts(_bed(out, "gch", s))
                          for s in ("up", "down", "ns")))
    assert three <= t


def test_the_two_background_choices_are_both_offered_and_explained(tmp_path):
    out, r = _run(tmp_path)
    assert "tested" in r.stderr
    assert "measured-and-unchanged" in r.stderr
    assert "both" in r.stderr.lower()
