"""Per-cell reference concordance on a chromosome with no chromatin.

The correction this feeds divides out each cell's assay reliability.  That is
only sound if the reference is measured the SAME way in every cell, so the
properties pinned here are mostly about refusing to estimate it differently:
a fixed pair count in every cell (Pearson r is biased toward zero in small
samples, and mtDNA copy number differs severalfold between cell types), and a
loud NaN rather than a number when a cell cannot meet it.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "reference_concordance.py")


def _mod():
    spec = importlib.util.spec_from_file_location("refconc", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["refconc"] = m
    spec.loader.exec_module(m)
    return m


def _posteriors(n_pairs=3000, chrom="chrM", rho=0.6, seed=0, n_sites=6,
                n_other=500):
    """A posteriors table: correlated ends on the reference chromosome, plus
    unrelated nuclear rows that must not leak in."""
    rng = np.random.default_rng(seed)
    shared = rng.normal(0, 1, n_pairs)
    p1 = rho * shared + np.sqrt(1 - rho ** 2) * rng.normal(0, 1, n_pairs)
    p2 = rho * shared + np.sqrt(1 - rho ** 2) * rng.normal(0, 1, n_pairs)
    ref = pd.DataFrame({
        "chrom": chrom, "pos1": rng.integers(0, 16000, n_pairs),
        "pos2": rng.integers(0, 16000, n_pairs),
        "distance": rng.integers(1, 16000, n_pairs),
        "n1": n_sites, "n2": n_sites,
        "posterior1": 1 / (1 + np.exp(-p1)), "posterior2": 1 / (1 + np.exp(-p2))})
    nuc = pd.DataFrame({
        "chrom": "chr1", "pos1": rng.integers(0, 10 ** 8, n_other),
        "pos2": rng.integers(0, 10 ** 8, n_other),
        "distance": rng.integers(1000, 10 ** 6, n_other),
        "n1": n_sites, "n2": n_sites,
        "posterior1": rng.random(n_other), "posterior2": rng.random(n_other)})
    return pd.concat([ref, nuc], ignore_index=True)


def test_chrom_naming_variants_all_match():
    """A reference that silently matches nothing is worse than one that
    fails: BAM headers say 'MT' where a 2bit says 'chrM'."""
    m = _mod()
    assert {m.canon_chrom(c) for c in ("chrM", "M", "MT", "chrMT")} == {"M"}
    assert m.canon_chrom("chr1") == "1" and m.canon_chrom("1") == "1"


def test_nuclear_rows_do_not_leak_into_the_reference():
    m = _mod()
    df = _posteriors(n_pairs=500, n_other=900)
    assert len(m.reference_pairs(df, "chrM")) == 500
    assert len(m.reference_pairs(df, "MT")) == 500      # same sequence


def test_short_ends_are_dropped_to_match_the_nuclear_precision():
    """Attenuation depends on sites per read end, so the reference has to be
    read at the same --min-gch-sites as the nuclear run."""
    m = _mod()
    df = _posteriors(n_pairs=400)
    df.loc[:99, "n1"] = 1
    assert len(m.reference_pairs(df, "chrM", min_gch_sites=3)) == 300


def test_every_cell_uses_the_same_number_of_pairs():
    """The design guarantee.  Pearson r is biased toward zero in small
    samples, and mtDNA copy number differs severalfold between cell types --
    so a free pair count would make the reference differently biased in each
    arm, injecting the artefact it exists to remove."""
    m = _mod()
    # ONE source, so the only difference is how much of it each cell has --
    # which is what mtDNA copy number varies in the real data
    full = _posteriors(n_pairs=40000, seed=1)
    ref_rows = full[full["chrom"] == "chrM"]
    thin_df = pd.concat([ref_rows.head(3000), full[full["chrom"] != "chrM"]],
                        ignore_index=True)
    thin = m.reference_d(thin_df, "cell", 2000)
    deep = m.reference_d(full, "cell", 2000)
    assert thin["n_pairs_used"] == deep["n_pairs_used"] == 2000
    assert thin["n_pairs_available"] == 3000
    assert deep["n_pairs_available"] == 40000
    # 13x the mtDNA, same estimate: the copy-number difference has been
    # taken out of the reference by construction, not modelled away
    assert thin["reference_d"] == pytest.approx(deep["reference_d"], abs=0.10)


def test_a_cell_short_of_the_floor_is_refused_not_estimated_from_fewer():
    m = _mod()
    r = m.reference_d(_posteriors(n_pairs=100), "thin", 2000)
    assert np.isnan(r["reference_d"])
    assert r["n_pairs_used"] == 0
    assert "100" in r["note"] and "2000" in r["note"]


def test_a_degenerate_reference_is_nan_with_a_reason():
    """var_A ~ 0 means the molecules do not differ, so there is no
    reliability to measure.  Dividing the genome by that is silently wrong,
    so it must not come back as a number."""
    m = _mod()
    df = _posteriors(n_pairs=800)
    df.loc[df["chrom"] == "chrM", "posterior1"] = 0.5
    r = m.reference_d(df, "flat", 500)
    assert np.isnan(r["reference_d"])
    assert "variance" in r["note"]


def test_the_subsample_is_reproducible_and_cell_specific():
    m = _mod()
    df = _posteriors(n_pairs=2000)
    a = m.reference_d(df, "cellA", 400, seed=7)["reference_d"]
    again = m.reference_d(df, "cellA", 400, seed=7)["reference_d"]
    other = m.reference_d(df, "cellB", 400, seed=7)["reference_d"]
    assert a == again                       # same cell, same draw
    assert a != other                       # seeded from the NAME, so cells
                                            # do not share one draw
    assert a != m.reference_d(df, "cellA", 400, seed=8)["reference_d"]


def test_it_recovers_a_known_correlation():
    m = _mod()
    r = m.reference_d(_posteriors(n_pairs=8000, rho=0.6, seed=3), "c", 4000)
    # the logistic squash weakens it, but it must track rho, not drift free
    assert 0.3 < r["reference_d"] < 0.75


def test_cli_writes_a_per_cell_table(tmp_path):
    m = _mod()
    d = tmp_path / "04_posteriors"
    d.mkdir()
    for i, n in enumerate([3000, 3000, 50]):        # the last is too thin
        _posteriors(n_pairs=n, seed=i).to_parquet(
            d / f"cell{i}.posteriors.parquet")
    out = tmp_path / "ref.tsv"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--posteriors", str(d),
         "--n-pairs", "1000", "--output", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t = pd.read_csv(out, sep="\t").set_index("cell")
    assert list(t.index) == ["cell0", "cell1", "cell2"]
    assert t.loc["cell0", "reference_d"] > 0
    assert np.isnan(t.loc["cell2", "reference_d"])   # refused, not estimated
    assert "1 cells unusable" in r.stderr or "unusable" in r.stderr


def test_cli_warns_when_the_reference_is_degenerate(tmp_path):
    """A near-zero reference means there is almost no reliability signal to
    divide out; correcting by it would mostly add noise, and the run must say
    so rather than produce confident numbers."""
    d = tmp_path / "04_posteriors"
    d.mkdir()
    df = _posteriors(n_pairs=3000, rho=0.0, seed=5)
    df.to_parquet(d / "flat.posteriors.parquet")
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--posteriors", str(d),
         "--n-pairs", "1000", "--output", str(tmp_path / "r.tsv")],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "near zero" in r.stderr


def test_a_deduplicated_reference_is_detected_and_called_out(tmp_path):
    """The failure this catches is silent by construction.  A dedup'd chrM
    still yields a number, and a plausible one -- but the between-molecule
    variance it is supposed to measure has been removed by keeping one read
    per coordinate.  Absence of coordinate collisions on 16.6 kb carried in
    thousands of copies is the giveaway."""
    d = tmp_path / "04_posteriors"
    d.mkdir()
    df = _posteriors(n_pairs=3000, seed=1)
    ref = df["chrom"] == "chrM"
    # every pair at its own coordinate: what markdup leaves behind
    df.loc[ref, "pos1"] = np.arange(int(ref.sum())) * 5
    df.loc[ref, "pos2"] = np.arange(int(ref.sum())) * 5 + 3
    df.to_parquet(d / "dedup.posteriors.parquet")
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--posteriors", str(d),
         "--n-pairs", "1000", "--output", str(tmp_path / "r.tsv")],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "--keep-duplicates-chrom" in r.stderr
    # worded as a NOTE, not a verdict: a library that simply never collides
    # looks identical from here, and claiming otherwise would be wrong
    assert "cannot be told" in r.stderr
    assert pd.read_csv(tmp_path / "r.tsv", sep="\t"
                       )["frac_repeated_coords"].iloc[0] == 0.0


def test_a_properly_parsed_reference_does_not_trip_the_warning(tmp_path):
    """mtDNA reads pile onto shared coordinates; that is the healthy case
    and must not be reported as a problem."""
    d = tmp_path / "04_posteriors"
    d.mkdir()
    df = _posteriors(n_pairs=3000, seed=2)
    ref = df["chrom"] == "chrM"
    rng = np.random.default_rng(0)
    n = int(ref.sum())
    df.loc[ref, "pos1"] = rng.integers(0, 60, n)       # heavy collisions
    df.loc[ref, "pos2"] = rng.integers(0, 60, n)
    df.to_parquet(d / "ok.posteriors.parquet")
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--posteriors", str(d),
         "--n-pairs", "1000", "--output", str(tmp_path / "r.tsv")],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "--keep-duplicates-chrom" not in r.stderr
    # comfortably clear of the threshold; the exact rate depends on library
    # prep, which is why the check only fires at essentially zero
    assert pd.read_csv(tmp_path / "r.tsv", sep="\t"
                       )["frac_repeated_coords"].iloc[0] > 0.05
