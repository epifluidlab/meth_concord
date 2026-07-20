"""--exclude-chrom, shared by every CLI.

One spelling has to work everywhere, because the tools disagree about naming:
a 2bit says 'chrM', a BAM header may say 'MT', and a matrix column says
'chr1:250000'.  An exclusion that silently matches nothing is worse than none
at all -- the run still looks like it worked -- so an unmatched value must warn.
"""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

from nome_hic_concordance.utils import (bin_label_chrom, chrom_excluded,
                                        chrom_key, parse_exclude_chroms,
                                        report_excluded)


# ------------------------------------------------------------------ parsing --

def test_comma_separated_and_repeated_both_work():
    """--exclude-chrom chrM,chrX --exclude-chrom chrY"""
    assert parse_exclude_chroms(["chrM,chrX", "chrY"]) == {"M", "X", "Y"}
    assert parse_exclude_chroms(["chrM", "chrX", "chrY"]) == {"M", "X", "Y"}
    assert parse_exclude_chroms(["chrM,chrX,chrY"]) == {"M", "X", "Y"}
    assert parse_exclude_chroms("chrM,chrY") == {"M", "Y"}


def test_empty_and_none_are_noops():
    for v in (None, [], "", [""], ["  "], [",", " , "]):
        assert parse_exclude_chroms(v) == set(), v


def test_naming_variants_are_interchangeable():
    """chrM / M / MT are one contig; chr1 / 1 are one chromosome."""
    for spelling in ("chrM", "M", "MT", "chrMT", "CHRM", " chrM "):
        assert chrom_key(spelling) == "M", spelling
    for spelling in ("chr1", "1", "CHR1"):
        assert chrom_key(spelling) == "1", spelling
    ex = parse_exclude_chroms(["MT"])
    assert chrom_excluded("chrM", ex) and chrom_excluded("M", ex)
    assert not chrom_excluded("chr1", ex)


def test_exclusion_does_not_catch_lookalike_names():
    ex = parse_exclude_chroms(["chrX"])
    assert chrom_excluded("chrX", ex)
    for other in ("chrX_random", "chr1", "chrXY", "chrM"):
        assert not chrom_excluded(other, ex), other


def test_bin_label_chrom_handles_matrix_columns():
    assert bin_label_chrom("chr1:250000") == "chr1"
    assert bin_label_chrom("chrX:0") == "chrX"
    assert bin_label_chrom("chr1") == "chr1"          # no offset present


def test_report_excluded_keeps_the_rest_and_flags_a_miss(caplog):
    ex = parse_exclude_chroms(["chrM,chrZZ"])
    with caplog.at_level("WARNING"):
        kept = report_excluded(ex, ["chr1", "chrM", "chrX"], "chromosome")
    assert kept == ["chr1", "chrX"]
    assert "ZZ" in caplog.text, "an unmatched exclusion must warn, not pass quietly"


def test_report_excluded_is_a_noop_without_exclusions():
    got = report_excluded(set(), ["chr1", "chrM"])
    assert got == ["chr1", "chrM"]


# ----------------------------------------------------------------- the CLIs --

@pytest.mark.parametrize("mod", ["cli", "comparison_cli", "compartments_cli"])
def test_every_cli_exposes_the_flag(mod):
    r = subprocess.run([sys.executable, "-m", f"nome_hic_concordance.{mod}",
                        "--help"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "--exclude-chrom" in r.stdout, mod
    assert "autosomes" in r.stdout, f"{mod}: help should name the common use"


@pytest.mark.parametrize("script", ["site_distance_correlation.py",
                                    "contact_concordance.py"])
def test_every_script_exposes_the_flag(script):
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "scripts" / script
    r = subprocess.run([sys.executable, str(p), "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "--exclude-chrom" in r.stdout, script


def test_matrix_bin_columns_filter_by_chromosome():
    """What nome_hic_compare actually does: drop bins on excluded contigs."""
    df = pd.DataFrame(
        {"chr1:0": [1.0], "chr1:250000": [2.0], "chrX:0": [3.0],
         "chrM:0": [4.0], "chrY:500000": [5.0]})
    ex = parse_exclude_chroms(["chrM,chrX,chrY"])
    keep = [c for c in df.columns
            if not chrom_excluded(bin_label_chrom(c), ex)]
    assert keep == ["chr1:0", "chr1:250000"]


def test_context_tables_filter_by_chromosome():
    """What the site-level scripts do: drop whole contigs from the site lists."""
    import importlib.util
    from pathlib import Path
    import numpy as np
    p = (Path(__file__).resolve().parent.parent / "scripts"
         / "site_distance_correlation.py")
    spec = importlib.util.spec_from_file_location("sdc_ex", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["sdc_ex"] = m
    spec.loader.exec_module(m)

    ctx = {"GCH": {"chr1": np.arange(9), "chrM": np.arange(4),
                   "chrX": np.arange(6)},
           "HCG": {"chr1": np.arange(5), "MT": np.arange(2)}}
    out, removed = m._drop_chroms(ctx, ["chrM,chrX"])
    assert sorted(out["GCH"]) == ["chr1"]
    assert sorted(out["HCG"]) == ["chr1"], "MT must match chrM"
    assert removed == {"GCH:chrM": 4, "GCH:chrX": 6, "HCG:MT": 2}
