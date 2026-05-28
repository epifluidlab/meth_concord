"""Unit tests for the shared blacklist module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nome_hic_concordance import blacklist as bl


def _write_bed(tmp_path, lines: list[str]):
    p = tmp_path / "bl.bed"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_load_blacklist_merges_overlapping(tmp_path):
    path = _write_bed(tmp_path, [
        "chr1\t0\t1500",
        "chr1\t1000\t2500",   # overlaps previous
        "chr1\t5000\t6000",   # disjoint
        "chr2\t0\t1000",
    ])
    out = bl.load_blacklist_bed(path)
    assert out["chr1"].tolist() == [[0, 2500], [5000, 6000]]
    assert out["chr2"].tolist() == [[0, 1000]]


def test_load_blacklist_skips_comment_and_track(tmp_path):
    path = _write_bed(tmp_path, [
        "# header comment",
        "track name=foo",
        "browser hide all",
        "chr1\t100\t200",
    ])
    out = bl.load_blacklist_bed(path)
    assert out["chr1"].tolist() == [[100, 200]]


def test_interval_overlaps_blacklist_basic():
    blacklist = {"chr1": np.array([[100, 200], [300, 400]], dtype=np.int64)}
    # Overlaps [100, 200)
    assert bl.interval_overlaps_blacklist("chr1", 150, 250, blacklist)
    # Touches at right edge (bin_end == start of next): NOT overlapping.
    assert not bl.interval_overlaps_blacklist("chr1", 0, 100, blacklist)
    # Touches at left edge of bin (bin_start == end of prev): NOT overlapping.
    assert not bl.interval_overlaps_blacklist("chr1", 200, 300, blacklist)
    # Fully contains an interval
    assert bl.interval_overlaps_blacklist("chr1", 50, 250, blacklist)
    # Unknown chromosome: never overlaps.
    assert not bl.interval_overlaps_blacklist("chrZ", 0, 1000, blacklist)


def test_filter_read_pairs_drops_pos1_or_pos2_hits():
    blacklist = {"chr1": np.array([[1000, 2000]], dtype=np.int64)}
    df = pd.DataFrame({
        "chrom": ["chr1"] * 5,
        "pos1":  [500, 1500, 2500,  900, 1999],
        "pos2":  [800, 2500, 2600, 1500, 5000],
        "n1":    [3, 3, 3, 3, 3],
        "n2":    [3, 3, 3, 3, 3],
    })
    out, stats = bl.filter_read_pairs_blacklist(df, blacklist)
    # Row 0: clean.  Row 1: pos1=1500 in BL.  Row 2: clean (both outside).
    # Row 3: pos2=1500 in BL.  Row 4: pos1=1999 in BL.
    assert out["pos1"].tolist() == [500, 2500]
    assert out["pos2"].tolist() == [800, 2600]
    assert stats["n_in"] == 5
    assert stats["n_out"] == 2
    assert stats["n_dropped_either"] == 3
    assert stats["n_dropped_pos1"] == 2  # rows 1, 4 (pos1 in BL)
    assert stats["n_dropped_pos2"] == 1  # only row 3 (pos2=1500 in BL)


def test_filter_read_pairs_unknown_chromosome_pass_through():
    blacklist = {"chr1": np.array([[100, 200]], dtype=np.int64)}
    df = pd.DataFrame({
        "chrom": ["chr2", "chr3"],
        "pos1":  [150, 1500],
        "pos2":  [180, 1600],
    })
    out, stats = bl.filter_read_pairs_blacklist(df, blacklist)
    assert len(out) == 2
    assert stats["n_dropped_either"] == 0


def test_filter_read_pairs_empty_input():
    blacklist = {"chr1": np.array([[100, 200]], dtype=np.int64)}
    df = pd.DataFrame({"chrom": [], "pos1": [], "pos2": []})
    out, stats = bl.filter_read_pairs_blacklist(df, blacklist)
    assert len(out) == 0
    assert stats["n_in"] == 0 and stats["n_out"] == 0


def test_filter_read_pairs_no_blacklist():
    df = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "pos1":  [100, 200],
        "pos2":  [150, 250],
    })
    out, stats = bl.filter_read_pairs_blacklist(df, {})
    assert len(out) == 2
    assert stats["n_dropped_either"] == 0


def test_filter_position_at_interval_start_is_inside():
    """A position equal to a blacklist start is inside the interval."""
    blacklist = {"chr1": np.array([[1000, 2000]], dtype=np.int64)}
    df = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "pos1":  [1000, 999],   # 1000 is INSIDE [1000, 2000); 999 is outside.
        "pos2":  [50, 50],
    })
    out, stats = bl.filter_read_pairs_blacklist(df, blacklist)
    assert out["pos1"].tolist() == [999]


def test_filter_position_at_interval_end_is_outside():
    """A position equal to a blacklist end is outside (half-open semantics)."""
    blacklist = {"chr1": np.array([[1000, 2000]], dtype=np.int64)}
    df = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "pos1":  [2000, 1999],  # 2000 is OUTSIDE [1000, 2000); 1999 is inside.
        "pos2":  [50, 50],
    })
    out, stats = bl.filter_read_pairs_blacklist(df, blacklist)
    assert out["pos1"].tolist() == [2000]


def test_infer_bin_size():
    cols = pd.Index([
        "chr1:0", "chr1:250000", "chr1:500000",
        "chr2:0", "chr2:250000",
    ])
    assert bl.infer_bin_size(cols) == 250000


def test_infer_bin_size_returns_none_when_singletons():
    cols = pd.Index(["chr1:0", "chr2:0"])
    assert bl.infer_bin_size(cols) is None
