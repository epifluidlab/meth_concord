"""Density of bin sets around domain boundaries."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "boundary_profile.py")


def _mod():
    spec = importlib.util.spec_from_file_location("bp", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["bp"] = m
    spec.loader.exec_module(m)
    return m


def _bed(path, rows, header=True):
    df = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    df.to_csv(path, sep="\t", index=False, header=header)
    return str(path)


def _grid(n, chrom="chr1", step=250_000):
    return pd.DataFrame({"chrom": [chrom] * n,
                         "start": np.arange(n) * step,
                         "end": (np.arange(n) + 1) * step})


# ------------------------------------------------------------- reading -----

def test_a_bin_grid_is_not_merged(tmp_path):
    """Every bin abuts its neighbour, so merging collapses a chromosome into
    one interval -- silently, leaving a universe of one row per contig."""
    m = _mod()
    p = _bed(tmp_path / "u.bed", [("chr1", i * 250_000, (i + 1) * 250_000)
                                  for i in range(5)])
    assert len(m.read_bed(p)) == 5
    assert len(m.read_bed(p, merge=True)) == 1


def test_domains_are_merged_when_asked(tmp_path):
    """Touching domains are one domain and their shared edge is not a
    boundary."""
    m = _mod()
    p = _bed(tmp_path / "d.bed", [("chr1", 0, 100), ("chr1", 100, 200),
                                  ("chr1", 500, 600)])
    got = m.read_bed(p, merge=True)
    assert len(got) == 2
    assert list(got.end) == [200, 600]


def test_a_headerless_bed_is_read(tmp_path):
    m = _mod()
    p = _bed(tmp_path / "n.bed", [("chr1", 0, 100)], header=False)
    assert len(m.read_bed(p)) == 1


# ------------------------------------------------------------ distance ----

def test_distance_is_positive_inside_and_negative_outside():
    """Both edges of a domain are boundaries; folding them without flipping
    the sign at one cancels the asymmetry the plot exists for."""
    m = _mod()
    bins = _grid(8, step=100)      # centres 50, 150, ... 750
    dom = pd.DataFrame({"chrom": ["chr1"], "start": [300], "end": [500]})
    d = m.signed_distance(bins, dom)
    assert d[1] < 0, "centre 150 is outside"
    assert d[3] > 0, "centre 350 is inside"
    assert d[4] > 0, "centre 450 is inside"
    assert d[6] < 0, "centre 650 is outside"


def test_distance_is_to_the_nearest_edge():
    m = _mod()
    bins = pd.DataFrame({"chrom": ["chr1"], "start": [380], "end": [420]})
    dom = pd.DataFrame({"chrom": ["chr1"], "start": [300], "end": [500]})
    d = m.signed_distance(bins, dom)
    assert d[0] == pytest.approx(100.0)      # centre 400 -> 100 from each


def test_a_contig_without_domains_gets_nan():
    """A bin on a chromosome the domains never mention has no distance, and
    NaN keeps it out of the profile instead of placing it at zero."""
    m = _mod()
    bins = pd.concat([_grid(2, "chr1", 100), _grid(2, "chr9", 100)],
                     ignore_index=True)
    dom = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [200]})
    d = m.signed_distance(bins, dom)
    assert np.isfinite(d[:2]).all()
    assert np.isnan(d[2:]).all()


# ----------------------------------------------------------- occupancy ----

def _iv(rows):
    return pd.DataFrame(rows, columns=["chrom", "start", "end"])


def test_occupancy_asks_whether_the_set_is_there_per_boundary():
    """One row per boundary edge, 1 where a bin of the set overlaps that
    offset window.  This is the denominator that makes a random set flat."""
    m = _mod()
    dom = _iv([("chr1", 1_000_000, 3_000_000)])
    # a single set bin sitting just inside the left edge
    bed = _iv([("chr1", 1_000_000, 1_250_000)])
    occ, centres = m.occupancy(dom, bed, flank=1_000_000, step=250_000)
    assert occ.shape == (2, 8)          # two edges, eight offsets
    left = occ[0]
    assert left[centres == 125_000][0]  # just inside the left edge
    assert not left[centres == -875_000][0]


def test_offsets_are_oriented_so_positive_is_inside():
    """For a right edge that means walking left; without the flip the two
    edges cancel."""
    m = _mod()
    dom = _iv([("chr1", 1_000_000, 3_000_000)])
    bed = _iv([("chr1", 2_750_000, 3_000_000)])   # just inside the RIGHT edge
    occ, centres = m.occupancy(dom, bed, flank=1_000_000, step=250_000)
    right = occ[1]
    assert right[centres == 125_000][0], "inside should be positive"
    assert not right[centres == -125_000][0]


def test_a_uniformly_placed_set_is_flat():
    """The whole point.  Histogramming a set's own bins by distance peaks at
    the boundary for ANY set, because the positions available at each
    distance are not uniform; occupancy per boundary does not."""
    m = _mod()
    step = 250_000
    dom = _iv([("chr1", i * 20_000_000, i * 20_000_000 + 4_000_000)
               for i in range(1, 9)])
    rng = np.random.default_rng(0)
    starts = np.sort(rng.choice(np.arange(0, 200_000_000, step),
                                size=400, replace=False))
    bed = _iv([("chr1", int(s), int(s) + step) for s in starts])
    occ, centres = m.occupancy(dom, bed, flank=5_000_000, step=step)
    dens, n = m.profile_from_occupancy(occ)
    assert n == 16
    # flat: the centre is no higher than the rest, within sampling noise
    near = dens[np.abs(centres) < 1_000_000].mean()
    far = dens[np.abs(centres) >= 3_000_000].mean()
    assert abs(near - far) < 0.12, f"near {near:.3f} vs far {far:.3f}"


def test_a_boundary_hugging_set_is_not_flat():
    m = _mod()
    step = 250_000
    # 12 Mb wide, so a domain's own far edge falls OUTSIDE the 5 Mb window
    # and the test measures boundary-hugging rather than domain width.
    dom = _iv([("chr1", i * 30_000_000, i * 30_000_000 + 12_000_000)
               for i in range(1, 7)])
    rows = []
    for i in range(1, 7):
        for edge in (i * 30_000_000, i * 30_000_000 + 12_000_000):
            # one bin each side of the edge, so both nearest offsets are hit
            rows.append(("chr1", edge - step, edge))
            rows.append(("chr1", edge, edge + step))
    occ, centres = m.occupancy(dom, _iv(rows), flank=5_000_000, step=step)
    dens, _ = m.profile_from_occupancy(occ)
    assert dens[np.abs(centres) < 300_000].mean() > 0.9
    assert dens[np.abs(centres) > 3_000_000].mean() < 0.1


def test_profile_from_empty_occupancy_is_nan():
    m = _mod()
    dens, n = m.profile_from_occupancy(np.zeros((0, 8), bool))
    assert n == 0 and np.isnan(dens).all()


def test_occupancy_ignores_a_chromosome_the_set_never_touches():
    m = _mod()
    dom = _iv([("chr9", 1_000_000, 3_000_000)])
    bed = _iv([("chr1", 1_000_000, 1_250_000)])
    occ, _ = m.occupancy(dom, bed, flank=1_000_000, step=250_000)
    assert occ.shape[0] == 2 and not occ.any()


# ---------------------------------------------------------------- test ----

def test_the_occupancy_test_compares_near_offsets_with_far_ones():
    m = _mod()
    centres = np.array([-750_000.0, -250_000.0, 250_000.0, 750_000.0])
    occ = np.array([[False, True, True, False],
                    [False, True, True, False],
                    [True, True, True, True]])
    st = m.occupancy_test(occ, centres, near=500_000)
    assert (st["n_occupied_near"], st["n_empty_near"]) == (6, 0)
    assert (st["n_occupied_far"], st["n_empty_far"]) == (2, 4)
    assert st["frac_near"] == pytest.approx(1.0)
    assert st["odds_ratio"] > 1


def test_a_flat_set_gives_an_odds_ratio_near_one():
    m = _mod()
    rng = np.random.default_rng(1)
    centres = np.linspace(-5e6, 5e6, 40)
    occ = rng.random((200, 40)) < 0.2
    st = m.occupancy_test(occ, centres, near=5e5)
    assert 0.7 < st["odds_ratio"] < 1.4
    assert st["pvalue"] > 0.05


def test_the_occupancy_test_on_nothing_returns_nothing():
    m = _mod()
    assert m.occupancy_test(np.zeros((0, 4), bool),
                            np.zeros(4), near=1.0) == {}


# ------------------------------------------------------ set-fraction ------

def test_set_fraction_sums_to_one_across_the_axis():
    """The old view: of the SET's bins, the fraction at each offset.  It keeps
    its range however large the set is, which is why it stays readable when
    occupancy would saturate."""
    m = _mod()
    dom = _iv([("chr1", 1_000_000, 3_000_000)])
    bed = _iv([("chr1", s, s + 250_000)
               for s in range(0, 4_000_000, 250_000)])
    freq, counts = m.set_fraction_profile(bed, dom, flank=1_000_000,
                                          step=250_000)
    assert np.nansum(freq) == pytest.approx(1.0)
    assert counts.sum() > 0


def test_set_fraction_peaks_for_a_boundary_hugging_set():
    m = _mod()
    dom = _iv([("chr1", 10_000_000, 20_000_000)])
    bed = _iv([("chr1", 9_750_000, 10_000_000),
               ("chr1", 10_000_000, 10_250_000)])
    freq, _ = m.set_fraction_profile(bed, dom, flank=2_000_000, step=250_000)
    centres = m.offset_edges(2_000_000, 250_000)[1]
    assert freq[np.abs(centres) < 300_000].sum() == pytest.approx(1.0)


def test_set_fraction_peaks_for_a_UNIFORM_set_too():
    """The cost of this view, and the reason a control is not optional in it:
    the positions available at each distance are not uniform, so a set that
    ignores the boundaries entirely still peaks at one."""
    m = _mod()
    step = 250_000
    dom = _iv([("chr1", 10_000_000, 14_000_000)])
    bed = _iv([("chr1", s, s + step)
               for s in range(0, 24_000_000, step)])
    freq, _ = m.set_fraction_profile(bed, dom, flank=5_000_000, step=step)
    centres = m.offset_edges(5_000_000, step)[1]
    near = freq[np.abs(centres) < 500_000].mean()
    far = freq[np.abs(centres) > 3_000_000].mean()
    assert near > far, "a uniform set peaks at the boundary in this view"


def test_an_empty_set_fraction_is_nan():
    m = _mod()
    dom = _iv([("chr1", 0, 100)])
    bed = _iv([("chr9", 0, 100)])
    freq, counts = m.set_fraction_profile(bed, dom, flank=100, step=50)
    assert counts.sum() == 0 and np.isnan(freq).all()


# ------------------------------------------------ class-partitioned BEDs ---

def _classed(path, rows):
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, header=False)
    return str(path)


def test_a_genome_partitioning_bed_is_refused_without_a_class(tmp_path):
    """The Zhou PMD file tiles every chromosome as PMD / HMD / NA.  Read as
    three columns and merged it becomes ONE interval per chromosome, and a
    boundary profile against that measures chromosome ends -- silently."""
    m = _mod()
    rows = [("chr1", i * 100_000, (i + 1) * 100_000, 0.5,
             "PMD" if i % 2 else "HMD", "x") for i in range(200)]
    p = _classed(tmp_path / "pmd.bed", rows)
    with pytest.raises(SystemExit, match="class column"):
        m.read_bed(p, merge=True)


def test_naming_the_class_selects_only_those_rows(tmp_path):
    m = _mod()
    rows = [("chr1", i * 100_000, (i + 1) * 100_000, 0.5,
             "PMD" if i % 2 else "HMD", "x") for i in range(200)]
    p = _classed(tmp_path / "pmd.bed", rows)
    got = m.read_bed(p, keep_class="PMD")
    assert len(got) == 100
    assert (got.start % 200_000 == 100_000).all()


def test_an_absent_class_is_an_error_not_an_empty_result(tmp_path):
    m = _mod()
    rows = [("chr1", 0, 100, 0.5, "PMD", "x"), ("chr1", 100, 200, 0.5,
                                                "HMD", "x")]
    p = _classed(tmp_path / "pmd.bed", rows)
    with pytest.raises(SystemExit, match="no row has class"):
        m.read_bed(p, keep_class="NOPE")


def test_a_plain_three_column_bed_needs_no_class(tmp_path):
    m = _mod()
    p = _bed(tmp_path / "d.bed", [("chr1", 0, 100), ("chr1", 500, 600)])
    assert len(m.read_bed(p)) == 2


def test_a_numeric_fourth_column_is_not_mistaken_for_a_class(tmp_path):
    """A BED6 with a score must still read as plain intervals."""
    m = _mod()
    rows = [("chr1", i * 100, (i + 1) * 100, i * 0.37, ".", "+")
            for i in range(50)]
    p = _classed(tmp_path / "scored.bed", rows)
    assert len(m.read_bed(p)) == 50


def test_a_chromosome_with_one_domain_has_no_neighbour_to_exclude():
    """Which is why a file merged to one block per chromosome reports every
    domain as isolated -- the tell that it was not a domain file."""
    m = _mod()
    dom = pd.DataFrame({"chrom": ["chr1", "chr2"], "start": [0, 0],
                        "end": [200_000_000, 150_000_000]})
    assert len(m.isolated_boundaries(dom, flank=5_000_000)) == 2


# --------------------------------------------------------- signal mode ----

def _matrix(groups, n_bins=12, step=250_000, seed=0):
    rng = np.random.default_rng(seed)
    cells, types = [], []
    for g, k in groups:
        cells += [f"{g}_{i}" for i in range(k)]
        types += [g] * k
    cols = [f"chr1:{i * step}" for i in range(n_bins)]
    M = pd.DataFrame(rng.normal(0.5, 0.01, (len(cells), n_bins)),
                     index=cells, columns=cols)
    return M, pd.Series(types, index=cells)


def test_the_signal_profile_separates_cell_types():
    """The reference types on the same axis ARE the control: if every type
    dips inside the domain that is the domain, and if only one does that is
    the finding."""
    m = _mod()
    M, ct = _matrix([("tum", 6), ("ref", 6)])
    dist = np.array([-1e6, -0.5e6, 0.5e6, 1e6] * 3)[:12]
    M.loc[ct.index[ct == "tum"], M.columns[dist > 0]] -= 0.30
    got = m.signal_profile(M, ct, ["tum", "ref"], dist,
                           flank=2_000_000, step=1_000_000)
    centres = m.offset_edges(2_000_000, 1_000_000)[1]
    inside = centres > 0
    assert got["tum"]["mean"][inside].mean() == pytest.approx(0.20, abs=0.02)
    assert got["ref"]["mean"][inside].mean() == pytest.approx(0.50, abs=0.02)


def test_the_band_is_across_cells_not_across_bins():
    """Bins inside a domain are heavily correlated; pooling them as
    independent observations gives a band far too tight to mean anything."""
    m = _mod()
    M, ct = _matrix([("g", 4)], n_bins=40)
    dist = np.linspace(-1e6, 1e6, 40)
    got = m.signal_profile(M, ct, ["g"], dist, flank=2_000_000,
                           step=2_000_000)
    half = got["g"]["hi"][0] - got["g"]["mean"][0]
    # four cells: the interval reflects those four, not the bins.  Offset bin
    # 0 spans [-2 Mb, 0), so it holds only the negative-distance columns.
    per_cell = M.loc[:, M.columns[dist < 0]].mean(axis=1).to_numpy()
    from scipy.stats import t as _t
    want = _t.ppf(0.975, 3) * per_cell.std(ddof=1) / np.sqrt(4)
    assert half == pytest.approx(want, rel=0.05)


def test_a_group_with_one_cell_gets_no_interval():
    m = _mod()
    M, ct = _matrix([("solo", 1), ("pair", 2)])
    dist = np.zeros(12)
    got = m.signal_profile(M, ct, ["solo", "pair"], dist, flank=1e6, step=1e6)
    assert np.isfinite(got["solo"]["mean"]).any()
    assert np.isnan(got["solo"]["lo"]).all(), "one cell cannot carry a band"
    assert np.isfinite(got["pair"]["lo"]).any()


def test_bins_outside_the_flank_are_not_counted():
    m = _mod()
    M, ct = _matrix([("g", 3)])
    dist = np.concatenate([np.full(6, 0.5e6), np.full(6, 9e6)])
    got = m.signal_profile(M, ct, ["g"], dist, flank=1_000_000, step=1e6)
    assert got["g"]["n_bins"].sum() == 6


def test_a_group_absent_from_the_matrix_is_skipped():
    m = _mod()
    M, ct = _matrix([("a", 3)])
    got = m.signal_profile(M, ct, ["a", "absent"], np.zeros(12),
                           flank=1e6, step=1e6)
    assert list(got) == ["a"]
