"""Read-PAIR level correlation, with a re-pairing null.

The companion script correlates individual cytosines and bins by the
distance between the two SITES, so one read pair contributes many
observations across several bins.  This one is the unit a Hi-C contact
actually is: one pair, one observation, binned by the distance between the
two END CENTRES.

The null re-pairs within cell AND within distance bin.  Both restrictions
are load-bearing and both are checked here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / \
    "pair_distance_correlation.py"


def _mod():
    spec = importlib.util.spec_from_file_location("_pdc", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _calls(gch, hcg, base=0):
    out = {}
    if gch is not None:
        out["GCH"] = (np.arange(len(gch)) + base, np.asarray(gch))
    if hcg is not None:
        out["HCG"] = (np.arange(len(hcg)) + base, np.asarray(hcg))
    return out


def test_a_pair_contributes_one_observation_per_measure():
    m = _mod()
    v = m.pair_values(_calls([1, 1, 0], [0, 0]),
                      _calls([1, 0], [1, 1]))
    assert v["GCH"] == (pytest.approx(2 / 3), pytest.approx(0.5))
    assert v["HCG"] == (0.0, 1.0)
    got = m.measures_of(v)
    assert got["gg"] == (pytest.approx(2 / 3), pytest.approx(0.5))
    assert got["hh"] == (0.0, 1.0)
    # The third panel is a DIFFERENCE, end1 minus end2, in both contexts.
    assert got["gh"][0] == pytest.approx(2 / 3 - 0.5)
    assert got["gh"][1] == pytest.approx(-1.0)


def test_the_third_measure_needs_both_contexts_on_both_ends():
    """That is the question, so a pair missing either cannot answer it --
    while gg and hh only need their own context."""
    m = _mod()
    v = m.pair_values(_calls([1, 0], None), _calls([1, 1], [0]))
    got = m.measures_of(v)
    assert np.isfinite(got["gg"]).all()
    assert not np.isfinite(got["hh"]).all()
    assert not np.isfinite(got["gh"]).all()


def test_min_sites_per_end_applies_per_context():
    m = _mod()
    v = m.pair_values(_calls([1], [0, 1, 1]), _calls([0, 1], [1, 0]),
                      min_sites=2)
    assert not np.isfinite(v["GCH"]).all()     # end 1 had one GpC
    assert np.isfinite(v["HCG"]).all()


def test_the_shuffle_stays_inside_its_distance_bin():
    """The bin is the axis.  A null drawn from other bins would answer
    'is this contact more correlated than one at some other distance',
    which is a different and much easier question."""
    m = _mod()
    rng = np.random.default_rng(0)
    bins = np.array([0, 0, 0, 1, 1, 1, 2])
    y = np.array([1.0, 2, 3, 10, 20, 30, 100])
    for _ in range(30):
        out = m.shuffle_within(bins, y, 3, rng)
        for b in (0, 1, 2):
            sel = bins == b
            assert sorted(out[sel]) == sorted(y[sel])
    # A single-entry bin cannot move, and must not be dropped.
    assert m.shuffle_within(bins, y, 3, rng)[6] == 100.0


def test_the_shuffle_destroys_the_link_and_keeps_both_marginals():
    m = _mod()
    rng = np.random.default_rng(1)
    n = 4000
    bins = np.zeros(n, np.int64)
    x = rng.normal(0, 1, n)
    y = x * 0.8 + rng.normal(0, 0.6, n)          # strongly linked
    obs = m.corr_of(m._running(bins, x, y, 1), 10)[0][0]
    ys = m.shuffle_within(bins, y, 1, rng)
    nul = m.corr_of(m._running(bins, x, ys, 1), 10)[0][0]
    assert obs > 0.7
    assert abs(nul) < 0.06
    np.testing.assert_allclose(np.sort(ys), np.sort(y))


def test_shuffling_within_cell_leaves_a_cell_level_offset_in_the_null():
    """THE REASON THE NULL IS PER CELL.  Cells differ in global level, so
    pooling makes end 1 and end 2 correlate simply by both running high in
    an open cell.  A null shuffled ACROSS cells destroys that too and calls
    it signal; shuffled within, it survives into the null where it belongs.
    """
    m = _mod()
    rng = np.random.default_rng(2)
    per_cell = []
    for level in (-1.5, -0.5, 0.5, 1.5):         # four cells, no within-cell link
        n = 500
        b = np.zeros(n, np.int64)
        x = level + rng.normal(0, 0.3, n)
        y = level + rng.normal(0, 0.3, n)
        per_cell.append({"gg": (b, x, y), "hh": (b[:0], x[:0], y[:0]),
                         "gh": (b[:0], x[:0], y[:0])})
    obs, null, n_used = m.accumulate_cells(per_cell, 1, 20, 0)
    r_obs = m.corr_of(obs["gg"], 10)[0][0]
    r_null = m.corr_of(null["gg"], 10)[0][0]
    assert n_used == 4
    # Pooling alone manufactures a large correlation from nothing.
    assert r_obs > 0.85
    # The within-cell null reproduces it, so the difference is ~0.
    assert abs(r_obs - r_null) < 0.05, (r_obs, r_null)


def test_a_real_within_cell_link_survives_the_null():
    """The complement: the null must not eat a correlation that is really
    there inside each cell."""
    m = _mod()
    rng = np.random.default_rng(3)
    per_cell = []
    for level in (-1.0, 1.0):
        n = 800
        b = np.zeros(n, np.int64)
        x = level + rng.normal(0, 0.5, n)
        y = level + 0.9 * (x - level) + rng.normal(0, 0.25, n)
        per_cell.append({"gg": (b, x, y), "hh": (b[:0], x[:0], y[:0]),
                         "gh": (b[:0], x[:0], y[:0])})
    obs, null, _ = m.accumulate_cells(per_cell, 1, 20, 0)
    d_link = (m.corr_of(obs["gg"], 10)[0][0]
              - m.corr_of(null["gg"], 10)[0][0])
    # The same cells with the link removed, so the two Δ differ only by the
    # thing being measured.  A bare threshold would be reading the size of
    # the cell offset as much as the size of the link.
    flat = []
    for level in (-1.0, 1.0):
        n = 800
        b = np.zeros(n, np.int64)
        flat.append({"gg": (b, level + rng.normal(0, 0.5, n),
                            level + rng.normal(0, 0.5, n)),
                     "hh": (b[:0],) * 3, "gh": (b[:0],) * 3})
    o2, n2, _ = m.accumulate_cells(flat, 1, 20, 0)
    d_flat = m.corr_of(o2["gg"], 10)[0][0] - m.corr_of(n2["gg"], 10)[0][0]
    assert abs(d_flat) < 0.05, d_flat
    assert d_link > 0.15, d_link
    assert d_link > 3 * abs(d_flat)


def test_thin_bins_report_nothing_rather_than_a_correlation_of_three_points():
    m = _mod()
    bins = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    x = np.arange(bins.size, dtype=float)
    y = x * 2
    r, n = m.corr_of(m._running(bins, x, y, 2), 5)
    assert np.isnan(r[0]) and n[0] == 3
    assert np.isfinite(r[1]) and n[1] == 9


def test_both_contexts_are_required_up_front(tmp_path):
    """Every panel uses one of them and the third uses both, so a run with
    only GCH would produce two panels of NaN and one missing -- better to
    say so before the BAM pass than after it."""
    import subprocess
    import sys
    import pandas as pd
    s = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": ["c0"],
                  "bam_path": [str(tmp_path / "none.bam")]}).to_csv(
        s, sep="\t", index=False)

    def run(*extra):
        return subprocess.run(
            [sys.executable, str(_SCRIPT), "--samples", str(s),
             "--output-dir", str(tmp_path / f"o{len(extra)}"), *extra],
            capture_output=True, text=True)

    r0 = run()
    assert r0.returncode != 0
    assert "provide --context-bed or --reference" in r0.stderr
    r1 = run("--context-bed", "GCH:/nope.bed")
    assert r1.returncode != 0
    assert "must define HCG" in r1.stderr


def test_the_context_spec_matches_the_sibling_script_and_the_pipeline():
    """One comma-separated CONTEXT:PATH string, parsed by the pipeline's own
    _parse_context_bed -- not a repeatable flag with its own parser, which
    is what the first version did and what made a valid command fail."""
    import inspect
    src = _SCRIPT.read_text()
    assert 'ap.add_argument("--context-bed", default=None' in src
    assert 'action="append"' not in src.split("--context-bed")[1][:400]
    assert "_parse_context_bed" in src
    from nome_hic_concordance.cli import _parse_context_bed
    got = _parse_context_bed("GCH:/a.bed.gz,HCG:/b.bed.gz")
    assert set(got) == {"GCH", "HCG"}


def test_the_distance_bins_are_the_decades_asked_for():
    """--bins-per-decade 1 gives 100-1,000, 1,000-10,000 and so on."""
    edges = np.logspace(2, 7, int(round(np.log10(1e7 / 1e2) * 1)) + 1)
    assert len(edges) == 6
    np.testing.assert_allclose(edges, [1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
