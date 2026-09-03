"""Bin-set vs signal-track summary figure.

The comparison is background-vs-set WITHIN a track, so the tests are about the
sets and the background being the same bins measured the same way, and about
tracks on different scales never being forced onto one axis.
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
           / "bin_set_signal_summary.py")
_BS = 250_000
_TRACKS = ("Cas129_NPC_rep1_early", "Cas129_NPC_rep2_early", "F121_NPC_early",
           "Cas129_NPC_rep1_late", "Cas129_NPC_rep2_late", "F121_NPC_late")


def _mod():
    spec = importlib.util.spec_from_file_location("bss", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["bss"] = m
    spec.loader.exec_module(m)
    return m


def _fixture(d: Path, nb=400, seed=0):
    """A domain-structured timing landscape, with the invariant set in early
    sequence and the variable set in late.  Late tracks are on a ~40x larger
    scale, so a figure that shared one y-axis would be unreadable."""
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    bins = pd.DataFrame({"chrom": "chr1", "start": np.arange(nb) * _BS,
                         "end": (np.arange(nb) + 1) * _BS})
    rt = np.repeat(rng.normal(0, 1, nb // 20), 20) + rng.normal(0, 0.25, nb)
    order = np.argsort(rt)[::-1]
    inv = np.zeros(nb, bool)
    inv[order[:80]] = True
    var = np.zeros(nb, bool)
    var[order[-80:]] = True
    b = bins.copy()
    b["selected"] = var
    b["stable"] = inv
    b.to_csv(d / "all.bed", sep="\t", index=False)
    bins[var].to_csv(d / "variable.bed", sep="\t", index=False)
    bins[inv].to_csv(d / "invariant.bed", sep="\t", index=False)
    for name in _TRACKS:
        sign = 1 if name.endswith("early") else -1
        scale = 1.0 if sign > 0 else 40.0
        v = scale * (5 + sign * rt + rng.normal(0, 0.2, nb))
        with open(d / f"{name}.bedGraph", "w") as fh:
            for i in range(nb):
                fh.write(f"chr1\t{i * _BS}\t{(i + 1) * _BS}\t{v[i]:.4f}\n")
    return d


def _argv(d, out, contexts=("GCH", "HCG"), extra=None):
    args = []
    for ctx in contexts:
        args += ["--background", f"{ctx}={d / 'all.bed'}",
                 "--bin-set", f"{ctx}:variable={d / 'variable.bed'}",
                 "--bin-set", f"{ctx}:invariant={d / 'invariant.bed'}"]
    for name in _TRACKS:
        args += ["--signal-track", f"{name}={d / (name + '.bedGraph')}"]
    return args + ["--plot-format", "png", "--output-dir", str(out)] + (extra
                                                                       or [])


# --------------------------------------------------------------- unit tests --

def test_track_group_reads_the_conventional_suffix():
    m = _mod()
    assert m.track_group("Cas129_NPC_rep1_early") == "early"
    assert m.track_group("F121_NPC_late") == "late"
    assert m.track_group("GC_content") == "signal"
    # A name that merely CONTAINS the word is not a timing track.
    assert m.track_group("clearly_mapped") == "signal"
    assert m.track_group("earlyish") == "signal"


def test_track_group_override_wins():
    m = _mod()
    assert m.track_group("G1b", {"G1b": "early"}) == "early"


def test_label_parsing_is_explicit_about_its_format():
    m = _mod()
    assert m._parse_labelled("GCH:stable=/a.bed", "--bin-set") == (
        "GCH", "stable", "/a.bed")
    with pytest.raises(SystemExit, match="CONTEXT:LABEL"):
        m._parse_labelled("GCH_stable=/a.bed", "--bin-set")


# --------------------------------------------------------------- end to end --

def test_the_reversal_is_recovered_in_the_table(tmp_path):
    """The result the figure exists for: the invariant set above background in
    early and below in late, the variable set the mirror image."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("scipy")
    m = _mod()
    d = _fixture(tmp_path / "in")
    out = tmp_path / "o"
    assert m.main(_argv(d, out)) == 0
    st = pd.read_csv(out / "bin_set_signal_stats.tsv", sep="\t")
    assert len(st) == 24, "2 contexts x 6 tracks x 2 sets"
    for (grp, bset), sub in st.groupby(["track_group", "bin_set"]):
        d_mean = sub["difference_of_means"]
        expect_up = (grp == "early") == (bset == "invariant")
        assert (d_mean > 0).all() == expect_up, (grp, bset, d_mean.tolist())


def test_the_figure_is_one_panel_per_group_and_context(tmp_path):
    pytest.importorskip("matplotlib")
    m = _mod()
    d = _fixture(tmp_path / "in")
    out = tmp_path / "o"
    assert m.main(_argv(d, out)) == 0
    assert (out / "bin_set_signal_summary.png").exists()


def test_tracks_keep_their_own_units(tmp_path):
    """No z-scoring, no common axis: the late tracks are ~40x the early ones
    and the table must still report them in their own scale."""
    pytest.importorskip("matplotlib")
    m = _mod()
    d = _fixture(tmp_path / "in")
    out = tmp_path / "o"
    assert m.main(_argv(d, out)) == 0
    st = pd.read_csv(out / "bin_set_signal_stats.tsv", sep="\t")
    early = st[st["track_group"] == "early"]["mean_background"].mean()
    late = st[st["track_group"] == "late"]["mean_background"].mean()
    assert late > 20 * early


def test_background_excludes_the_set_being_tested(tmp_path):
    """Otherwise the set sits on both sides and the difference shrinks."""
    pytest.importorskip("matplotlib")
    m = _mod()
    d = _fixture(tmp_path / "in")
    out = tmp_path / "o"
    assert m.main(_argv(d, out, contexts=("GCH",))) == 0
    st = pd.read_csv(out / "bin_set_signal_stats.tsv", sep="\t")
    allb = pd.read_csv(d / "all.bed", sep="\t")
    for _, r in st.iterrows():
        n_set = int(allb["stable" if r["bin_set"] == "invariant"
                         else "selected"].sum())
        assert r["n_set"] == n_set
        assert r["n_background"] == len(allb) - n_set


def test_a_set_off_the_background_grid_is_reported(tmp_path, caplog):
    import logging
    pytest.importorskip("matplotlib")
    m = _mod()
    d = _fixture(tmp_path / "in")
    off = pd.read_csv(d / "invariant.bed", sep="\t")
    off["start"] += 1                       # same region, wrong grid
    off["end"] += 1
    off.to_csv(d / "off.bed", sep="\t", index=False)
    out = tmp_path / "o"
    with caplog.at_level(logging.WARNING):
        with pytest.raises(SystemExit, match="no bin overlaps"):
            m.main(["--background", f"GCH={d / 'all.bed'}",
                    "--bin-set", f"GCH:x={d / 'off.bed'}",
                    "--signal-track", f"e={d / (_TRACKS[0] + '.bedGraph')}",
                    "--plot-format", "png", "--output-dir", str(out)])
    assert "not in the GCH background" in caplog.text


def test_a_bin_set_needs_a_background_for_its_context(tmp_path):
    m = _mod()
    d = _fixture(tmp_path / "in")
    with pytest.raises(SystemExit, match="no --background HCG"):
        m.main(["--background", f"GCH={d / 'all.bed'}",
                "--bin-set", f"HCG:x={d / 'invariant.bed'}",
                "--signal-track", f"e={d / (_TRACKS[0] + '.bedGraph')}",
                "--output-dir", str(tmp_path / "o")])


def test_ungrouped_tracks_still_plot(tmp_path):
    """A track with no early/late token is its own row, not an error."""
    pytest.importorskip("matplotlib")
    m = _mod()
    d = _fixture(tmp_path / "in")
    out = tmp_path / "o"
    assert m.main(["--background", f"GCH={d / 'all.bed'}",
                   "--bin-set", f"GCH:invariant={d / 'invariant.bed'}",
                   "--signal-track", f"GC_content={d / (_TRACKS[0] + '.bedGraph')}",
                   "--plot-format", "png", "--output-dir", str(out)]) == 0
    st = pd.read_csv(out / "bin_set_signal_stats.tsv", sep="\t")
    assert set(st["track_group"]) == {"signal"}
    assert (out / "bin_set_signal_summary.png").exists()


def test_runs_as_a_subprocess(tmp_path):
    pytest.importorskip("matplotlib")
    d = _fixture(tmp_path / "in")
    r = subprocess.run([sys.executable, str(_SCRIPT)]
                       + _argv(d, tmp_path / "o"),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "median" in r.stderr


# ----------------------------------------------- single panel + normalisation --

def _single_argv(d, out, tracks=("F121_NPC_early", "F121_NPC_late"),
                 contexts=("HCG", "GCH"), extra=None):
    args = []
    for ctx in contexts:
        args += ["--background", f"{ctx}={d / 'all.bed'}",
                 "--bin-set", f"{ctx}:variable={d / 'variable.bed'}",
                 "--bin-set", f"{ctx}:invariant={d / 'invariant.bed'}"]
    for n in tracks:
        args += ["--signal-track", f"{n}={d / (n + '.bedGraph')}"]
    return args + ["--layout", "single", "--plot-format", "png",
                   "--output-dir", str(out)] + (extra or [])


def test_single_panel_layout_writes_the_figure(tmp_path):
    pytest.importorskip("matplotlib")
    m = _mod()
    d = _fixture(tmp_path / "in")
    out = tmp_path / "o"
    assert m.main(_single_argv(d, out, extra=["--normalize",
                                              "log2-median"])) == 0
    assert (out / "bin_set_signal_summary.png").exists()
    st = pd.read_csv(out / "bin_set_signal_stats.tsv", sep="\t")
    assert len(st) == 8, "2 contexts x 2 tracks x 2 sets"
    assert set(st["track"]) == {"F121_NPC_early", "F121_NPC_late"}


def test_log2_median_puts_the_background_at_zero(tmp_path):
    """The point of the conventional enrichment axis: the background is the
    reference, so a set's box reads as fold-enrichment over the genome."""
    m = _mod()
    rng = np.random.default_rng(0)
    val = np.exp(rng.normal(0, 0.5, 501))          # strictly positive
    bg = np.ones(501, bool)
    bg[:100] = False                                # a set, excluded from ref
    # An ODD background count so the median is a real element: with an even one
    # the median of the log ratios lands on the GEOMETRIC midpoint of the two
    # middle values while the reference is their arithmetic one, and the two
    # differ in the sixth decimal.
    assert int(bg.sum()) % 2 == 1
    out, ylab = m.normalise(val, "log2-median", bg)
    assert np.median(out[bg]) == pytest.approx(0.0, abs=1e-12)
    assert "log" in ylab


def test_log2_median_refuses_a_signal_that_is_already_a_log_ratio(tmp_path):
    m = _mod()
    val = np.array([-1.0, 0.5, 2.0])
    with pytest.raises(SystemExit, match="already a log ratio"):
        m.normalise(val, "log2-median", np.ones(3, bool))


def test_zscore_handles_a_signed_signal(tmp_path):
    m = _mod()
    rng = np.random.default_rng(1)
    val = rng.normal(-3, 2, 400)                    # negative throughout
    out, ylab = m.normalise(val, "zscore", np.ones(400, bool))
    assert abs(float(np.mean(out))) < 1e-9
    assert abs(float(np.std(out)) - 1) < 1e-9
    assert "z-score" in ylab


def test_the_table_keeps_raw_units_whatever_the_figure_is_drawn_in(tmp_path):
    pytest.importorskip("matplotlib")
    m = _mod()
    d = _fixture(tmp_path / "in")
    raw = m.main(_single_argv(d, tmp_path / "raw",
                              extra=["--normalize", "none"]))
    norm = m.main(_single_argv(d, tmp_path / "norm",
                               extra=["--normalize", "log2-median"]))
    assert raw == norm == 0
    a = pd.read_csv(tmp_path / "raw" / "bin_set_signal_stats.tsv", sep="\t")
    b = pd.read_csv(tmp_path / "norm" / "bin_set_signal_stats.tsv", sep="\t")
    cols = ["median_set", "median_background", "difference_of_means"]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    assert a["median_background"].max() > 5, "raw units, not a log ratio"


def test_single_panel_without_a_normalisation_warns(tmp_path, caplog):
    """Two tracks on one raw axis would show the assay difference, not the
    bin-set difference."""
    import logging
    pytest.importorskip("matplotlib")
    m = _mod()
    d = _fixture(tmp_path / "in")
    with caplog.at_level(logging.WARNING):
        assert m.main(_single_argv(d, tmp_path / "o",
                                   extra=["--normalize", "none"])) == 0
    assert "only honest if the tracks share a scale" in caplog.text


def test_grid_layout_needs_no_normalisation_and_says_nothing(tmp_path, caplog):
    import logging
    pytest.importorskip("matplotlib")
    m = _mod()
    d = _fixture(tmp_path / "in")
    args = [a for a in _single_argv(d, tmp_path / "o") if a != "single"]
    args[args.index("--layout")] = "--layout"
    args.insert(args.index("--layout") + 1, "grid")
    with caplog.at_level(logging.WARNING):
        assert m.main(args) == 0
    assert "share a scale" not in caplog.text


# ------------------------------------------------------- --signal-ratio --

def _rt_fixture(tmp_path, n=120, span=250_000):
    """Early high in the first half, late high in the second, so log2(late/
    early) is a known -2 / +2 split."""
    pytest.importorskip("pyBigWig")
    import pyBigWig
    bins = [("chr1", i * span, (i + 1) * span) for i in range(n)]

    def w(name, rows):
        pd.DataFrame(rows).to_csv(tmp_path / name, sep="\t", header=False,
                                  index=False)
        return str(tmp_path / name)

    w("all.bed", bins)
    w("sel.bed", bins[:30])              # early half  -> log2 = -2
    w("stab.bed", bins[n // 2:n // 2 + 30])   # late half -> log2 = +2
    half = n // 2
    for nm, vals in (("early", np.r_[np.full(half, 1.6), np.full(half, 0.4)]),
                     ("late", np.r_[np.full(half, 0.4), np.full(half, 1.6)])):
        bw = pyBigWig.open(str(tmp_path / f"{nm}.bw"), "w")
        bw.addHeader([("chr1", n * span)])
        st = list(range(0, n * span, span))
        bw.addEntries(["chr1"] * n, st, ends=[x + span for x in st],
                      values=[float(v) for v in vals])
        bw.close()
    return tmp_path


def _run_ratio(d, extra, out):
    return subprocess.run(
        [sys.executable, str(_SCRIPT),
         "--background", f"GCH={d / 'all.bed'}",
         "--bin-set", f"GCH:variable={d / 'sel.bed'}",
         "--bin-set", f"GCH:invariant={d / 'stab.bed'}",
         "--signal-track", f"early={d / 'early.bw'}",
         "--signal-track", f"late={d / 'late.bw'}",
         "--plot-format", "png", "--output-dir", str(out)] + extra,
        capture_output=True, text=True)


def test_the_ratio_replaces_its_two_source_tracks(tmp_path):
    """Two tracks answering one question read better as one axis; leaving the
    sources in would keep asking the reader to divide by eye."""
    d = _rt_fixture(tmp_path)
    out = tmp_path / "o"
    r = _run_ratio(d, ["--signal-ratio", "RT=late/early"], out)
    assert r.returncode == 0, r.stderr[-2000:]
    t = pd.read_csv(out / "bin_set_signal_stats.tsv", sep="\t")
    assert set(t["track"]) == {"RT"}


def test_the_ratio_values_are_log2_of_the_two_tracks(tmp_path):
    d = _rt_fixture(tmp_path)
    out = tmp_path / "o"
    _run_ratio(d, ["--signal-ratio", "RT=late/early"], out)
    t = pd.read_csv(out / "bin_set_signal_stats.tsv", sep="\t").set_index("bin_set")
    # variable sits in the early half: log2(0.4/1.6) = -2
    assert t.loc["variable", "median_set"] == pytest.approx(-2.0, abs=1e-6)
    # invariant in the late half: log2(1.6/0.4) = +2
    assert t.loc["invariant", "median_set"] == pytest.approx(2.0, abs=1e-6)


def test_linear_scale_gives_the_plain_ratio(tmp_path):
    d = _rt_fixture(tmp_path)
    out = tmp_path / "o"
    _run_ratio(d, ["--signal-ratio", "RT=late/early",
                   "--ratio-scale", "linear"], out)
    t = pd.read_csv(out / "bin_set_signal_stats.tsv", sep="\t").set_index("bin_set")
    assert t.loc["variable", "median_set"] == pytest.approx(0.25, abs=1e-6)
    assert t.loc["invariant", "median_set"] == pytest.approx(4.0, abs=1e-6)


def test_keep_source_tracks_keeps_all_three(tmp_path):
    d = _rt_fixture(tmp_path)
    out = tmp_path / "o"
    _run_ratio(d, ["--signal-ratio", "RT=late/early",
                   "--keep-source-tracks"], out)
    t = pd.read_csv(out / "bin_set_signal_stats.tsv", sep="\t")
    assert set(t["track"]) == {"RT", "early", "late"}


def test_a_non_positive_bin_is_dropped_not_clipped(tmp_path):
    """log2 of a zero-signal bin is undefined.  Inventing a value would put a
    made-up number into the distribution AND the test."""
    m = _mod()
    raw = {"late": np.array([1.0, 2.0, 0.0, 4.0]),
           "early": np.array([1.0, 0.0, 2.0, 2.0])}
    got = m.apply_ratios(raw, [("RT", "late", "early")], "log2")
    assert np.isfinite(got["RT"]).tolist() == [True, False, False, True]
    assert got["RT"][0] == pytest.approx(0.0)
    assert got["RT"][3] == pytest.approx(1.0)


def test_an_unknown_track_name_lists_what_is_available(tmp_path):
    d = _rt_fixture(tmp_path)
    r = _run_ratio(d, ["--signal-ratio", "RT=late/typo"], tmp_path / "o")
    assert r.returncode != 0
    assert "typo" in r.stderr and "early" in r.stderr


def test_a_malformed_spec_shows_the_expected_form(tmp_path):
    d = _rt_fixture(tmp_path)
    r = _run_ratio(d, ["--signal-ratio", "late_over_early"], tmp_path / "o")
    assert r.returncode != 0
    assert "LABEL=NUMERATOR/DENOMINATOR" in r.stderr


def test_the_y_label_says_it_is_a_ratio(tmp_path):
    """The axis is no longer a signal level; labelling it 'per-bin mean
    signal' would invite reading a log ratio as an intensity."""
    m = _mod()
    d = _rt_fixture(tmp_path)
    mbe = m._enrich_mod()
    got = m.gather({"GCH": str(d / "all.bed")},
                   {("GCH", "variable"): str(d / "sel.bed")},
                   {"early": str(d / "early.bw"), "late": str(d / "late.bw")},
                   mbe, "none", ratios=[("RT", "late", "early")])
    assert got["GCH"]["ylabel"] == "log2(late / early)"
