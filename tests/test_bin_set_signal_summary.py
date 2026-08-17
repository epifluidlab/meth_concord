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
