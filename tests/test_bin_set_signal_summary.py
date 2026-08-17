"""Bin-set vs signal-track summary panel.

The figure exists to show a 2x2 interaction -- (bin set) x (early vs late) --
so the tests are about the interaction being computed and labelled correctly,
and about the scale-free effect size the comparison depends on.
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


def _mod():
    spec = importlib.util.spec_from_file_location("bss", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["bss"] = m
    spec.loader.exec_module(m)
    return m


_EARLY = ["Cas129_NPC_rep1_early", "Cas129_NPC_rep2_early", "F121_NPC_early"]
_LATE = ["Cas129_NPC_rep1_late", "Cas129_NPC_rep2_late", "F121_NPC_late"]


def _tsv(path, prefers, jitter=0.01, seed=0):
    """One signal_enrichment.tsv where the set prefers 'early' or 'late'."""
    rng = np.random.default_rng(seed)
    sign = 1.0 if prefers == "early" else -1.0
    rows = []
    for t in _EARLY + _LATE:
        d = sign * (0.18 if t in _EARLY else -0.16) + rng.normal(0, jitter)
        auc = 0.5 + d
        rows.append({"annotation": t, "mean_signal_fg": 1.0,
                     "mean_signal_bg": 0.9, "difference": d,
                     "sd_signal_fg": 0.3, "sd_signal_bg": 0.3,
                     "cohens_d": 3 * d, "auc": auc, "auc_lo": auc - 0.03,
                     "auc_hi": auc + 0.03, "n_fg": 500, "n_bg": 9000,
                     "pvalue": 1e-20, "fdr": 1e-19})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return str(path)


def _four(tmp_path):
    return [f"GCH:variable={_tsv(tmp_path / 'gv.tsv', 'late', seed=1)}",
            f"GCH:invariant={_tsv(tmp_path / 'gi.tsv', 'early', seed=2)}",
            f"HCG:variable={_tsv(tmp_path / 'hv.tsv', 'late', seed=3)}",
            f"HCG:invariant={_tsv(tmp_path / 'hi.tsv', 'early', seed=4)}"]


# --------------------------------------------------------------- unit tests --

def test_label_parsing():
    m = _mod()
    assert m._parse_enrichment("GCH:stable=/a/b.tsv") == ("GCH", "stable",
                                                          "/a/b.tsv")
    with pytest.raises(SystemExit, match="CONTEXT:SET"):
        m._parse_enrichment("GCH_stable=/a/b.tsv")
    with pytest.raises(SystemExit, match="CONTEXT:SET=PATH"):
        m._parse_enrichment("GCH:stable")


def test_track_groups_are_read_off_the_conventional_suffix():
    m = _mod()
    df = pd.DataFrame({"annotation": _EARLY + _LATE + ["GC_content"]})
    got = m.assign_groups(df.copy())
    assert list(got.loc[got["annotation"].isin(_EARLY), "track_group"]) == \
        ["early"] * 3
    assert list(got.loc[got["annotation"].isin(_LATE), "track_group"]) == \
        ["late"] * 3
    assert got.loc[got["annotation"] == "GC_content", "track_group"].iloc[0] \
        == "other"


def test_a_name_that_merely_contains_early_is_not_matched():
    """'earlyish' or 'clearly' must not be read as an early track."""
    m = _mod()
    df = pd.DataFrame({"annotation": ["clearly_mapped", "earlyish",
                                      "sample.early"]})
    got = m.assign_groups(df.copy())
    assert list(got["track_group"]) == ["other", "other", "early"]


def test_track_group_override():
    m = _mod()
    df = pd.DataFrame({"annotation": ["G1b", "G2", "unknown"]})
    got = m.assign_groups(df.copy(), ["early=G1b", "late=G2"])
    assert list(got["track_group"]) == ["early", "late", "other"]


def test_interaction_signs_the_preference_per_set():
    m = _mod()
    rows = []
    for bset, prefers in (("invariant", "early"), ("variable", "late")):
        for t in _EARLY + _LATE:
            hi = (t in _EARLY) == (prefers == "early")
            rows.append({"context": "GCH", "bin_set": bset, "annotation": t,
                         "auc": 0.7 if hi else 0.3})
    inter = m.interaction(m.assign_groups(pd.DataFrame(rows)))
    by = inter.set_index("bin_set")["contrast"]
    assert by["invariant"] > 0.3, "prefers early -> positive"
    assert by["variable"] < -0.3, "prefers late -> negative"
    # Opposite signs IS the reversal the figure is about.
    assert np.sign(by["invariant"]) != np.sign(by["variable"])


def test_interaction_skips_a_set_missing_one_group():
    m = _mod()
    rows = [{"context": "GCH", "bin_set": "x", "annotation": t, "auc": 0.6}
            for t in _EARLY]                        # early only, no late
    assert m.interaction(m.assign_groups(pd.DataFrame(rows))).empty


def test_interaction_reports_the_across_track_range_not_a_ci():
    m = _mod()
    rows = []
    for t, auc in zip(_EARLY, (0.60, 0.70, 0.80)):
        rows.append({"context": "GCH", "bin_set": "x", "annotation": t,
                     "auc": auc})
    for t in _LATE:
        rows.append({"context": "GCH", "bin_set": "x", "annotation": t,
                     "auc": 0.40})
    r = m.interaction(m.assign_groups(pd.DataFrame(rows))).iloc[0]
    assert r["min_auc_early"] == 0.60 and r["max_auc_early"] == 0.80
    assert r["contrast"] == pytest.approx(0.70 - 0.40)


# --------------------------------------------------------------- end to end --

def test_end_to_end_writes_the_panel_and_its_numbers(tmp_path):
    pytest.importorskip("matplotlib")
    m = _mod()
    out = tmp_path / "o"
    assert m.main([a for s in _four(tmp_path) for a in ("--enrichment", s)]
                  + ["--plot-format", "png", "--output-dir", str(out)]) == 0
    assert (out / "bin_set_signal_summary.png").exists()
    long = pd.read_csv(out / "bin_set_signal_long.tsv", sep="\t")
    assert len(long) == 24, "4 set/context combinations x 6 tracks"
    assert set(long["track_group"]) == {"early", "late"}
    inter = pd.read_csv(out / "bin_set_signal_interaction.tsv", sep="\t")
    assert len(inter) == 4
    # The claim, in four numbers: invariant positive, variable negative, in
    # both contexts.
    for _, r in inter.iterrows():
        assert (r["contrast"] > 0) == (r["bin_set"] == "invariant"), r.to_dict()


def test_an_old_tsv_without_the_auc_says_what_to_re_run(tmp_path):
    m = _mod()
    old = tmp_path / "old.tsv"
    pd.DataFrame([{"annotation": "F121_NPC_early", "mean_signal_fg": 1.0,
                   "mean_signal_bg": 0.9, "difference": 0.1,
                   "pvalue": 1e-9, "fdr": 1e-8}]).to_csv(
        old, sep="\t", index=False)
    with pytest.raises(SystemExit, match="not comparable between tracks"):
        m.main(["--enrichment", f"GCH:stable={old}",
                "--output-dir", str(tmp_path / "o")])


def test_tracks_in_neither_group_are_kept_but_warned_about(tmp_path):
    pytest.importorskip("matplotlib")
    m = _mod()
    df = pd.read_csv(_tsv(tmp_path / "a.tsv", "early"), sep="\t")
    df.loc[len(df)] = {**df.iloc[0].to_dict(), "annotation": "GC_content"}
    df.to_csv(tmp_path / "a.tsv", sep="\t", index=False)
    out = tmp_path / "o"
    assert m.main(["--enrichment", f"GCH:stable={tmp_path / 'a.tsv'}",
                   "--plot-format", "png", "--output-dir", str(out)]) == 0
    long = pd.read_csv(out / "bin_set_signal_long.tsv", sep="\t")
    assert (long["track_group"] == "other").sum() == 1
    inter = pd.read_csv(out / "bin_set_signal_interaction.tsv", sep="\t")
    assert inter.iloc[0]["n_early"] == 3, "the stray track is not counted in"


def test_runs_as_a_subprocess(tmp_path):
    pytest.importorskip("matplotlib")
    args = [a for s in _four(tmp_path) for a in ("--enrichment", s)]
    r = subprocess.run([sys.executable, str(_SCRIPT)] + args
                       + ["--plot-format", "png",
                          "--output-dir", str(tmp_path / "o")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "contrast" in r.stderr
