"""--replot (redraw from TSVs) and the across-cell-group overlay figure.

The parse is ~30 min on 1577 cells while every figure is a pure function of the
TSV, so re-drawing must never touch a BAM.  The one thing it must NOT do is
carry the stored Δ CI and p onto a different --metric: those were computed for
whichever metric the run used.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "site_distance_correlation.py")
_FIX = Path("/tmp/sdc")

_GROUPS = ["IT-L2/3", "IT-L4", "PC/EC"]
_COLORS = ["#E4572E", "#17BEBB", "#2E282A"]


def _mod():
    spec = importlib.util.spec_from_file_location("sdc_rp", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["sdc_rp"] = m
    spec.loader.exec_module(m)
    return m


def _samples(tmp_path):
    src = _FIX / "s.tsv"
    if not src.exists():
        pytest.skip("synthetic fixture not present")
    rows = [ln.rstrip("\n").split("\t") for ln in src.read_text().splitlines()]
    out = ["\t".join(rows[0] + ["color"])]
    for g, col in zip(_GROUPS, _COLORS):
        for rep in range(3):
            r = list(rows[1 + (rep % 2)])
            r[0] = f"{g.replace('/', '_')}_c{rep}"
            r[2] = g
            out.append("\t".join(r + [col]))
    p = tmp_path / "s.tsv"
    p.write_text("\n".join(out) + "\n")
    return p


def _full_run(tmp_path, *extra):
    pytest.importorskip("pysam")
    if not (_FIX / "GCH.bed.gz").exists():
        pytest.skip("fixture context BEDs not present")
    out_dir = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--samples", str(_samples(tmp_path)),
         "--context-bed",
         f"GCH:{_FIX / 'GCH.bed.gz'},HCG:{_FIX / 'HCG.bed.gz'}",
         "--output-dir", str(out_dir), "--backend", "htslib",
         "--n-threads", "2", "--min-pairs", "500", "--n-bootstrap", "100",
         "--group-column", "group", "--color-column", "color",
         "--plot-format", "png", *extra],
        capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    return r, out_dir


def _replot(out_dir, samples, *extra):
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--replot", "--output-dir", str(out_dir),
         "--samples", str(samples), "--group-column", "group",
         "--color-column", "color", "--plot-format", "png", *extra],
        capture_output=True, text=True)


# ------------------------------------------------------------------ replot ---

def test_replot_needs_no_bam_and_redraws_every_group(tmp_path):
    _r, out_dir = _full_run(tmp_path)
    figs = sorted(p.name for p in out_dir.glob("site_distance_corr_*.png"))
    for p in out_dir.glob("*.png"):
        p.unlink()                              # prove they are regenerated

    rp = _replot(out_dir, tmp_path / "s.tsv")
    assert rp.returncode == 0, f"stdout:\n{rp.stdout}\nstderr:\n{rp.stderr}"
    assert sorted(p.name for p in out_dir.glob("site_distance_corr_*.png")) == figs
    assert "without touching a BAM" in rp.stdout


def test_replot_can_switch_metric_and_drops_the_stale_ci(tmp_path):
    """The CI and p in the TSV belong to the metric the RUN used."""
    _r, out_dir = _full_run(tmp_path, "--metric", "phi")
    rp = _replot(out_dir, tmp_path / "s.tsv", "--metric", "yule_q")
    assert rp.returncode == 0, rp.stderr
    assert "belong to 'phi'" in rp.stdout, rp.stdout


def test_replot_keeps_the_ci_when_the_metric_is_unchanged(tmp_path):
    _r, out_dir = _full_run(tmp_path, "--metric", "phi")
    rp = _replot(out_dir, tmp_path / "s.tsv", "--metric", "phi")
    assert rp.returncode == 0, rp.stderr
    assert "belong to" not in rp.stdout


def test_replot_warns_when_min_pairs_is_loosened(tmp_path):
    """Bins masked on disk are gone; replot can only tighten."""
    _r, out_dir = _full_run(tmp_path, "--min-pairs", "5000")
    rp = _replot(out_dir, tmp_path / "s.tsv", "--min-pairs", "10")
    assert rp.returncode == 0, rp.stderr
    assert "can only tighten" in rp.stderr, rp.stderr


def test_replot_tightening_min_pairs_masks_more(tmp_path):
    m = _mod()
    _r, out_dir = _full_run(tmp_path, "--min-pairs", "100")
    tsv = out_dir / "site_distance_corr_all.tsv"
    curves, _boot, _c, _e, _env = m._read_tsv_curves(tsv, "phi")
    r, n = curves[("gg", "linked")]
    loose = np.isfinite(r).sum()
    tight = np.isfinite(np.where(n >= 1e9, r, np.nan)).sum()
    assert tight < loose


def test_replot_without_a_summary_exits_cleanly(tmp_path):
    out_dir = tmp_path / "empty"
    out_dir.mkdir()
    rp = subprocess.run(
        [sys.executable, str(_SCRIPT), "--replot", "--output-dir", str(out_dir)],
        capture_output=True, text=True)
    assert rp.returncode != 0
    assert "summary" in (rp.stdout + rp.stderr).lower()


def test_summary_records_the_metric_so_replot_can_check_it(tmp_path):
    _r, out_dir = _full_run(tmp_path, "--metric", "yule_q")
    meta = json.loads(
        (out_dir / "site_distance_corr_summary.json").read_text())
    assert meta["metric"] == "yule_q"
    assert "min_pairs" in meta


def test_tsv_round_trip_preserves_the_curves(tmp_path):
    m = _mod()
    _r, out_dir = _full_run(tmp_path)
    curves, boot, centers, edges, env = m._read_tsv_curves(
        out_dir / "site_distance_corr_all.tsv", "phi")
    assert set(curves) == {(k, c) for k, _ in m._MEASURES for c, _ in m._CONDS}
    assert boot is not None and "gg" in boot and len(boot["gg"]) == 4
    assert env is not None and "gg" in env
    assert len(edges) == len(centers) + 1
    assert np.isfinite(curves[("gg", "linked")][0]).any()


def test_read_tsv_returns_none_for_a_missing_metric(tmp_path):
    m = _mod()
    _r, out_dir = _full_run(tmp_path)
    df_path = out_dir / "site_distance_corr_all.tsv"
    import pandas as pd
    df = pd.read_csv(df_path, sep="\t")
    df[[c for c in df.columns if "yule_q" not in c]].to_csv(
        tmp_path / "trimmed.tsv", sep="\t", index=False)
    assert m._read_tsv_curves(tmp_path / "trimmed.tsv", "yule_q") is None


# ------------------------------------------------------------ group figure ---

def test_group_overlay_is_written_for_multiple_groups(tmp_path):
    _r, out_dir = _full_run(tmp_path)
    assert (out_dir / "site_distance_corr_by_group.png").exists()


def test_no_group_overlay_for_a_single_group(tmp_path):
    """'all' alone is not a comparison."""
    pytest.importorskip("pysam")
    if not (_FIX / "GCH.bed.gz").exists():
        pytest.skip("fixture not present")
    out_dir = tmp_path / "out1"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--samples", str(_samples(tmp_path)),
         "--context-bed",
         f"GCH:{_FIX / 'GCH.bed.gz'},HCG:{_FIX / 'HCG.bed.gz'}",
         "--output-dir", str(out_dir), "--backend", "htslib",
         "--n-threads", "2", "--min-pairs", "500", "--n-bootstrap", "0",
         "--plot-format", "png"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert not (out_dir / "site_distance_corr_by_group.png").exists()


def test_group_order_puts_the_requested_groups_first():
    m = _mod()
    got = m._order_groups(["Vip", "IT-L4", "all", "IT-L2/3"], "IT-L2/3,IT-L4")
    assert got == ["IT-L2/3", "IT-L4", "Vip", "all"], got


def test_group_order_defaults_to_alphabetical_with_all_last():
    m = _mod()
    assert m._order_groups(["Vip", "all", "IT-L4"], None) == \
        ["IT-L4", "Vip", "all"]


def test_colors_come_from_the_samples_sheet():
    m = _mod()
    import pandas as pd
    s = pd.DataFrame({"group": ["A", "A", "B"],
                      "color": ["#112233", "#112233", "#445566"]})
    got = m._group_colors(["A", "B"], s, "color", "group")
    assert got["A"] == "#112233" and got["B"] == "#445566"


def test_groups_without_a_colour_still_get_one():
    m = _mod()
    import pandas as pd
    s = pd.DataFrame({"group": ["A", "B"], "color": ["#112233", ""]})
    got = m._group_colors(["A", "B", "C"], s, "color", "group")
    assert got["A"] == "#112233"
    assert got["B"] is not None and got["C"] is not None
    assert len({str(got[k]) for k in ("A", "B", "C")}) == 3


def _overlay_fixture(m, gh_small=True):
    """per_group with GCH-GCH large, HCG-HCG mid, HCG-GCH near-zero + a null."""
    edges = m._build_edges(1, 1_000_000, 4)
    centers = np.sqrt(edges[:-1] * edges[1:])
    nb = len(centers)
    per_group = {}
    for i, g in enumerate(("A", "B", "all")):
        curves = {}
        for k, _ in m._MEASURES:
            base = {"gg": 0.4 + 0.1 * i, "hh": 0.3, "gh": 0.02 + 0.01 * i}[k]
            for c, _l in m._CONDS:
                lv = base if c == "linked" else base * 0.2
                curves[(k, c)] = (np.full(nb, lv), np.full(nb, 1e6))
        per_group[g] = (curves, centers, edges)
    return per_group, centers, edges


def _grab_fig(monkeypatch, plt):
    """Capture the Figure _plot_groups would close, so its axes stay inspectable."""
    cap = {}
    monkeypatch.setattr(plt, "close",
                        lambda a=None: cap.__setitem__("fig", a)
                        if hasattr(a, "get_axes") else None)
    return cap


def test_group_overlay_yscale_shared_gg_hh_but_gh_independent(tmp_path, monkeypatch):
    """GCH-GCH & HCG-HCG must share a y-scale (so their variation is
    comparable), but the near-zero HCG-GCH panel gets its OWN autoscaled axis so
    its small cell-type spread is visible rather than flattened."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = _mod()
    per_group, _c, _e = _overlay_fixture(m)
    cap = _grab_fig(monkeypatch, plt)
    out = tmp_path / "g.png"
    m._plot_groups(per_group, out, "phi", 100, 10.0,
                   {"A": "#111111", "B": "#222222"}, ["A", "B", "all"])
    assert out.exists()
    axes = cap["fig"].get_axes()          # row-major: gg, hh, gh, then delta row
    gg, hh, gh = axes[0].get_ylim(), axes[1].get_ylim(), axes[2].get_ylim()
    assert gg == hh, "GCH-GCH and HCG-HCG must share a y-scale"
    assert gh != gg, "HCG-GCH must be on its own (autoscaled) y-scale"
    # its own scale is actually tighter around zero than the shared one
    assert (gh[1] - gh[0]) < (gg[1] - gg[0])
    monkeypatch.undo()
    plt.close("all")


def test_group_overlay_draws_pooled_null_dashed_in_linked_row(tmp_path, monkeypatch):
    """The across-all-cells null is overlaid as a dashed line in the linked row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = _mod()
    per_group, _c, _e = _overlay_fixture(m)
    cap = _grab_fig(monkeypatch, plt)
    m._plot_groups(per_group, tmp_path / "g.png", "phi", 100, 10.0,
                   {"A": "#111111", "B": "#222222"}, ["A", "B", "all"])
    linked_gg = cap["fig"].get_axes()[0]
    dashed = [ln for ln in linked_gg.get_lines()
              if ln.get_label() == "null (all cells)"]
    assert len(dashed) == 1, "linked row must carry the pooled-null line"
    assert dashed[0].get_linestyle() in ("--", (0, (6.4, 1.6)))
    # no pooled 'all' present -> no null line, and still renders
    per_group.pop("all")
    cap.clear()
    m._plot_groups(per_group, tmp_path / "g2.png", "phi", 100, 10.0,
                   {"A": "#111111", "B": "#222222"}, ["A", "B"])
    assert (tmp_path / "g2.png").exists()
    assert not [ln for ln in cap["fig"].get_axes()[0].get_lines()
                if ln.get_label() == "null (all cells)"]
    monkeypatch.undo()
    plt.close("all")
