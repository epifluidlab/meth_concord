"""deepTools-style scaled-region profiles of D, tumour against reference.

The scaling is the point and the hazard.  Compressing every domain to the
same number of body bins is what lets domains spanning tens of kb to tens of
Mb be averaged onto one axis; it also means x stops being distance, so the
tests below pin WHERE signal lands on the axis rather than how wide it looks.
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
           / "scaled_region_profile.py")

BIN = 250_000


def _mod():
    spec = importlib.util.spec_from_file_location("srp", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["srp"] = m
    spec.loader.exec_module(m)
    return m


def _bins(n=400, chrom="chr1"):
    start = np.arange(n) * BIN
    return pd.DataFrame({"chrom": chrom, "start": start, "end": start + BIN})


def _domains(spans, chrom="chr1"):
    return pd.DataFrame({"chrom": chrom,
                         "start": [a for a, _ in spans],
                         "end": [b for _, b in spans]})


# ------------------------------------------------------------ the geometry --

def test_domains_of_different_width_map_onto_the_same_body_axis():
    """The whole reason to scale.  A 1 Mb and a 20 Mb domain must both fill
    the body, or averaging them would compare interior to exterior."""
    m = _mod()
    bins = _bins(400)
    narrow = _domains([(20 * BIN, 24 * BIN)])           # 1 Mb
    wide = _domains([(100 * BIN, 180 * BIN)])           # 20 Mb
    out = []
    for dom in (narrow, wide):
        bi, pi, total = m.scaled_index(bins, dom, 40, 2_000_000, 20)
        body = pi[(pi >= 20) & (pi < 60)]
        out.append((body.min(), body.max(), np.unique(body).size, total))
    assert out[0][3] == out[1][3] == 80
    for lo, hi, _, _ in out:
        assert 20 <= lo < 30 and 50 < hi < 60   # both span the body

    # But they do NOT resolve it equally.  A 1 Mb domain holds four 250 kb
    # bins, so its first midpoint sits 12.5% in -- at position 25, not 20 --
    # and it can occupy at most four of the forty body positions.  The rest
    # of its body axis is empty and gets filled by other domains, or by
    # nothing.  That is what --min-domain-width guards against: a domain far
    # narrower than the body resolution contributes a nearly flat smear
    # wherever the average is thin.
    assert out[0][2] <= 4 < out[1][2]


def test_the_flanks_stay_in_base_pairs():
    """Only the body is scaled.  A bin 1 Mb outside a domain must land at the
    same x whatever the domain's width, or the flank axis would not be a
    distance axis at all."""
    m = _mod()
    bins = _bins(400)
    pos = []
    for span in ((100 * BIN, 104 * BIN), (100 * BIN, 200 * BIN)):
        bi, pi, _ = m.scaled_index(bins, _domains([span]), 40, 2_000_000, 20)
        # the bin whose midpoint sits ~1 Mb upstream of the start
        want = int(np.where(bins.index == 96)[0][0])
        pos.append(pi[bi == want][0])
    assert pos[0] == pos[1]


def test_a_bin_can_serve_two_nearby_domains():
    """Both placements are real; dropping either would thin the axis exactly
    where domains cluster."""
    m = _mod()
    bins = _bins(400)
    dom = _domains([(40 * BIN, 50 * BIN), (56 * BIN, 66 * BIN)])
    bi, pi, _ = m.scaled_index(bins, dom, 40, 2_000_000, 20)
    counts = pd.Series(bi).value_counts()
    assert (counts > 1).any()


def test_no_domains_yields_an_empty_axis_not_a_crash():
    m = _mod()
    bi, pi, total = m.scaled_index(_bins(50), _domains([]), 40, 2_000_000, 20)
    assert bi.size == 0 and pi.size == 0 and total == 80


def test_positions_never_escape_the_axis():
    m = _mod()
    bins = _bins(400)
    dom = _domains([(0, 3 * BIN), (390 * BIN, 400 * BIN)])   # at both edges
    bi, pi, total = m.scaled_index(bins, dom, 40, 2_000_000, 20)
    assert pi.min() >= 0 and pi.max() < total


# ------------------------------------------------------------- the profile --

def _matrix(n_bins=400, body=(100, 180), effect=0.25, seed=0, bodies=None):
    """Tumour cells raised inside the domain(s); reference cells flat."""
    rng = np.random.default_rng(seed)
    cts = (["tumor1"] * 12 + ["tumor2"] * 12
           + ["GM12878"] * 10 + ["IMR90"] * 10 + ["monocyte"] * 10)
    X = rng.normal(0.40, 0.02, (len(cts), n_bins))
    tum = np.array([c.startswith("tumor") for c in cts])
    for lo, hi in (bodies or [body]):
        X[np.ix_(tum, np.arange(lo, hi))] += effect
    cells = [f"cell{i}" for i in range(len(cts))]
    cols = [f"chr1:{i * BIN}" for i in range(n_bins)]
    return (pd.DataFrame(X, index=cells, columns=cols),
            pd.Series(cts, index=cells))


def test_the_group_mean_rises_inside_the_domain_and_not_outside():
    m = _mod()
    M, ct = _matrix()
    bins = _bins(400)
    bi, pi, total = m.scaled_index(bins, _domains([(100 * BIN, 180 * BIN)]),
                                   40, 2_000_000, 20)
    tum = m.group_profile(M, ct, ["tumor1", "tumor2"], bi, pi, total)
    ref = m.group_profile(M, ct, ["GM12878", "IMR90", "monocyte"],
                          bi, pi, total)
    body = slice(20, 60)
    assert np.nanmean(tum["mean"][body]) > np.nanmean(tum["mean"][:20]) + 0.15
    # the reference is the control: it must be flat across the same axis
    assert abs(np.nanmean(ref["mean"][body])
               - np.nanmean(ref["mean"][:20])) < 0.05


def test_the_interval_is_across_cells_not_across_bins():
    """Bins within a domain are heavily correlated.  Counting them as
    independent would give a band far too tight to mean anything, so the
    width must track the CELL count."""
    m = _mod()
    M, ct = _matrix()
    bins = _bins(400)
    bi, pi, total = m.scaled_index(bins, _domains([(100 * BIN, 180 * BIN)]),
                                   40, 2_000_000, 20)
    few = m.group_profile(M, ct, ["tumor1"], bi, pi, total)
    many = m.group_profile(M, ct, ["tumor1", "tumor2"], bi, pi, total)
    w_few = np.nanmean(few["hi"] - few["lo"])
    w_many = np.nanmean(many["hi"] - many["lo"])
    assert w_many < w_few
    assert many["n_cells"] == 24 and few["n_cells"] == 12


def test_a_single_cell_gets_a_mean_but_no_interval():
    m = _mod()
    M, ct = _matrix()
    ct = ct.copy()
    ct.iloc[1:12] = "other"                      # leave one tumor1 cell
    bins = _bins(400)
    bi, pi, total = m.scaled_index(bins, _domains([(100 * BIN, 180 * BIN)]),
                                   40, 2_000_000, 20)
    r = m.group_profile(M, ct, ["tumor1"], bi, pi, total)
    assert r["n_cells"] == 1
    assert np.isfinite(r["mean"]).any()
    assert not np.isfinite(r["lo"]).any()


def test_member_curves_expose_a_group_carried_by_one_donor():
    """With two tumours, 'tumour is up' and 'tumour1 is up' produce the same
    group mean and the same band.  Only the per-member lines separate them."""
    m = _mod()
    rng = np.random.default_rng(1)
    cts = ["tumor1"] * 12 + ["tumor2"] * 12
    X = rng.normal(0.40, 0.02, (len(cts), 400))
    only1 = np.array([c == "tumor1" for c in cts])
    X[np.ix_(only1, np.arange(100, 180))] += 0.5      # ONE donor moves
    cells = [f"cell{i}" for i in range(len(cts))]
    M = pd.DataFrame(X, index=cells,
                     columns=[f"chr1:{i * BIN}" for i in range(400)])
    ct = pd.Series(cts, index=cells)
    bins = _bins(400)
    bi, pi, total = m.scaled_index(bins, _domains([(100 * BIN, 180 * BIN)]),
                                   40, 2_000_000, 20)
    curves = m.member_curves(M, ct, ["tumor1", "tumor2"], bi, pi, total)
    body = slice(20, 60)
    assert np.nanmean(curves["tumor1"][body]) > 0.6
    assert abs(np.nanmean(curves["tumor2"][body]) - 0.40) < 0.05


def test_missing_bins_do_not_drag_the_mean():
    m = _mod()
    M, ct = _matrix()
    M.iloc[:, 100:120] = np.nan
    bins = _bins(400)
    bi, pi, total = m.scaled_index(bins, _domains([(100 * BIN, 180 * BIN)]),
                                   40, 2_000_000, 20)
    r = m.group_profile(M, ct, ["tumor1", "tumor2"], bi, pi, total)
    assert np.isfinite(r["mean"][20:60]).any()
    assert np.nanmin(r["mean"][20:60]) > 0.5


# ------------------------------------------------------------- end to end --

# Each region gets its OWN span, far enough apart that no region's flank
# falls inside another region's planted effect.  Sharing one effect block
# across three overlapping domains would put the elevated signal in the
# FLANKS of the narrower ones and shrink body-minus-flank toward zero -- a
# fixture artefact that looks exactly like a real null.
_SPANS = {"PMD": (40, 80), "late_RT": (140, 180), "cLAD": (240, 280)}


def _run(tmp_path, extra=None):
    M, ct = _matrix(bodies=list(_SPANS.values()))
    mat = tmp_path / "gch.parquet"
    M.to_parquet(mat)
    pd.DataFrame({"cell_name": M.index, "bam_path": "x",
                  "cell_type": ct.values,
                  "sample": ["b1" if i % 2 else "b2"
                             for i in range(len(M))]}
                 ).to_csv(tmp_path / "s.tsv", sep="\t", index=False)
    for nm, (lo, hi) in _SPANS.items():
        _domains([(lo * BIN, hi * BIN)]).to_csv(
            tmp_path / f"{nm}.bed", sep="\t", header=False, index=False)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix", f"gch={mat}",
         "--samples", str(tmp_path / "s.tsv"),
         "--group", "tumor=tumor1,tumor2",
         "--group", "other=GM12878,IMR90,monocyte",
         "--region", f"PMD={tmp_path / 'PMD.bed'}",
         "--region", f"late_RT={tmp_path / 'late_RT.bed'}",
         "--region", f"cLAD={tmp_path / 'cLAD.bed'}",
         "--output-dir", str(out), "--plot-format", "png"] + (extra or []),
        capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    return out, r


def test_end_to_end_writes_one_panel_set_and_a_table(tmp_path):
    out, r = _run(tmp_path)
    assert (out / "scaled_profile_gch.png").exists()
    tab = pd.read_csv(out / "scaled_profile_gch.tsv", sep="\t")
    assert set(tab["region"]) == {"PMD", "late_RT", "cLAD"}
    assert set(tab["group"]) == {"tumor", "other"}


def test_end_to_end_recovers_the_planted_direction(tmp_path):
    out, _ = _run(tmp_path)
    tab = pd.read_csv(out / "scaled_profile_gch.tsv",
                      sep="\t").set_index(["region", "group"])
    for region in ("PMD", "late_RT", "cLAD"):
        assert tab.loc[(region, "tumor"), "body_minus_flank"] > 0.15
        assert abs(tab.loc[(region, "other"), "body_minus_flank"]) < 0.05


def test_it_says_the_band_is_across_cells_not_donors(tmp_path):
    """Two donors is the honest n for a claim about tumours, and the run
    should not let a cell-count band imply otherwise."""
    out, r = _run(tmp_path)
    assert "not independent replicates" in r.stderr


def test_an_unknown_cell_type_in_a_group_is_an_error(tmp_path):
    M, ct = _matrix()
    mat = tmp_path / "gch.parquet"
    M.to_parquet(mat)
    pd.DataFrame({"cell_name": M.index, "bam_path": "x",
                  "cell_type": ct.values}).to_csv(
        tmp_path / "s.tsv", sep="\t", index=False)
    _domains([(100 * BIN, 180 * BIN)]).to_csv(
        tmp_path / "d.bed", sep="\t", header=False, index=False)
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix", f"gch={mat}",
         "--samples", str(tmp_path / "s.tsv"),
         "--group", "tumor=tumor1,tumour_typo",
         "--region", f"D={tmp_path / 'd.bed'}",
         "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "tumour_typo" in r.stderr and "have:" in r.stderr


# ---------------------------------------------------------- normalisation --

def _offset_matrix(n_bins=400, body=(100, 180), seed=2,
                   offset=-0.06, structure=0.010):
    """A large genome-wide arm offset with small domain structure on top.

    This is the shape the real data has: the tumour arm sits far from the
    reference EVERYWHERE, and the domain effect is an order of magnitude
    smaller.  Plotted raw, the panel is a picture of the offset.
    """
    rng = np.random.default_rng(seed)
    cts = (["tumor1"] * 12 + ["tumor2"] * 12
           + ["GM12878"] * 10 + ["IMR90"] * 10 + ["monocyte"] * 10)
    X = rng.normal(0.40, 0.002, (len(cts), n_bins))
    tum = np.array([c.startswith("tumor") for c in cts])
    X[tum] += offset                                   # the nuisance
    X[np.ix_(tum, np.arange(*body))] += structure      # the finding
    cells = [f"cell{i}" for i in range(len(cts))]
    return (pd.DataFrame(X, index=cells,
                         columns=[f"chr1:{i * BIN}" for i in range(n_bins)]),
            pd.Series(cts, index=cells))


def _profile(M, ct, tmp_path, extra=None):
    mat = tmp_path / "gch.parquet"
    M.to_parquet(mat)
    pd.DataFrame({"cell_name": M.index, "bam_path": "x",
                  "cell_type": ct.values}).to_csv(
        tmp_path / "s.tsv", sep="\t", index=False)
    _domains([(100 * BIN, 180 * BIN)]).to_csv(
        tmp_path / "d.bed", sep="\t", header=False, index=False)
    out = tmp_path / ("out" + ("_n" if extra else ""))
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix", f"gch={mat}",
         "--samples", str(tmp_path / "s.tsv"),
         "--group", "tumor=tumor1,tumor2",
         "--group", "other=GM12878,IMR90,monocyte",
         "--region", f"D={tmp_path / 'd.bed'}",
         "--output-dir", str(out), "--plot-format", "png"] + (extra or []),
        capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    tab = pd.read_csv(out / "scaled_profile_gch.tsv",
                      sep="\t").set_index("group")
    return tab, r


def test_without_centring_the_arms_differ_by_the_offset_everywhere(tmp_path):
    """The reported problem, stated as a test: raw, the panel separates the
    arms by a constant that has nothing to do with the domains."""
    M, ct = _offset_matrix()
    tab, r = _profile(M, ct, tmp_path)
    gap_flank = (tab.loc["tumor", "mean_flank"]
                 - tab.loc["other", "mean_flank"])
    assert gap_flank < -0.05          # the whole-cell offset, not biology
    assert "no normalisation applied" in r.stderr


def test_centring_removes_the_offset_and_leaves_the_structure(tmp_path):
    M, ct = _offset_matrix()
    tab, r = _profile(M, ct, tmp_path,
                      extra=["--contrast-center", "cell-median"])
    gap_flank = (tab.loc["tumor", "mean_flank"]
                 - tab.loc["other", "mean_flank"])
    assert abs(gap_flank) < 0.005     # the arms now sit together outside
    # and the domain effect survives, which is the half that matters
    assert tab.loc["tumor", "body_minus_flank"] > 0.005
    assert abs(tab.loc["other", "body_minus_flank"]) < 0.002
    assert "centred each cell on its median" in r.stderr


def test_the_normalisation_that_ran_is_named_in_the_output(tmp_path):
    """A panel is only interpretable next to what was done to the values."""
    M, ct = _offset_matrix()
    _, r = _profile(M, ct, tmp_path,
                    extra=["--contrast-center", "cell-median"])
    assert "normalised:" in r.stderr


def test_regressing_coverage_needs_the_counts_and_says_so(tmp_path):
    M, ct = _offset_matrix()
    mat = tmp_path / "gch.parquet"
    M.to_parquet(mat)
    pd.DataFrame({"cell_name": M.index, "bam_path": "x",
                  "cell_type": ct.values}).to_csv(
        tmp_path / "s.tsv", sep="\t", index=False)
    _domains([(100 * BIN, 180 * BIN)]).to_csv(
        tmp_path / "d.bed", sep="\t", header=False, index=False)
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix", f"gch={mat}",
         "--samples", str(tmp_path / "s.tsv"),
         "--group", "tumor=tumor1,tumor2",
         "--region", f"D={tmp_path / 'd.bed'}",
         "--output-dir", str(tmp_path / "o"),
         "--contrast-regress-coverage"],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "--pair-counts" in r.stderr


def test_excluding_contigs_matches_what_the_detection_run_saw(tmp_path):
    """--contrast-center subtracts a per-cell median over the bins present,
    so profiling a different contig set silently re-centres everything."""
    M, ct = _offset_matrix()
    extra = pd.DataFrame(
        np.full((len(M), 20), 5.0), index=M.index,
        columns=[f"chrX:{i * BIN}" for i in range(20)])
    M2 = pd.concat([M, extra], axis=1)
    other_dir = tmp_path / "b"
    other_dir.mkdir()
    with_x, _ = _profile(M2, ct, tmp_path,
                         extra=["--contrast-center", "cell-median"])
    without_x, r = _profile(M2, ct, other_dir,
                            extra=["--contrast-center", "cell-median",
                                   "--exclude-chrom", "chrX"])
    assert "dropped 20 bins" in r.stderr
    assert (with_x.loc["tumor", "mean_body"]
            != without_x.loc["tumor", "mean_body"])
