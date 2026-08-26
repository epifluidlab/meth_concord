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
        bi, pi, total, _dm = m.scaled_index(bins, dom, 40, 2_000_000, 20)
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
        bi, pi, _, _dm = m.scaled_index(bins, _domains([span]), 40, 2_000_000, 20)
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
    bi, pi, _, _dm = m.scaled_index(bins, dom, 40, 2_000_000, 20)
    counts = pd.Series(bi).value_counts()
    assert (counts > 1).any()


def test_no_domains_yields_an_empty_axis_not_a_crash():
    m = _mod()
    bi, pi, total, _dm = m.scaled_index(_bins(50), _domains([]), 40, 2_000_000, 20)
    assert bi.size == 0 and pi.size == 0 and total == 80


def test_positions_never_escape_the_axis():
    m = _mod()
    bins = _bins(400)
    dom = _domains([(0, 3 * BIN), (390 * BIN, 400 * BIN)])   # at both edges
    bi, pi, total, _dm = m.scaled_index(bins, dom, 40, 2_000_000, 20)
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
    bi, pi, total, _dm = m.scaled_index(bins, _domains([(100 * BIN, 180 * BIN)]),
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
    bi, pi, total, _dm = m.scaled_index(bins, _domains([(100 * BIN, 180 * BIN)]),
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
    bi, pi, total, _dm = m.scaled_index(bins, _domains([(100 * BIN, 180 * BIN)]),
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
    bi, pi, total, _dm = m.scaled_index(bins, _domains([(100 * BIN, 180 * BIN)]),
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
    bi, pi, total, _dm = m.scaled_index(bins, _domains([(100 * BIN, 180 * BIN)]),
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


# ------------------------------------------------------- flank anchoring --

def _residual_matrix(n_bins=400, body=(100, 180), seed=4,
                     arm_gap=2.0e-5, structure=2.0e-5, cell_sd=1.5e-5):
    """A residual arm difference that survives cell-median centring.

    Centring equalises each cell's GENOME median, which does not make one
    particular set of flank bins line up.  This is the shape left on the
    real figure: the arms are parallel but offset outside the domain.
    """
    rng = np.random.default_rng(seed)
    cts = (["tumor1"] * 12 + ["tumor2"] * 12
           + ["GM12878"] * 10 + ["IMR90"] * 10 + ["monocyte"] * 10)
    X = rng.normal(1.0e-4, 3.0e-5, (len(cts), n_bins))
    # Cells differ in overall level, which real cells do: it is the thing
    # anchoring exists to remove.  Without it the flank mean is pure noise
    # and subtracting it can only ADD variance.
    X += rng.normal(0.0, cell_sd, (len(cts), 1))
    tum = np.array([c.startswith("tumor") for c in cts])
    X[tum] += arm_gap                                   # parallel offset
    X[np.ix_(tum, np.arange(*body))] += structure       # and the finding
    cells = [f"cell{i}" for i in range(len(cts))]
    return (pd.DataFrame(X, index=cells,
                         columns=[f"chr1:{i * BIN}" for i in range(n_bins)]),
            pd.Series(cts, index=cells))


def _idx(n_bins=400, span=(100, 180)):
    m = _mod()
    return m.scaled_index(_bins(n_bins),
                          _domains([(span[0] * BIN, span[1] * BIN)]),
                          40, 2_000_000, 20)


def test_without_anchoring_the_flanks_stay_apart():
    """The reported problem."""
    m = _mod()
    M, ct = _residual_matrix()
    bi, pi, total, _dm = _idx()
    a = m.group_profile(M, ct, ["tumor1", "tumor2"], bi, pi, total)
    b = m.group_profile(M, ct, ["GM12878", "IMR90", "monocyte"], bi, pi, total)
    gap = np.nanmean(a["mean"][:20]) - np.nanmean(b["mean"][:20])
    assert gap > 1.0e-5


def test_anchoring_puts_both_flanks_on_zero():
    m = _mod()
    M, ct = _residual_matrix()
    bi, pi, total, _dm = _idx()
    kw = dict(baseline="flank", n_flank=20, n_body=40)
    a = m.group_profile(M, ct, ["tumor1", "tumor2"], bi, pi, total, **kw)
    b = m.group_profile(M, ct, ["GM12878", "IMR90", "monocyte"], bi, pi,
                        total, **kw)
    for r in (a, b):
        flank = np.concatenate([r["mean"][:20], r["mean"][60:]])
        assert abs(np.nanmean(flank)) < 1.0e-9


def test_anchoring_keeps_the_domain_effect_it_is_meant_to_reveal():
    """It must move the baseline, not the finding."""
    m = _mod()
    M, ct = _residual_matrix()
    bi, pi, total, _dm = _idx()
    plain = m.group_profile(M, ct, ["tumor1", "tumor2"], bi, pi, total)
    anch = m.group_profile(M, ct, ["tumor1", "tumor2"], bi, pi, total,
                           baseline="flank", n_flank=20, n_body=40)
    body = slice(20, 60)
    d_plain = np.nanmean(plain["mean"][body]) - np.nanmean(plain["mean"][:20])
    d_anch = np.nanmean(anch["mean"][body]) - np.nanmean(anch["mean"][:20])
    assert d_anch == pytest.approx(d_plain, rel=1e-6)


def test_anchoring_is_per_cell_so_it_tightens_the_interval():
    """Subtracting a GROUP mean would leave each cell's own level intact and
    the band would still be dominated by it.  Per cell, the between-cell
    level spread is removed and the band reports the change."""
    m = _mod()
    M, ct = _residual_matrix()
    bi, pi, total, _dm = _idx()
    plain = m.group_profile(M, ct, ["tumor1", "tumor2"], bi, pi, total)
    anch = m.group_profile(M, ct, ["tumor1", "tumor2"], bi, pi, total,
                           baseline="flank", n_flank=20, n_body=40)
    body = slice(20, 60)
    w_plain = np.nanmean(plain["hi"][body] - plain["lo"][body])
    w_anch = np.nanmean(anch["hi"][body] - anch["lo"][body])
    assert w_anch < w_plain


def test_anchoring_removes_a_genuine_global_difference_too():
    """The honest cost, pinned so it is not forgotten: with no domain effect
    at all, anchoring makes two arms that really do differ look identical."""
    m = _mod()
    M, ct = _residual_matrix(structure=0.0)
    bi, pi, total, _dm = _idx()
    tum, ref = ["tumor1", "tumor2"], ["GM12878", "IMR90", "monocyte"]
    kw = dict(baseline="flank", n_flank=20, n_body=40)
    before = (np.nanmean(m.group_profile(M, ct, tum, bi, pi, total)["mean"])
              - np.nanmean(m.group_profile(M, ct, ref, bi, pi,
                                           total)["mean"]))
    after = (np.nanmean(m.group_profile(M, ct, tum, bi, pi, total,
                                        **kw)["mean"])
             - np.nanmean(m.group_profile(M, ct, ref, bi, pi, total,
                                          **kw)["mean"]))
    assert before > 1.0e-5                     # the arms really do differ
    assert abs(after) < 0.2 * before           # and the figure no longer says so


def test_member_curves_are_anchored_the_same_way():
    """Faint lines on a different baseline from the line they explain would
    be worse than not drawing them."""
    m = _mod()
    M, ct = _residual_matrix()
    bi, pi, total, _dm = _idx()
    cur = m.member_curves(M, ct, ["tumor1", "tumor2"], bi, pi, total,
                          baseline="flank", n_flank=20, n_body=40)
    for y in cur.values():
        flank = np.concatenate([y[:20], y[60:]])
        assert abs(np.nanmean(flank)) < 1.0e-9


def test_the_cli_anchors_and_says_what_that_means(tmp_path):
    M, ct = _residual_matrix()
    tab, r = _profile(M, ct, tmp_path, extra=["--baseline", "flank"])
    assert abs(tab.loc["tumor", "mean_flank"]) < 1.0e-9
    assert abs(tab.loc["other", "mean_flank"]) < 1.0e-9
    assert "not as level" in r.stderr


def test_anchoring_alone_does_not_silence_the_normalisation_warning(tmp_path):
    """It is a display choice made after the fact and cannot stand in for
    matching what the detection run did to the values."""
    M, ct = _residual_matrix()
    _, r = _profile(M, ct, tmp_path, extra=["--baseline", "flank"])
    assert "no normalisation applied" in r.stderr


# --------------------- the reference correction only fits one kind of matrix --

def _marker():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "scripts" / "celltype_marker_bins.py"
    spec = importlib.util.spec_from_file_location("_mk", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_log_odds_matrix_is_refused_not_flattened():
    """direction_logor is a log odds ratio centred near zero.  logit pins
    everything outside (0, 1) to the floor, so the whole negative half --
    about half the matrix -- would collapse onto one value.  That is a
    silently destroyed signal, not a rough correction."""
    import numpy as np
    import pytest
    mk = _marker()
    rng = np.random.default_rng(0)
    L = rng.normal(0.0, 0.4, (6, 400))
    with pytest.raises(ValueError, match="logit-based"):
        mk.correct_by_reference(L, np.full(6, 0.35))
    # what it used to do instead: flatten every bin in a cell to one value
    out, info = mk.correct_by_reference(np.abs(L) * 0 + 0.3, np.full(6, 0.35))
    assert info["n_corrected"] == 6


def test_the_refusal_says_what_the_right_operation_would_be():
    import numpy as np
    import pytest
    mk = _marker()
    with pytest.raises(ValueError) as e:
        mk.correct_by_reference(np.array([[-0.5, 0.5]]), np.array([0.3]))
    msg = str(e.value)
    assert "DIVIDE by a reference C" in msg
    assert "--contrast-center" in msg          # what still applies


def test_a_signed_coordination_matrix_is_refused_too():
    """Post --dm-prior shared, C reaches below zero -- which is the point of
    the fix, and also what makes this correction inapplicable to it."""
    import numpy as np
    import pytest
    mk = _marker()
    C = np.array([[-0.0011, 0.0002, 0.0008, -0.0004]])
    with pytest.raises(ValueError, match="outside that range"):
        mk.correct_by_reference(C, np.array([0.35]))


def test_a_reliability_matrix_still_passes():
    """A Pearson r in (0, 1) is what the transform was built for, and a few
    bins dipping to the floor from sampling noise must not trip the guard."""
    import numpy as np
    mk = _marker()
    rng = np.random.default_rng(1)
    D = np.clip(rng.normal(0.35, 0.08, (8, 500)), 0.0, 0.999)
    out, info = mk.correct_by_reference(D, np.full(8, 0.35))
    assert info["n_corrected"] == 8
    assert np.isfinite(out).all()


def test_the_reference_runs_before_the_centring_that_would_break_it():
    """Centring puts the median at 0, and the reference correction takes
    logit, which needs (0, 1).  Run in the old order the two together
    flattened every bin in a cell while the run reported success -- so the
    order is part of the contract, not an implementation detail."""
    from pathlib import Path
    for script in ("celltype_marker_bins.py", "scaled_region_profile.py"):
        src = (Path(__file__).resolve().parents[1] / "scripts"
               / script).read_text()
        i_ref = src.index("correct_by_reference(")
        i_cen = src.rindex("center_cells(Xc)")
        assert i_ref < i_cen, f"{script}: centring must come after the reference"


def test_rescaling_removes_an_e_squared_attenuation():
    """C carries e^2 var_A, so dividing by a reference C leaves var_A."""
    import numpy as np
    mk = _marker()
    rng = np.random.default_rng(0)
    e = rng.uniform(0.4, 0.9, 40)
    X = rng.normal(0, 1, (40, 100)) * (e ** 2)[:, None]
    out, info = mk.scale_by_reference(X, 0.05 * e ** 2)
    assert info["n_corrected"] == 40
    spread = lambda M: float(np.std(np.std(M, axis=1)))
    assert spread(out) < 0.25 * spread(X)


def test_a_reference_centred_on_zero_is_refused_not_divided_by():
    """A chrM log odds ratio at a symmetric marginal is ~0.  Dividing by it
    does not add variance, it flips signs."""
    import numpy as np
    mk = _marker()
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (40, 100))
    _out, info = mk.scale_by_reference(X, rng.normal(-0.0003, 0.01, 40))
    assert info["n_corrected"] == 0
    assert "no definite magnitude or sign" in info["note"]


def test_a_cell_far_below_the_cohort_is_left_alone_not_amplified():
    import numpy as np
    mk = _marker()
    ref = np.full(20, 0.05)
    ref[0] = 0.001                       # a fiftieth of everyone else
    ref[1] = -0.05                       # and one of the wrong sign
    X = np.ones((20, 10))
    out, info = mk.scale_by_reference(X, ref)
    assert info["n_corrected"] == 18
    assert out[0, 0] == 1.0 and out[1, 0] == 1.0


# ------------------------- a region's statistic is not the mean of its bins --

def test_averaging_per_bin_statistics_attenuates_by_the_share_of_thin_bins():
    """C and the log odds ratio are ratios of aggregated counts, so they are
    not additive across bins, and a thin bin's value is mostly the prior.

    That makes the error a BIAS rather than a scale factor: it follows the
    share of thin bins, and a domain and its flank differ in accessibility
    and so in depth.  They are attenuated by different amounts and the
    contrast between them -- which is the whole figure -- is distorted."""
    import numpy as np
    from nome_hic_concordance.concordance import decompose_components as dc
    rng = np.random.default_rng(0)
    true_lor, nb, p_bar = 0.35, 20, 0.30
    e_o, e_c = p_bar ** 2, (1 - p_bar) ** 2
    depth = np.r_[rng.integers(1, 4, 16), rng.integers(200, 400, 4)]
    odds = (e_o / e_c) * np.exp(true_lor)
    ko = rng.binomial(depth, odds / (1 + odds)).astype(float)
    kc = (depth - ko).astype(float)
    a_o, a_c, a0 = 10.0, 49.0, 100.0
    post = lambda k_o, k_c, n: ((a_o + k_o) / (a0 + n), (a_c + k_c) / (a0 + n))
    per = dc(*post(ko, kc, depth),
             *post(e_o * depth, e_c * depth, depth))["direction_logor"]
    t = depth.sum()
    pooled = dc(*[np.array([x]) for x in post(ko.sum(), kc.sum(), t)],
                *[np.array([x]) for x in post(e_o * t, e_c * t, t)]
                )["direction_logor"][0]
    # measured: mean-of-bins is 0.307 from the truth, pooled 0.128
    assert abs(np.mean(per) - true_lor) > 2 * abs(pooled - true_lor)
    assert np.mean(per[depth < 4]) < 0.1 * np.mean(per[depth > 100])


def test_pooled_profiles_uses_counts_not_the_per_bin_values():
    import numpy as np
    m = _mod()
    n_cells, n_bins, total = 6, 8, 3
    counts = {k: np.zeros((n_cells, n_bins)) for k in m.COUNT_KEYS}
    for k in ("k_open", "k_dm_open"):
        counts[k][:] = 3.0
    for k in ("k_closed", "k_dm_closed"):
        counts[k][:] = 5.0
    counts["n_linked"][:] = counts["n_dm"][:] = 8.0
    counts["k_open"][:, :4] = 7.0            # a real excess in the first half
    bin_idx = np.arange(n_bins)
    pos_idx = np.repeat(np.arange(total), [4, 2, 2])
    P, rows, nb = m.pooled_profiles(
        counts, np.arange(n_cells), bin_idx, pos_idx, total,
        {"open": 10.0, "closed": 49.0, "mixed": 41.0}, "direction_logor")
    assert P.shape == (n_cells, total)
    assert list(nb) == [4, 2, 2]
    assert P[0, 0] > P[0, 1]                 # the excess shows where it is
    assert np.isfinite(P).all()


def test_from_counts_refuses_the_corrections_it_cannot_apply(tmp_path):
    """--contrast-center and the reference corrections act on the statistic,
    not on counts.  Silently ignoring them would make two runs look
    comparable when they are not."""
    import subprocess
    import sys as _sys
    r = subprocess.run(
        [_sys.executable, str(_SCRIPT), "--matrix", "gch=/nope.parquet",
         "--samples", "/nope.tsv", "--group", "a=A", "--region", "r=/n.bed",
         "--from-counts", "/nope", "--contrast-center", "cell-median",
         "--output-dir", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode != 0
    assert "--contrast-center" in r.stderr
    assert "--baseline flank still works" in r.stderr


def test_every_component_the_pipeline_emits_can_be_pooled():
    """They are all functions of the same four pooled rates, so none is
    privileged and the choices must not be a retyped subset that drifts
    from what decompose_components actually returns -- direction_expected
    was missing from a hand-written list while being a component people
    profile as a diagnostic."""
    import numpy as np
    from nome_hic_concordance.concordance import decompose_components as dc
    m = _mod()
    emitted = set(dc(*(np.zeros(1),) * 4))
    assert m.POOLABLE == emitted
    assert {"coordination", "direction_logor", "direction_excess",
            "direction_expected"} <= m.POOLABLE


def test_pooling_gives_coordination_as_readily_as_the_log_odds_ratio():
    import numpy as np
    m = _mod()
    n_cells, n_bins, total = 4, 6, 2
    counts = {k: np.zeros((n_cells, n_bins)) for k in m.COUNT_KEYS}
    counts["n_linked"][:] = counts["n_dm"][:] = 20.0
    counts["k_open"][:] = 6.0
    counts["k_closed"][:] = 6.0
    counts["k_dm_open"][:] = 4.0
    counts["k_dm_closed"][:] = 4.0
    counts["k_open"][:, :3] = 9.0            # excess concordant mass, half 1
    alpha = {"open": 10.0, "closed": 49.0, "mixed": 41.0}
    bin_idx, pos_idx = np.arange(n_bins), np.repeat([0, 1], 3)
    got = {}
    for stat in ("coordination", "direction_logor"):
        P, _r, _n = m.pooled_profiles(counts, np.arange(n_cells), bin_idx,
                                      pos_idx, total, alpha, stat)
        assert np.isfinite(P).all()
        got[stat] = P[0]
    # more concordant mass on the left for BOTH, since both read the same
    # pooled table
    assert got["coordination"][0] > got["coordination"][1]
    assert got["direction_logor"][0] > got["direction_logor"][1]


def test_both_profile_callers_take_the_same_keywords():
    """main builds ONE kw dict and passes it to group_profile and to
    member_curves.  Adding a keyword to one of them and not the other is
    invisible until a run that supplies it reaches the per-member curves,
    which is after the group curves have already been computed and
    printed -- so it fails late, on a real dataset, with everything
    looking fine up to that point."""
    import inspect
    m = _mod()
    a = set(inspect.signature(m.group_profile).parameters)
    b = set(inspect.signature(m.member_curves).parameters)
    shared = {"baseline", "n_flank", "n_body", "pooled"}
    assert shared <= a, sorted(shared - a)
    assert shared <= b, sorted(shared - b)


def test_member_curves_pools_when_the_group_does(tmp_path):
    """Otherwise the faint per-cell-type lines would be computed a
    different way from the bold line they are meant to explain."""
    import numpy as np
    import pandas as pd
    m = _mod()
    n_cells, n_bins, total = 6, 6, 2
    cells = [f"c{i}" for i in range(n_cells)]
    bins = [f"chr1:{i * 250000}" for i in range(n_bins)]
    M = pd.DataFrame(np.zeros((n_cells, n_bins)), index=cells, columns=bins)
    ct = pd.Series(["A"] * 3 + ["B"] * 3, index=cells)
    counts = {k: np.zeros((n_cells, n_bins)) for k in m.COUNT_KEYS}
    counts["n_linked"][:] = counts["n_dm"][:] = 20.0
    counts["k_open"][:] = 6.0; counts["k_closed"][:] = 6.0
    counts["k_dm_open"][:] = 4.0; counts["k_dm_closed"][:] = 4.0
    counts["k_open"][:, :3] = 9.0
    pooled = {"counts": counts, "stat": "coordination",
              "alpha": {"open": 10.0, "closed": 49.0, "mixed": 41.0}}
    bi, pi = np.arange(n_bins), np.repeat([0, 1], 3)
    curves = m.member_curves(M, ct, ["A", "B"], bi, pi, total, pooled=pooled)
    # M is all zeros, so a non-zero curve can only have come from the counts
    for g in ("A", "B"):
        assert np.isfinite(curves[g]).all()
        assert np.any(np.abs(curves[g]) > 1e-9)


# ------------------------------ is the difference across a region real? -----

def _profile_fixture(tmp_path, n_bins=80, n_cells=24, shift=0.6, seed=0):
    """A matrix with a planted domain effect, and the files to profile it."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    cells = [f"c{i}" for i in range(n_cells)]
    bins = [f"chr1:{i * 250_000}" for i in range(n_bins)]
    X = rng.normal(0.0, 0.1, (n_cells, n_bins))
    arm = np.arange(n_cells) < n_cells // 2
    body = slice(30, 50)
    X[np.ix_(arm, np.arange(30, 50))] += shift        # only inside the domain
    d = tmp_path / "run"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X, index=cells, columns=bins).to_parquet(
        d / "concordance_coordination.parquet")
    s = tmp_path / "samples.tsv"
    pd.DataFrame({"cell_name": cells, "bam_path": ["x"] * n_cells,
                  "cell_type": ["T" if a else "R" for a in arm],
                  "donor": [f"{'t' if a else 'r'}{i % 2}"
                            for i, a in enumerate(arm)]}
                 ).to_csv(s, sep="\t", index=False)
    bed = tmp_path / "dom.bed"
    # many domains inside the planted stretch, so the per-domain test has
    # something to count
    bed.write_text("".join(f"chr1\t{7_500_000 + i * 500_000}\t"
                           f"{7_750_000 + i * 500_000}\n"
                           for i in range(10)))
    return d, s, bed, X, cells, bins, arm


def test_group_profile_hands_back_one_value_per_cell():
    """The test has to use exactly what the panel draws, not a second
    computation of it that could differ by a baseline or a slice."""
    import numpy as np
    import pandas as pd
    m = _mod()
    n_cells, n_bins = 10, 12
    cells = [f"c{i}" for i in range(n_cells)]
    M = pd.DataFrame(np.ones((n_cells, n_bins)), index=cells,
                     columns=[f"chr1:{i * 250000}" for i in range(n_bins)])
    ct = pd.Series(["A"] * n_cells, index=cells)
    bi = np.arange(n_bins)
    pi = np.repeat(np.arange(6), 2)
    r = m.group_profile(M, ct, ["A"], bi, pi, 6, n_flank=1, n_body=4)
    assert r["body"].shape == (n_cells,)
    assert list(r["cells"]) == cells
    np.testing.assert_allclose(r["body"], 1.0)


def test_the_region_test_finds_a_planted_domain_difference(tmp_path):
    import numpy as np
    import pandas as pd
    m = _mod()
    d, s, _bed, X, cells, bins, arm = _profile_fixture(tmp_path)
    M = pd.read_parquet(d / "concordance_coordination.parquet")
    ct = pd.Series(["T" if a else "R" for a in arm], index=cells)
    bi = np.arange(len(bins))
    pi = np.clip((np.arange(len(bins)) - 20) // 10 + 1, 0, 5)
    prof = {g: m.group_profile(M, ct, [g], bi, pi, 6, n_flank=1, n_body=4)
            for g in ("T", "R")}
    tt = m.region_group_test(prof, ct)
    assert len(tt) == 1
    assert tt["p_cell"].iloc[0] < 1e-3
    assert abs(tt["diff_cell"].iloc[0]) > 0.1


def test_the_donor_test_is_reported_with_its_own_ceiling(tmp_path):
    """Two donors against three references cannot reach below p = 0.2 --
    that is arithmetic, not a null result, so the floor is reported beside
    the p."""
    import numpy as np
    import pandas as pd
    m = _mod()
    d, s, _bed, X, cells, bins, arm = _profile_fixture(tmp_path)
    M = pd.read_parquet(d / "concordance_coordination.parquet")
    ct = pd.Series(["T" if a else "R" for a in arm], index=cells)
    donor = pd.Series([f"t{i % 2}" if a else f"r{i % 3}"
                       for i, a in enumerate(arm)], index=cells)
    bi = np.arange(len(bins))
    pi = np.clip((np.arange(len(bins)) - 20) // 10 + 1, 0, 5)
    prof = {g: m.group_profile(M, ct, [g], bi, pi, 6, n_flank=1, n_body=4)
            for g in ("T", "R")}
    tt = m.region_group_test(prof, ct, donor=donor)
    r = tt.iloc[0]
    assert r["n_donor_a"] == 2 and r["n_donor_b"] == 3
    assert r["p_floor_donor"] == pytest.approx(0.2)
    assert r["p_donor"] >= r["p_floor_donor"] - 1e-9
    # the cell-level p is the one that looks impressive
    assert r["p_cell"] < 0.01


def test_the_cli_writes_the_region_test(tmp_path):
    import subprocess
    import sys as _sys
    import pandas as pd
    d, s, bed, _X, _c, _b, _a = _profile_fixture(tmp_path)
    out = tmp_path / "prof"
    r = subprocess.run(
        [_sys.executable, str(_SCRIPT), "--matrix",
         f"gch={d / 'concordance_coordination.parquet'}",
         "--samples", str(s), "--group", "tum=T", "--group", "ref=R",
         "--region", f"dom={bed}", "--cell-type-column", "cell_type",
         "--donor-column", "donor", "--baseline", "flank",
         "--body-bins", "10", "--flank", "5000000", "--flank-bins", "4",
         "--plot-format", "png", "--output-dir", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    tt = pd.read_csv(out / "region_group_test_gch.tsv", sep="\t")
    assert {"region", "p_cell", "p_donor", "p_floor_donor",
            "diff_cell"} <= set(tt.columns)
    assert tt["p_cell"].iloc[0] < 0.01
    assert "region test" in r.stderr


def test_without_a_donor_column_the_run_says_what_the_p_is_not(tmp_path):
    import subprocess
    import sys as _sys
    d, s, bed, _X, _c, _b, _a = _profile_fixture(tmp_path)
    r = subprocess.run(
        [_sys.executable, str(_SCRIPT), "--matrix",
         f"gch={d / 'concordance_coordination.parquet'}",
         "--samples", str(s), "--group", "tum=T", "--group", "ref=R",
         "--region", f"dom={bed}", "--cell-type-column", "cell_type",
         "--baseline", "flank", "--body-bins", "10", "--flank", "5000000",
         "--flank-bins", "4", "--plot-format", "png",
         "--output-dir", str(tmp_path / "p2")], capture_output=True,
        text=True)
    assert r.returncode == 0, r.stderr
    assert "NOT a replicate test" in r.stderr
    assert "--donor-column" in r.stderr


def test_the_domain_test_counts_domains_not_cells(tmp_path):
    """With thousands of PMDs the domain is the better-powered unit: the
    per-cell test has hundreds of observations and none of them biological
    replicates, while the per-domain one asks whether the difference is
    CONSISTENT across domains."""
    import numpy as np
    m = _mod()
    rng = np.random.default_rng(0)
    a = rng.normal(0.4, 0.2, 40)          # tumour, consistently higher
    b = rng.normal(0.0, 0.2, 40)
    r = m.domain_paired_test(a, b)
    assert r["n_domains"] == 40
    assert r["p_domain"] < 1e-5
    assert r["frac_positive"] > 0.8
    assert r["median_diff"] > 0.2


def test_a_difference_that_is_not_consistent_does_not_clear_it():
    """Half the domains up and half down is a real per-cell difference and
    no domain-level result, which is the distinction the unit exists for."""
    import numpy as np
    m = _mod()
    rng = np.random.default_rng(1)
    d = np.r_[rng.normal(0.5, 0.1, 20), rng.normal(-0.5, 0.1, 20)]
    r = m.domain_paired_test(d, np.zeros(40))
    assert r["p_domain"] > 0.05
    assert 0.4 < r["frac_positive"] < 0.6


def test_too_few_domains_report_nan_rather_than_a_number():
    import numpy as np
    m = _mod()
    r = m.domain_paired_test(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
    assert r["n_domains"] == 2 and not np.isfinite(r["p_domain"])


def test_scaled_index_keeps_which_domain_each_placement_came_from():
    import numpy as np
    import pandas as pd
    m = _mod()
    bins = pd.DataFrame({"chrom": ["chr1"] * 20,
                         "start": np.arange(20) * 250_000})
    bins["end"] = bins.start + 250_000
    dom = pd.DataFrame({"chrom": ["chr1", "chr1"],
                        "start": [1_000_000, 3_000_000],
                        "end": [2_000_000, 4_000_000]})
    bi, pi, total, dmi = m.scaled_index(bins, dom, 4, 500_000, 2)
    assert dmi.shape == bi.shape == pi.shape
    assert set(np.unique(dmi)) == {0, 1}


def test_the_cli_reports_the_domain_test(tmp_path):
    import subprocess
    import sys as _sys
    import pandas as pd
    d, s, bed, _X, _c, _b, _a = _profile_fixture(tmp_path)
    out = tmp_path / "prof_dom"
    r = subprocess.run(
        [_sys.executable, str(_SCRIPT), "--matrix",
         f"gch={d / 'concordance_coordination.parquet'}",
         "--samples", str(s), "--group", "tum=T", "--group", "ref=R",
         "--region", f"dom={bed}", "--cell-type-column", "cell_type",
         "--donor-column", "donor", "--baseline", "flank", "--domain-test",
         "--body-bins", "6", "--flank", "500000", "--flank-bins", "2",
         "--plot-format", "png", "--output-dir", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    tt = pd.read_csv(out / "region_group_test_gch.tsv", sep="\t")
    assert {"p_domain", "n_domains", "median_diff",
            "frac_positive"} <= set(tt.columns)
    assert tt["n_domains"].iloc[0] >= 5
    assert "p_domain" in r.stderr
