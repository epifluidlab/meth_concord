"""Cell-type marker-bin selection + the out-of-selection control.

The statistical trap this figure must avoid is circular analysis: pick the bins
most variable across cell types, then 'show' they separate cell types.  That is
guaranteed on noise.  The defences pinned here are (1) selection filters AGAINST
batch, so sample-driven bins do not enter, and (2) the claim is validated on the
bins that were NOT selected -- if those separate cell types too, it is real.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "celltype_marker_bins.py")


def _mod():
    spec = importlib.util.spec_from_file_location("mkbins", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["mkbins"] = m
    spec.loader.exec_module(m)
    return m


def _icc():
    p = _SCRIPT.parent / "celltype_variance_vs_distance.py"
    spec = importlib.util.spec_from_file_location("_ctv_t", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_ctv_t"] = m
    spec.loader.exec_module(m)
    return m.icc_per_region


class _Args:
    min_cells_per_group = 3
    min_groups = 3
    n_bins = 50
    max_batch_icc = 0.1


def _data(seed=0, n_per=40, n_ct=4, n_marker=30, n_pervasive=40, n_batchy=30):
    """cells x bins with three bin classes: cell-type markers, weaker pervasive
    cell-type bins, and pure batch bins."""
    rng = np.random.default_rng(seed)
    ct = np.repeat(np.arange(n_ct), n_per)
    batch = np.tile([0, 1], n_ct * n_per // 2)
    n = len(ct)
    mk = rng.normal(0, 2, (n_ct, n_marker))[ct] + rng.normal(0, .4, (n, n_marker))
    perv = rng.normal(0, .8, (n_ct, n_pervasive))[ct] + rng.normal(0, .5, (n, n_pervasive))
    batchy = np.where(batch[:, None] == 0, 1.5, -1.5) + rng.normal(0, .3, (n, n_batchy))
    X = np.hstack([mk, perv, batchy])
    classes = (["marker"] * n_marker + ["pervasive"] * n_pervasive
               + ["batchy"] * n_batchy)
    return X, ct, batch, np.array(classes)


# ---------------------------------------------------------------- selection --

def test_selection_prefers_markers_and_excludes_batch_bins():
    m = _mod()
    X, ct, batch, classes = _data()
    sel, ct_icc, b_icc, _nclean = m.select_marker_bins(
        X, ct, 4, [(batch, 2)], _icc(), _Args())
    picked = classes[sel]
    # markers have the highest cell-type ICC, so the TOP of the ranking is them
    # (the weaker 'pervasive' bins legitimately fill the rest -- they are also
    # cell-type-variable, just less so).
    assert (classes[sel[:30]] == "marker").mean() > 0.8, picked[:10]
    # the safety property: sample-driven bins never enter, at any depth.
    assert "batchy" not in picked, "batch-driven bins must be filtered out"


def test_batch_icc_flags_batch_bins_within_cell_type():
    m = _mod()
    X, ct, batch, classes = _data()
    b_icc = m._batch_icc(X, ct, batch, 2, _icc(), 3, 2)  # batch: 2 groups
    assert np.nanmean(b_icc[classes == "batchy"]) > 0.5, "batch bins -> high"
    assert np.nanmean(b_icc[classes == "marker"]) < 0.15, "marker bins -> low"


def test_selection_respects_the_batch_cut():
    """Loosening --max-batch-icc lets batch bins back in — proving the cut acts."""
    m = _mod()
    X, ct, batch, classes = _data()

    class Loose(_Args):
        max_batch_icc = 1.01                       # no batch filtering
        n_bins = 200
    sel, *_rest = m.select_marker_bins(X, ct, 4, [(batch, 2)], _icc(), Loose())
    assert "batchy" in classes[sel], "with no cut, batch bins can be selected"


def test_selection_ranks_by_celltype_icc():
    m = _mod()
    X, ct, batch, classes = _data()
    sel, ct_icc, _, _ = m.select_marker_bins(X, ct, 4, [(batch, 2)], _icc(), _Args())
    assert np.all(np.diff(ct_icc[sel]) <= 1e-9), "selected bins in ICC order"
    # everything selected must beat the median non-selected cell-type ICC
    rest = np.setdiff1d(np.arange(X.shape[1]), sel)
    assert np.nanmin(ct_icc[sel]) >= np.nanmedian(ct_icc[rest])


# --------------------------------------------------------------------- split --

# ----------------------------------------------------------------- r2 metric --

def test_nonselected_separation_fires_on_structure_not_noise():
    """The out-of-selection control: real cell-type structure -> low perm p."""
    m = _mod()
    rng = np.random.default_rng(0)
    n_ct, n_per, nb = 4, 40, 60
    ct = np.repeat(np.arange(n_ct), n_per)
    # bins carry a real (weaker) cell-type signal
    X = rng.normal(0, 1, (n_ct, nb))[ct] + rng.normal(0, 0.8, (len(ct), nb))
    M = pd.DataFrame(X, columns=[f"chr1:{i}" for i in range(nb)])
    sil, p = m._nonselected_separation(M, np.array([f"c{c}" for c in ct]),
                                       10, 1, 0, 100)
    assert p < 0.05 and sil > 0, f"structured bins should separate: sil={sil}, p={p}"
    # pure noise: no separation, p not significant
    Mn = pd.DataFrame(rng.normal(0, 1, (len(ct), nb)),
                      columns=M.columns)
    _siln, pn = m._nonselected_separation(Mn, np.array([f"c{c}" for c in ct]),
                                          10, 1, 0, 100)
    assert pn > 0.05, f"noise must not separate, got p={pn}"


def test_nanmedian_is_warning_free_on_all_nan():
    m = _mod()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(m._nanmedian(np.array([np.nan, np.nan])))
    assert m._nanmedian(np.array([1.0, 2.0, 3.0])) == 2.0


# ------------------------------------------------------------- end to end ----

def test_end_to_end_marker_and_control(tmp_path):
    m = _mod()
    X, ct, batch, classes = _data(seed=1, n_per=60)
    cts = [f"CT{c}" for c in ct]
    samples = [f"cell{i}" for i in range(len(ct))]
    bins = [f"chr1:{i * 250000}" for i in range(X.shape[1])]
    mat = tmp_path / "gch.parquet"
    pd.DataFrame(X, index=samples, columns=bins).to_parquet(mat)
    # A mean-signal sibling so --mean-heatmaps auto-derives it (a DIFFERENT
    # matrix over the same cells/bins, so the companion heatmaps have content).
    pd.DataFrame(np.abs(X), index=samples, columns=bins).to_parquet(
        tmp_path / "accessibility_mean.parquet")
    stab = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": samples, "bam_path": ["x"] * len(ct),
                  "cell_type": cts,
                  "sample": ["batch2" if b == 0 else "scNH" for b in batch]}
                 ).to_csv(stab, sep="\t", index=False)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix", f"gch={mat}",
         "--samples", str(stab), "--cell-type-column", "cell_type",
         "--batch-column", "sample", "--n-bins", "30", "--n-permutations", "60",
         "--output-dir", str(out), "--plot-format", "png"],
        capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert (out / "marker_heatmap_gch.png").exists()
    # the two mean-signal companion heatmaps were auto-produced
    assert (out / "marker_heatmap_gch_mean_fixedorder.png").exists()
    assert (out / "marker_heatmap_gch_mean_reclustered.png").exists()
    # no --remove-batch here -> the batch-corrected companion is NOT made
    assert not (out / "marker_heatmap_gch_mean_reclustered_batchcorrected.png"
                ).exists()
    # no compartment matrix available -> no compartment companions
    assert not (out / "marker_heatmap_gch_compartment_fixedorder.png").exists()
    scores = pd.read_csv(out / "bin_scores_gch.tsv", sep="\t")
    assert scores["selected"].sum() == 30
    assert {"celltype_pvalue", "celltype_fdr"} <= set(scores.columns)
    # BED files: all bins + selected-only, with chrom/start/end + the stats.
    allb = pd.read_csv(out / "bin_scores_gch.all.bed", sep="\t")
    selb = pd.read_csv(out / "bin_scores_gch.selected.bed", sep="\t")
    assert list(allb.columns[:3]) == ["chrom", "start", "end"]
    assert {"celltype_icc", "celltype_pvalue", "celltype_fdr", "batch_icc"} \
        <= set(allb.columns)
    assert len(allb) == len(scores)               # all bins
    assert len(selb) == 30                         # only the plotted bins
    assert "selected" not in selb.columns          # implied for the selected file
    assert (allb["end"] - allb["start"] == 250000).all()   # 0-based half-open
    assert allb[["chrom", "start"]].apply(tuple, axis=1).is_monotonic_increasing
    # the out-of-selection control must fire: non-selected bins separate types
    import json
    s = json.loads((out / "marker_bins_summary.json").read_text())["matrices"]["gch"]
    assert s["nonselected_perm_p"] is not None
    assert s["nonselected_perm_p"] < 0.05
    assert s["median_celltype_icc_selected"] > 0.5


def test_remove_batch_adds_batchcorrected_companion(tmp_path):
    """With --remove-batch, a reclustered companion on the batch-corrected mean
    signal is written alongside the raw one."""
    X, ct, batch, _ = _data(seed=2, n_per=60)
    samples = [f"cell{i}" for i in range(len(ct))]
    bins = [f"chr1:{i * 250000}" for i in range(X.shape[1])]
    mat = tmp_path / "gch.parquet"
    pd.DataFrame(X, index=samples, columns=bins).to_parquet(mat)
    pd.DataFrame(np.abs(X), index=samples, columns=bins).to_parquet(
        tmp_path / "accessibility_mean.parquet")
    stab = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": samples, "bam_path": ["x"] * len(ct),
                  "cell_type": [f"CT{c}" for c in ct],
                  "sample": ["batch2" if b == 0 else "scNH" for b in batch]}
                 ).to_csv(stab, sep="\t", index=False)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix", f"gch={mat}",
         "--samples", str(stab), "--cell-type-column", "cell_type",
         "--batch-column", "sample", "--n-bins", "30", "--n-permutations", "40",
         "--remove-batch", "within-celltype",
         "--output-dir", str(out), "--plot-format", "png"],
        capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    # Both orderings, both value versions.  The corrected FIXED-ORDER panel is
    # the one that makes the comparison honest: --remove-batch corrects the
    # concordance matrix too, so the borrowed cell order already comes from
    # corrected values and pairing it with a raw signal compares two different
    # things.
    for suffix in ("fixedorder", "fixedorder_batchcorrected",
                   "reclustered", "reclustered_batchcorrected"):
        assert (out / f"marker_heatmap_gch_mean_{suffix}.png").exists(), suffix


def test_without_remove_batch_no_corrected_companion_is_written(tmp_path):
    """The corrected panels are tied to --remove-batch, not always emitted."""
    X, ct, batch, _ = _data(seed=2, n_per=60)
    samples = [f"cell{i}" for i in range(len(ct))]
    bins = [f"chr1:{i * 250000}" for i in range(X.shape[1])]
    mat = tmp_path / "gch.parquet"
    pd.DataFrame(X, index=samples, columns=bins).to_parquet(mat)
    pd.DataFrame(np.abs(X), index=samples, columns=bins).to_parquet(
        tmp_path / "accessibility_mean.parquet")
    stab = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": samples, "bam_path": ["x"] * len(ct),
                  "cell_type": [f"CT{c}" for c in ct],
                  "sample": ["b1" if b == 0 else "b2" for b in batch]}).to_csv(
        stab, sep="\t", index=False)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix", f"gch={mat}",
         "--samples", str(stab), "--cell-type-column", "cell_type",
         "--batch-column", "sample", "--n-bins", "30", "--n-permutations", "40",
         "--output-dir", str(out), "--plot-format", "png"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "marker_heatmap_gch_mean_fixedorder.png").exists()
    assert not (out / "marker_heatmap_gch_mean_fixedorder_batchcorrected.png"
                ).exists()
    assert not (out / "marker_heatmap_gch_mean_reclustered_batchcorrected.png"
                ).exists()


def test_compartment_companions_with_coarser_grid(tmp_path):
    """--compartment-matrix (a COARSER bin grid, mapped onto the marker bins)
    yields the third companion group; with --remove-batch the batch-corrected
    variant appears too."""
    X, ct, batch, _ = _data(seed=4, n_per=60)
    samples = [f"cell{i}" for i in range(len(ct))]
    bins = [f"chr1:{i * 250000}" for i in range(X.shape[1])]
    mat = tmp_path / "gch.parquet"
    pd.DataFrame(X, index=samples, columns=bins).to_parquet(mat)
    pd.DataFrame(np.abs(X), index=samples, columns=bins).to_parquet(
        tmp_path / "accessibility_mean.parquet")
    # compartment at HALF the resolution (500kb) -> exercises align_to_grid.
    rng = np.random.default_rng(0)
    cbins = [f"chr1:{i * 500000}" for i in range(X.shape[1] // 2)]
    comp = np.tanh(rng.normal(size=(len(ct), len(cbins))))
    comp_path = tmp_path / "compartment_scores.parquet"
    pd.DataFrame(comp, index=samples, columns=cbins).to_parquet(comp_path)
    stab = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": samples, "bam_path": ["x"] * len(ct),
                  "cell_type": [f"CT{c}" for c in ct],
                  "sample": ["batch2" if b == 0 else "scNH" for b in batch]}
                 ).to_csv(stab, sep="\t", index=False)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix", f"gch={mat}",
         "--samples", str(stab), "--cell-type-column", "cell_type",
         "--batch-column", "sample", "--n-bins", "30", "--n-permutations", "40",
         "--compartment-matrix", str(comp_path), "--remove-batch",
         "within-celltype", "--output-dir", str(out), "--plot-format", "png"],
        capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    for suffix in ("fixedorder", "fixedorder_batchcorrected",
                   "reclustered", "reclustered_batchcorrected"):
        assert (out / f"marker_heatmap_gch_compartment_{suffix}.png").exists(), \
            suffix


def test_compartment_cmap_distinct_from_concordance_and_per_context():
    """The compartment palette differs from the concordance map of the SAME
    context and between gch and hcg; all are valid diverging colormaps."""
    import matplotlib
    m = _mod()
    assert m._COMPARTMENT_CMAPS["gch"] != m._CMAPS["gch"]        # != gch main
    assert m._COMPARTMENT_CMAPS["hcg"] != m._CMAPS["hcg"]        # != hcg main
    assert m._COMPARTMENT_CMAPS["gch"] != m._COMPARTMENT_CMAPS["hcg"]   # gch != hcg
    for c in list(m._COMPARTMENT_CMAPS.values()) + m._COMPARTMENT_FALLBACK:
        matplotlib.colormaps.get_cmap(c)                         # valid names


# --------------------------------------------------------- per-bin statistics --

def test_icc_pvalue_tracks_structure_and_bh_fdr():
    """icc_per_region(return_stats=True) gives an ANOVA p-value that is tiny for
    cell-type-structured bins and ~uniform for noise; _bh_fdr(q) >= p, in [0,1],
    and NaN p -> NaN q (unusable bins excluded from the correction)."""
    pytest.importorskip("scipy")
    icc_fn = _icc()
    m = _mod()
    rng = np.random.default_rng(0)
    n, n_ct, n_struct, n_noise = 240, 4, 10, 30
    ct = np.repeat(np.arange(n_ct), n // n_ct)
    struct = rng.normal(0, 3, (n_ct, n_struct))[ct] + rng.normal(0, .3, (n, n_struct))
    noise = rng.normal(0, 1, (n, n_noise))
    X = np.hstack([struct, noise])
    icc, usable, fstat, pval = icc_fn(X, ct, n_ct, return_stats=True)
    assert pval.shape == (n_struct + n_noise,)
    assert np.nanmax(pval[:n_struct]) < 1e-6                 # structured -> tiny p
    assert np.mean(pval[n_struct:] > 0.05) > 0.7            # noise -> mostly n.s.

    fdr = m._bh_fdr(pval)
    fin = np.isfinite(pval)
    assert np.all(fdr[fin] + 1e-12 >= pval[fin])           # BH q >= p
    assert np.all((fdr[fin] >= 0) & (fdr[fin] <= 1))
    p2 = pval.copy(); p2[0] = np.nan
    assert np.isnan(m._bh_fdr(p2)[0])


def test_icc_per_region_return_stats_is_backward_compatible():
    """The 2-tuple default is unchanged; return_stats just appends F + p."""
    icc_fn = _icc()
    rng = np.random.default_rng(1)
    ct = np.repeat(np.arange(3), 20)
    X = rng.normal(size=(60, 5))
    two = icc_fn(X, ct, 3)
    four = icc_fn(X, ct, 3, return_stats=True)
    assert len(two) == 2 and len(four) == 4
    assert np.allclose(two[0], four[0], equal_nan=True)     # same ICC
    assert np.array_equal(two[1], four[1])                   # same usable mask


def test_bins_to_bed_clamps_last_bin_to_chrom_size():
    """The final fixed-width bin must not overshoot the contig length; contigs
    without a size, and chr / no-chr naming, are handled."""
    m = _mod()
    sizes = {"1": 24_900_000}                    # keyed WITHOUT 'chr' (fai/sizes)
    bins = ["chr1:0", "chr1:24500000", "chr1:24750000", "chr2:0"]
    rows = m._bins_to_bed(bins, 250_000, sizes)
    assert rows[0] == ("chr1", 0, 250_000)                     # interior unchanged
    assert rows[2] == ("chr1", 24_750_000, 24_900_000)         # last bin clamped
    assert rows[3] == ("chr2", 0, 250_000)                      # no size -> unchanged
    assert all(e > s for _, s, e in rows)
    # without sizes -> original full-width behaviour
    assert m._bins_to_bed(["chr1:24750000"], 250_000)[0] == \
        ("chr1", 24_750_000, 25_000_000)


def test_read_chrom_sizes_parses_sizes_and_fai(tmp_path):
    m = _mod()
    p = tmp_path / "g.sizes"
    p.write_text("chr1\t195471971\nchr2\t182113224\n")
    assert m._read_chrom_sizes(str(p)) == {"1": 195471971, "2": 182113224}
    fai = tmp_path / "g.fai"                     # .fai has extra columns
    fai.write_text("chr1\t195471971\t7\t60\t61\n")
    assert m._read_chrom_sizes(str(fai))["1"] == 195471971


def test_chrom_sizes_clamps_bed_output(tmp_path):
    """--chrom-sizes clamps the last bin of each chromosome in the BED output."""
    X, ct, batch, _ = _data(seed=3, n_per=60)
    samples = [f"cell{i}" for i in range(len(ct))]
    bins = [f"chr1:{i * 250000}" for i in range(X.shape[1])]
    size = X.shape[1] * 250000 - 100000          # contig ends 100 kb before the
    pd.DataFrame(X, index=samples, columns=bins  # full-width end of the last bin
                 ).to_parquet(tmp_path / "gch.parquet")
    (tmp_path / "chrom.sizes").write_text(f"chr1\t{size}\n")
    pd.DataFrame({"cell_name": samples, "bam_path": ["x"] * len(ct),
                  "cell_type": [f"CT{c}" for c in ct],
                  "sample": ["batch2" if b == 0 else "scNH" for b in batch]}
                 ).to_csv(tmp_path / "s.tsv", sep="\t", index=False)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix", f"gch={tmp_path/'gch.parquet'}",
         "--samples", str(tmp_path / "s.tsv"), "--cell-type-column", "cell_type",
         "--batch-column", "sample", "--n-bins", "20", "--n-permutations", "20",
         "--chrom-sizes", str(tmp_path / "chrom.sizes"),
         "--output-dir", str(out), "--plot-format", "png"],
        capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    allb = pd.read_csv(out / "bin_scores_gch.all.bed", sep="\t")
    assert (allb["end"] <= size).all()                # nothing past the contig
    assert allb["end"].max() == size                  # last bin clamped to it
    assert (allb["end"] > allb["start"]).all()        # still valid intervals


# ------------------------------------------------------------ batch palettes --

def _rgb(t):
    return tuple(round(float(x), 4) for x in t)


def test_each_batch_column_gets_a_distinct_palette():
    """Stacked batch strips must be easy to tell apart: every batch COLUMN draws
    from its own colour family, so no two columns share a colour."""
    m = _mod()
    df = pd.DataFrame({
        "exp_batch":   ["batch2", "scNH"] * 10,
        "plate_batch": [f"plate{9 + (i % 4)}" for i in range(20)],
        "lib_batch":   [f"2024-02-{1 + (i % 12):02d}" for i in range(20)],
    })
    bcols = ["exp_batch", "plate_batch", "lib_batch"]
    colors, order = m._batch_color_maps(df, bcols)

    # Disjoint palettes across columns.
    sets = {c: {_rgb(v) for v in colors[c].values()} for c in bcols}
    assert not (sets["exp_batch"] & sets["plate_batch"])
    assert not (sets["exp_batch"] & sets["lib_batch"])
    assert not (sets["plate_batch"] & sets["lib_batch"])
    # A many-level column still gets one distinct colour per level.
    assert len(sets["lib_batch"]) == len(order["lib_batch"]) == 12


def test_first_batch_column_keeps_green_orange():
    """exp_batch (column 0) preserves the convention batch2=green, scNH=orange."""
    import matplotlib
    m = _mod()
    tab10 = [matplotlib.colors.to_rgb(c)
             for c in matplotlib.colormaps["tab10"].colors]
    df = pd.DataFrame({"exp_batch": ["batch2", "scNH"] * 5})
    colors, _ = m._batch_color_maps(df, ["exp_batch"])
    assert _rgb(colors["exp_batch"]["batch2"]) == _rgb(tab10[2])   # green
    assert _rgb(colors["exp_batch"]["scNH"]) == _rgb(tab10[1])     # orange


def test_batch_colors_are_deterministic_across_matrices():
    """Colours must be identical between the GCH and HCG heatmaps -> the map is
    a pure function of the (full) samples sheet and column order."""
    m = _mod()
    df = pd.DataFrame({
        "exp_batch":   ["batch2", "scNH"] * 6,
        "plate_batch": [f"p{i % 5}" for i in range(12)],
    })
    bcols = ["exp_batch", "plate_batch"]
    a, _ = m._batch_color_maps(df, bcols)
    b, _ = m._batch_color_maps(df, bcols)
    assert all(a[c] == b[c] for c in bcols)


# ------------------------------------------------------- companion heatmaps --

def test_heatmap_order_roundtrip(tmp_path):
    """_heatmap returns its (cells, bins, linkage) layout, and re-passing it as
    order= reproduces the EXACT cell + bin order on a DIFFERENT matrix -- the
    mechanism behind the fixed-order mean-signal companion heatmap."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("scipy")
    m = _mod()
    rng = np.random.default_rng(0)
    cells = [f"c{i}" for i in range(24)]
    bins = [f"chr1:{i * 250000}" for i in range(8)]
    ct = pd.Series([f"CT{i % 3}" for i in range(24)], index=cells)
    batch_df = pd.DataFrame({"b": ["x", "y"] * 12}, index=cells)
    bcolors, border = m._batch_color_maps(batch_df, ["b"])
    conc = pd.DataFrame(rng.normal(size=(24, 8)), index=cells, columns=bins)
    mean = pd.DataFrame(rng.normal(size=(24, 8)), index=cells, columns=bins)

    o1 = m._heatmap(conc, cells, bins, ct, batch_df, {}, bcolors, border,
                    "main", tmp_path / "a.png", "RdBu_r")
    # Fixed order: a DIFFERENT matrix, same layout returned verbatim; the
    # companion shows the RAW level (zscore=False) on a sequential cmap.
    o2 = m._heatmap(mean, cells, bins, ct, batch_df, {}, bcolors, border,
                    "fixed", tmp_path / "b.png", m._mean_cmap("gch"), order=o1,
                    value_label="mean accessibility", zscore=False)
    assert list(o2[0]) == list(o1[0])                 # identical cell order
    assert list(o2[1]) == list(o1[1])                 # identical bin order
    # Reclustered: fresh clustering on the mean matrix; still all bins/cells.
    o3 = m._heatmap(mean, cells, bins, ct, batch_df, {}, bcolors, border,
                    "reclust", tmp_path / "c.png", m._mean_cmap("gch"),
                    value_label="mean accessibility", zscore=False)
    assert sorted(o3[0]) == sorted(cells)
    assert sorted(o3[1]) == sorted(bins)
    for f in ("a.png", "b.png", "c.png"):
        assert (tmp_path / f).exists()


def test_mean_cmap_endpoints_match_nomeseq_convention():
    """GCH accessibility: white(low) -> teal(high); HCG methylation: blue(low)
    -> yellow(high).  Check the palette endpoints encode that."""
    pytest.importorskip("matplotlib")
    import matplotlib
    m = _mod()
    acc = matplotlib.colormaps.get_cmap(m._mean_cmap("gch"))
    lo, hi = acc(0.0)[:3], acc(1.0)[:3]
    assert lo == (1.0, 1.0, 1.0)                       # low accessibility = white
    assert hi[1] > 0.3 and hi[2] > 0.3 and hi[0] < 0.3   # high = teal (G+B, low R)
    meth = matplotlib.colormaps.get_cmap(m._mean_cmap("hcg"))
    b_lo, y_hi = meth(0.0)[:3], meth(1.0)[:3]
    assert b_lo[2] > b_lo[0]                           # low methylation = blue-ish
    assert y_hi[0] > 0.7 and y_hi[1] > 0.7 and y_hi[2] < 0.5   # high = yellow


# ------------------------------------ stable / directional / contrast sets --

class _Args2(_Args):
    sign_min_consistency = 0.6
    sign_reference = "zero"


def _flat_data(seed=0, n_per=40, n_ct=4):
    """Marker bins, FLAT bins (equal cell-type means, low noise), NOISY flat
    bins (equal means, high noise) and batch bins."""
    rng = np.random.default_rng(seed)
    ct = np.repeat(np.arange(n_ct), n_per)
    batch = np.tile([0, 1], n_ct * n_per // 2)
    n = len(ct)
    mk = rng.normal(0, 2, (n_ct, 20))[ct] + rng.normal(0, .3, (n, 20))
    flat = rng.normal(0, .15, (n, 20))                       # equal + quiet
    noisy = rng.normal(0, 3.0, (n, 20))                      # equal + loud
    batchy = np.where(batch[:, None] == 0, 1.5, -1.5) + rng.normal(0, .3, (n, 20))
    X = np.hstack([mk, flat, noisy, batchy])
    classes = np.array(["marker"] * 20 + ["flat"] * 20 + ["noisy"] * 20
                       + ["batchy"] * 20)
    return X, ct, batch, classes


def test_stable_set_finds_flat_bins_not_noisy_ones():
    """The trap this avoids: ranking on LOW cell-type ICC returns the noisiest
    bins, because ICC has sampling noise in its denominator."""
    m = _mod()
    X, ct, batch, classes = _flat_data()
    b_icc = m._batch_icc(X, ct, batch, 2, _icc(), 3, 2)
    sel, spread, _ = m.select_stable_bins(X, ct, 4, b_icc, _Args2(), 20)
    picked = classes[sel]
    assert (picked == "flat").mean() > 0.9, picked
    assert "batchy" not in picked, "a flat-but-batchy bin is sample, not biology"
    # The ICC ranking would have gone the other way, which is the whole point.
    ct_icc, _ = _icc()(X, ct, 4, 3, 3)
    by_icc = classes[np.argsort(np.where(np.isfinite(ct_icc), ct_icc, np.inf))[:20]]
    assert (by_icc == "noisy").mean() > 0.5, by_icc


def test_stable_and_marker_sets_do_not_overlap_when_both_fit():
    """Only when both sets fit the bins that qualify.  Ask for more marker bins
    than there are variable ones and the documented fill takes flat bins, which
    is why this pins n_bins to the 20 the fixture actually plants."""
    m = _mod()
    X, ct, batch, classes = _flat_data()

    class A(_Args2):
        n_bins = 20

    sel, ct_icc, b_icc, _ = m.select_marker_bins(X, ct, 4, [(batch, 2)],
                                                 _icc(), A())
    stable, _, _ = m.select_stable_bins(X, ct, 4, b_icc, A(), 20)
    assert not set(sel.tolist()) & set(stable.tolist())
    assert (classes[sel] == "marker").all()
    assert (classes[stable] == "flat").all()


def _signed_data(seed=0, n_per=30, n_ct=3):
    """Bins that are positive in every cell type, negative in every cell type,
    and ones whose sign flips between them."""
    rng = np.random.default_rng(seed)
    ct = np.repeat(np.arange(n_ct), n_per)
    n = len(ct)
    up = 1.5 + rng.normal(0, .2, (n, 10))
    down = -1.5 + rng.normal(0, .2, (n, 10))
    flip = np.where(ct[:, None] % 2 == 0, 1.2, -1.2) + rng.normal(0, .2, (n, 10))
    X = np.hstack([up, down, flip])
    classes = np.array(["up"] * 10 + ["down"] * 10 + ["flip"] * 10)
    return X, ct, classes


def test_sign_summary_calls_co_open_co_closed_and_refuses_flippers():
    m = _mod()
    X, ct, classes = _signed_data()
    sg = m.sign_summary(X, ct, 3, _Args2())
    assert (sg["always_co_open"][classes == "up"]).all()
    assert (sg["always_co_closed"][classes == "down"]).all()
    # A bin that is co-open in one cell type and co-closed in another is neither.
    assert not sg["always_co_open"][classes == "flip"].any()
    assert not sg["always_co_closed"][classes == "flip"].any()
    assert not (sg["always_co_open"] & sg["always_co_closed"]).any()


def test_sign_needs_every_cell_type_to_agree_not_just_the_mean():
    """A bin whose grand mean is positive only because one cell type is
    strongly positive is not 'always co-open'."""
    m = _mod()
    rng = np.random.default_rng(3)
    ct = np.repeat(np.arange(3), 30)
    x = np.where(ct == 0, 5.0, -0.4)[:, None] + rng.normal(0, .1, (90, 1))
    sg = m.sign_summary(x, ct, 3, _Args2())
    assert sg["d_mean"][0] > 0, "the grand mean IS positive"
    assert not sg["always_co_open"][0], "but two cell types are co-closed"


def test_contrast_finds_the_planted_split_and_scales_with_effect():
    m = _mod()
    rng = np.random.default_rng(7)
    ct = np.repeat(np.arange(4), 30)
    in_mask = ct < 2
    hit = np.where(in_mask, 1.5, -1.5)[:, None] + rng.normal(0, .5, (120, 10))
    miss = rng.normal(0, .5, (120, 10))
    X = np.hstack([hit, miss])
    cs = m.contrast_stats(X, in_mask, 3)
    q = m._bh_fdr(cs["p"])
    assert (q[:10] < 0.01).all(), "the planted bins must survive FDR"
    assert (q[10:] > 0.05).all(), "and the null bins must not"
    assert (cs["d"][:10] > 2).all() and (np.abs(cs["d"][10:]) < 1).all()
    assert np.allclose(cs["diff"][:10], 3.0, atol=0.4)


def test_contrast_is_undefined_when_one_arm_is_too_small():
    m = _mod()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    in_mask = np.zeros(20, bool)
    in_mask[:2] = True                      # 2 cells vs --min-cells-per-group 3
    cs = m.contrast_stats(X, in_mask, 3)
    assert np.isnan(cs["p"]).all(), "too few cells must not produce a p-value"


def test_cramers_v_spots_a_contrast_that_is_really_a_batch():
    m = _mod()
    import pandas as pd
    cells = [f"c{i}" for i in range(40)]
    grp = pd.Series([i < 20 for i in range(40)], index=cells)
    same = pd.Series(["b1"] * 20 + ["b2"] * 20, index=cells)
    mixed = pd.Series(["b1", "b2"] * 20, index=cells)
    assert m._cramers_v(grp, same) > 0.9, "perfectly confounded"
    assert m._cramers_v(grp, mixed) < 0.2, "balanced across batches"


def test_end_to_end_stable_sign_and_contrast(tmp_path):
    """All three new BEDs off one run of the real script."""
    import json
    rng = np.random.default_rng(11)
    n_ct, n_per = 4, 30
    ct = np.repeat(np.arange(n_ct), n_per)
    n = len(ct)
    marker = rng.normal(0, 2, (n_ct, 20))[ct] + rng.normal(0, .3, (n, 20))
    flat = rng.normal(0, .15, (n, 20))
    up = 1.5 + rng.normal(0, .2, (n, 20))          # co-open in every cell type
    down = -1.5 + rng.normal(0, .2, (n, 20))       # co-closed in every one
    split = np.where((ct < 2)[:, None], 2.0, -2.0) + rng.normal(0, .4, (n, 20))
    X = np.hstack([marker, flat, up, down, split])
    cells = [f"cell{i}" for i in range(n)]
    bins = [f"chr1:{i * 250000}" for i in range(X.shape[1])]
    # The filename is what --sign-split auto keys on.
    mat = tmp_path / "concordance_directional.parquet"
    pd.DataFrame(X, index=cells, columns=bins).to_parquet(mat)
    stab = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells, "bam_path": ["x"] * n,
                  "cell_type": [f"CT{c}" for c in ct],
                  "sample": ["b1", "b2"] * (n // 2)}).to_csv(
        stab, sep="\t", index=False)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix", f"gch={mat}",
         "--samples", str(stab), "--cell-type-column", "cell_type",
         "--batch-column", "sample", "--n-bins", "20", "--n-permutations", "40",
         "--contrast", "tumour=CT0,CT1",
         "--output-dir", str(out), "--plot-format", "png"],
        capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"

    def bed(kind):
        return pd.read_csv(out / f"bin_scores_gch.{kind}.bed", sep="\t")

    def block(df):        # which planted block each row came from
        return (df["start"] // 250000 // 20).value_counts()

    # Blocks 1/2/3 all lack cell-type structure -- after per-bin QN they are
    # equally flat, and the invariant set may draw from any of them.  What it
    # must never take is a block that DOES vary by cell type.
    stable = bed("stable")
    assert not {0, 4} & set(block(stable).index), block(stable)
    marker = bed("selected")
    assert block(marker).idxmax() in (0, 4), "the marker set takes the varying ones"
    co_open, co_closed = bed("always_co_open"), bed("always_co_closed")
    assert block(co_open).idxmax() == 2 and len(co_open) == 20
    assert block(co_closed).idxmax() == 3 and len(co_closed) == 20
    assert (co_open["d_group_min"] > 0).all()
    assert (co_closed["d_group_max"] < 0).all()
    contrast = bed("contrast_tumour")
    assert (contrast["contrast_tumour_fdr"] < 0.05).all()
    blk = contrast["start"] // 250000 // 20
    planted = contrast[blk == 4]
    assert len(planted) == 20, "the whole planted split should clear the FDR"
    assert (planted["contrast_tumour_diff"] > 0).all(), "in-set is the high arm"
    # The raw split is +2 against -2, and the difference is reported on that
    # scale rather than on the normalised one.
    assert planted["contrast_tumour_diff"].mean() == pytest.approx(4.0, abs=0.3)
    # Blocks 1/2/3 are identical in every cell, so nothing there can really
    # differ between the arms.  Before the contrast moved to the raw values,
    # the compositional per-cell rank made all 60 of them significant.
    assert int((blk.isin([1, 2, 3])).sum()) <= 6, contrast.groupby(blk).size()
    # Block 0 genuinely varies by cell type, so it may contribute in EITHER
    # direction; that is real signal, not an artefact.
    assert block(contrast).idxmax() == 4, "block 4 is the planted split"

    s = json.loads((out / "marker_bins_summary.json").read_text())["matrices"]["gch"]
    assert s["n_always_co_open"] == 20 and s["n_always_co_closed"] == 20
    assert s["median_spread_stable"] < s["median_spread_marker"]
    c = s["contrasts"]["tumour"]
    assert c["n_cells_in"] == 60 and c["n_cells_out"] == 60
    assert c["batch_confounding_cramers_v"]["sample"] < 0.2, "fixture is balanced"


def test_sign_is_read_off_raw_values_not_normalised_ones(tmp_path):
    """Every raw value is positive here, so every bin is co-open.  The per-cell
    rank + recenter used for SELECTION centres each cell on zero, so a sign read
    after normalisation would find nothing."""
    import json
    rng = np.random.default_rng(5)
    n_ct, n_per = 3, 30
    ct = np.repeat(np.arange(n_ct), n_per)
    n = len(ct)
    X = 2.0 + rng.normal(0, .3, (n, 40)) + rng.normal(0, 1.0, (n_ct, 40))[ct]
    X = np.abs(X)                                   # strictly positive
    cells = [f"cell{i}" for i in range(n)]
    bins = [f"chr1:{i * 250000}" for i in range(X.shape[1])]
    mat = tmp_path / "concordance_directional.parquet"
    pd.DataFrame(X, index=cells, columns=bins).to_parquet(mat)
    stab = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells, "bam_path": ["x"] * n,
                  "cell_type": [f"CT{c}" for c in ct],
                  "sample": ["b1", "b2"] * (n // 2)}).to_csv(
        stab, sep="\t", index=False)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix", f"gch={mat}",
         "--samples", str(stab), "--cell-type-column", "cell_type",
         "--batch-column", "sample", "--n-bins", "10", "--n-permutations", "40",
         "--output-dir", str(out), "--plot-format", "png"],
        capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    s = json.loads((out / "marker_bins_summary.json").read_text())["matrices"]["gch"]
    assert s["n_always_co_open_total"] == 40, "every bin is positive throughout"
    assert s["n_always_co_closed_total"] == 0
    scores = pd.read_csv(out / "bin_scores_gch.tsv", sep="\t")
    # The normalised values the selection ran on are NOT all positive, which is
    # exactly why the sign has to come from the raw matrix.
    assert (scores["frac_cells_positive"] == 1.0).all()


def test_sign_split_off_for_a_matrix_that_is_not_directional(tmp_path):
    """'auto' keys on the filename because only D's sign means open vs closed;
    for C it means concordant vs anti-concordant."""
    import json
    rng = np.random.default_rng(2)
    ct = np.repeat(np.arange(3), 30)
    X = 1.0 + rng.normal(0, .2, (90, 30))
    cells = [f"cell{i}" for i in range(90)]
    bins = [f"chr1:{i * 250000}" for i in range(30)]
    mat = tmp_path / "concordance_coordination.parquet"
    pd.DataFrame(X, index=cells, columns=bins).to_parquet(mat)
    stab = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells, "bam_path": ["x"] * 90,
                  "cell_type": [f"CT{c}" for c in ct],
                  "sample": ["b1", "b2"] * 45}).to_csv(stab, sep="\t", index=False)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix", f"c={mat}",
         "--samples", str(stab), "--cell-type-column", "cell_type",
         "--batch-column", "sample", "--n-bins", "10", "--n-permutations", "40",
         "--output-dir", str(out), "--plot-format", "png"],
        capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert not (out / "bin_scores_c.always_co_open.bed").exists()
    s = json.loads((out / "marker_bins_summary.json").read_text())["matrices"]["c"]
    assert "n_always_co_open" not in s
    # ... but the invariant set is emitted for any matrix.
    assert (out / "bin_scores_c.stable.bed").exists()


def test_contrast_warns_when_the_groups_are_their_own_batch(tmp_path):
    """The realistic failure: tumours sequenced in their own run against cell
    lines from another.  --max-batch-icc cannot catch it, because batch does
    not vary WITHIN a cell type there, so every bin's batch ICC is undefined
    and 'passed the filter' means 'was never tested'."""
    import json
    rng = np.random.default_rng(4)
    ct = np.repeat(["tumor1", "tumor2", "GM12878", "IMR90", "monocyte"], 24)
    n = len(ct)
    tumour = np.isin(ct, ["tumor1", "tumor2"])
    batch = np.where(tumour, "runA", "runB")       # nested inside cell type
    X = np.hstack([np.where(tumour, 2.0, -2.0)[:, None] + rng.normal(0, .4, (n, 20)),
                   rng.normal(0, 1, (n, 20))])
    cells = [f"c{i}" for i in range(n)]
    bins = [f"chr1:{i * 250000}" for i in range(X.shape[1])]
    mat = tmp_path / "concordance_directional.parquet"
    pd.DataFrame(X, index=cells, columns=bins).to_parquet(mat)
    stab = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells, "bam_path": ["x"] * n, "cell_type": ct,
                  "sample": batch}).to_csv(stab, sep="\t", index=False)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix", f"gch={mat}",
         "--samples", str(stab), "--cell-type-column", "cell_type",
         "--batch-column", "sample", "--n-bins", "10", "--n-permutations", "40",
         "--contrast", "meningioma=tumor1,tumor2",
         "--output-dir", str(out), "--plot-format", "png"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "confounded with sample" in r.stderr
    assert "cannot help" in r.stderr, "the nested-batch caveat must be said"
    s = json.loads((out / "marker_bins_summary.json").read_text())
    c = s["matrices"]["gch"]["contrasts"]["meningioma"]
    assert c["batch_confounding_cramers_v"]["sample"] > 0.9
    assert c["n_selected_batch_judged"] == 0, "nothing could be judged"


def test_contrast_on_an_absent_group_is_reported_not_silent(tmp_path):
    rng = np.random.default_rng(0)
    ct = np.repeat(["A", "B", "C"], 30)
    n = len(ct)
    X = rng.normal(0, 1, (n, 30))
    cells = [f"c{i}" for i in range(n)]
    bins = [f"chr1:{i * 250000}" for i in range(30)]
    mat = tmp_path / "m.parquet"
    pd.DataFrame(X, index=cells, columns=bins).to_parquet(mat)
    stab = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells, "bam_path": ["x"] * n, "cell_type": ct,
                  "sample": ["b1", "b2"] * (n // 2)}).to_csv(
        stab, sep="\t", index=False)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix", f"g={mat}",
         "--samples", str(stab), "--cell-type-column", "cell_type",
         "--batch-column", "sample", "--n-bins", "5", "--n-permutations", "40",
         "--contrast", "x=A,nosuchtype", "--contrast", "y=A,B,C",
         "--output-dir", str(out), "--plot-format", "png"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "no cells for ['nosuchtype']" in r.stderr
    assert (out / "bin_scores_g.contrast_x.bed").exists(), "A alone still works"
    # 'every group' leaves nothing to contrast against, so it is skipped.
    assert "skipped" in r.stderr
    assert not (out / "bin_scores_g.contrast_y.bed").exists()


class _SArgsZero(_Args2):
    sign_reference = "zero"


class _SArgsMedian(_Args2):
    sign_reference = "median"


def test_a_mark_that_is_mostly_closed_lands_on_one_side_against_zero():
    """The identity behind the degenerate result: sign(2F-1) = sign(mean level
    - 0.5), so D over a mark whose genome-wide mean is far from 0.5 is
    one-sided everywhere and 'co-closed' merely restates the base rate."""
    m = _mod()
    rng = np.random.default_rng(0)
    ct = np.repeat(np.arange(3), 30)
    nb = 60
    # Every bin negative (a mostly-closed mark) but with real spread BETWEEN
    # bins: some far more negative than others.
    level = rng.normal(-1.0, 0.4, nb)
    X = level + rng.normal(0, 0.15, (90, nb))

    zero = m.sign_summary(X, ct, 3, _SArgsZero())
    assert zero["always_co_closed"].mean() > 0.95
    assert zero["always_co_open"].sum() == 0

    med = m.sign_summary(X, ct, 3, _SArgsMedian())
    # Against the genome-wide median the same data splits, because the question
    # becomes "more co-open than typical" rather than "below 50%".
    assert 0.2 < med["always_co_open"].mean() < 0.8, med["always_co_open"].mean()
    assert 0.2 < med["always_co_closed"].mean() < 0.8


def test_sign_reference_median_is_the_genome_wide_median_of_the_bin_means():
    m = _mod()
    X = np.tile(np.array([[-3.0, -1.0, 1.0]]), (10, 1))
    means = np.nanmean(X, axis=0)
    assert m.sign_reference(means, "median") == -1.0
    assert m.sign_reference(means, "zero") == 0.0


def test_end_to_end_warns_when_one_state_takes_everything(tmp_path):
    """A one-sided split is the expected outcome against zero, so the run has
    to say so rather than let it read as a finding."""
    rng = np.random.default_rng(4)
    n_ct, n_per, nb = 3, 30, 40
    ct = np.repeat(np.arange(n_ct), n_per)
    n = len(ct)
    X = rng.normal(-1.0, 0.4, nb) + rng.normal(0, 0.15, (n, nb))
    cells = [f"cell{i}" for i in range(n)]
    bins = [f"chr1:{i * 250000}" for i in range(nb)]
    mat = tmp_path / "concordance_directional.parquet"
    pd.DataFrame(X, index=cells, columns=bins).to_parquet(mat)
    stab = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells, "bam_path": ["x"] * n,
                  "cell_type": [f"CT{c}" for c in ct],
                  "sample": ["b1", "b2"] * (n // 2)}).to_csv(
        stab, sep="\t", index=False)
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix", f"g={mat}",
         "--samples", str(stab), "--cell-type-column", "cell_type",
         "--batch-column", "sample", "--n-bins", "10", "--n-permutations", "40",
         "--output-dir", str(tmp_path / "out"), "--plot-format", "png"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "fall on one side" in r.stderr
    assert "--sign-reference median" in r.stderr
    # The BEDs are top-N by rank, so the warning is about the sign TEST, not
    # about what was written.
    assert "BEDs are unaffected" in r.stderr


class _SArgsExpected(_Args2):
    sign_reference = "expected"


def test_expected_reference_is_the_independence_prediction():
    m = _mod()
    # p=0.5 -> E[F]=0.5 -> 2E[F]-1 = 0, so the reference is zero whatever C is.
    assert m.expected_directional(np.array([0.5]), np.array([3.0]))[0] == 0.0
    # p=0.8 -> E[F] = .64/(.64+.04) = 0.941; 2E[F]-1 = 0.882
    got = m.expected_directional(np.array([0.8]), np.array([1.0]))[0]
    assert got == pytest.approx(0.64 / 0.68 * 2 - 1, abs=1e-12)
    # The reference scales with C and flips with it.
    assert m.expected_directional(np.array([0.8]), np.array([-1.0]))[0] == \
        pytest.approx(-got)
    # A closed mark predicts co-CLOSED, i.e. a negative reference.
    assert m.expected_directional(np.array([0.2]), np.array([1.0]))[0] < 0


def test_expected_reference_finds_excess_direction_a_global_one_cannot():
    """Two bins, both co-closed in absolute terms, but one is co-closed by MORE
    than its own level predicts and the other by less.  A single genome-wide
    threshold cannot separate them; each bin's own prediction can."""
    m = _mod()
    rng = np.random.default_rng(0)
    ct = np.repeat(np.arange(3), 30)
    n = len(ct)
    # bin 0: level 0.2 (mostly closed), bin 1: level 0.35 (less so).
    p = np.array([0.2, 0.35])
    C = np.array([1.0, 1.0])
    ref = m.expected_directional(p, C)                    # both negative
    # Bin 0 sits ABOVE its prediction, bin 1 BELOW — opposite excesses, and
    # both raw values are negative, so zero calls them both co-closed.
    D = np.vstack([np.full(n, ref[0] + 0.25), np.full(n, ref[1] - 0.25)]).T
    D = D + rng.normal(0, 0.02, D.shape)
    assert (D < 0).all(), "both bins are co-closed in absolute terms"

    zero = m.sign_summary(D, ct, 3, _SArgsZero())
    assert zero["always_co_closed"].all(), "zero cannot tell them apart"

    exp = m.sign_summary(D, ct, 3, _SArgsExpected(),
                         expected=np.tile(ref, (n, 1)))
    assert exp["always_co_open"][0] and not exp["always_co_closed"][0]
    assert exp["always_co_closed"][1] and not exp["always_co_open"][1]


def test_expected_reference_without_the_components_stops_the_run():
    m = _mod()
    X = np.zeros((30, 4))
    ct = np.repeat(np.arange(3), 10)
    with pytest.raises(SystemExit, match="marginal and coordination"):
        m.sign_summary(X, ct, 3, _SArgsExpected())


def test_end_to_end_expected_reference_reads_the_sibling_components(tmp_path):
    rng = np.random.default_rng(6)
    n_ct, n_per, nb = 3, 30, 40
    ct = np.repeat(np.arange(n_ct), n_per)
    n = len(ct)
    cells = [f"cell{i}" for i in range(n)]
    bins = [f"chr1:{i * 250000}" for i in range(nb)]
    p = rng.uniform(0.1, 0.4, nb)                 # a mostly-closed mark
    C = np.abs(rng.normal(1.0, 0.2, nb))
    import importlib.util as _il
    spec = _il.spec_from_file_location("mk2", _SCRIPT)
    mm = _il.module_from_spec(spec)
    spec.loader.exec_module(mm)
    ref = mm.expected_directional(p, C)
    # Small relative to |ref| (0.38..0.88 here), so D stays negative in
    # every bin -- one-sided against zero -- while still splitting around
    # each bin's OWN reference.
    excess = rng.normal(0, 0.08, nb)
    D = ref + excess + rng.normal(0, 0.01, (n, nb))
    out_dir = tmp_path / "ctx"
    out_dir.mkdir()
    pd.DataFrame(D, index=cells, columns=bins).to_parquet(
        out_dir / "concordance_directional.parquet")
    pd.DataFrame(np.tile(p, (n, 1)), index=cells, columns=bins).to_parquet(
        out_dir / "concordance_marginal.parquet")
    pd.DataFrame(np.tile(C, (n, 1)), index=cells, columns=bins).to_parquet(
        out_dir / "concordance_coordination.parquet")
    stab = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells, "bam_path": ["x"] * n,
                  "cell_type": [f"CT{c}" for c in ct],
                  "sample": ["b1", "b2"] * (n // 2)}).to_csv(
        stab, sep="\t", index=False)

    import json

    def run(mode, sub):
        r = subprocess.run(
            [sys.executable, str(_SCRIPT), "--matrix",
             f"g={out_dir / 'concordance_directional.parquet'}",
             "--samples", str(stab), "--cell-type-column", "cell_type",
             "--batch-column", "sample", "--n-bins", "10",
             "--n-permutations", "40", "--sign-reference", mode,
             "--output-dir", str(tmp_path / sub), "--plot-format", "png"],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        # The *_total counts, not the BED membership flags: those are capped
        # at --n-bins, so they measure the cap rather than the classification.
        s = json.loads(
            (tmp_path / sub / "marker_bins_summary.json").read_text())
        return (pd.read_csv(tmp_path / sub / "bin_scores_g.tsv", sep="\t"),
                s["matrices"]["g"], r)

    _, zsum, _ = run("zero", "z")
    assert zsum["n_always_co_closed_total"] == nb, "one-sided against zero"
    assert zsum["n_always_co_open_total"] == 0
    exp, esum, res = run("expected", "e")
    assert "expected-D reference" in res.stderr
    # Each bin now has its OWN reference, so the column is not constant.
    assert exp["d_reference"].nunique() > 1
    frac_open = esum["n_always_co_open_total"] / nb
    assert 0.2 < frac_open < 0.8, frac_open
    assert (tmp_path / "e" / "bin_scores_g.always_co_open.bed").exists()


def test_sign_bins_are_top_n_by_rank_not_by_threshold():
    """A reference biased the same way for every bin slides a threshold but
    cannot move a ranking — which is the whole reason for a fixed size."""
    m = _mod()
    rng = np.random.default_rng(0)
    ct = np.repeat(np.arange(3), 20)
    n, nb = len(ct), 200
    strength = np.linspace(-1.0, 1.0, nb)
    D = strength + rng.normal(0, 0.02, (n, nb))
    sg = m.sign_summary(D, ct, 3, _SArgsZero())
    open_idx, n_agree = m.select_sign_bins(sg, "always_co_open",
                                           "d_group_min", False, 50)
    assert open_idx.size == 50 and n_agree == 50
    assert set(open_idx.tolist()) == set(range(nb - 50, nb)), "the most positive"

    # Shift every bin by a constant, as a biased reference would: now nearly
    # everything passes the threshold, yet the SAME 50 bins are selected.
    shifted = m.sign_summary(D + 0.9, ct, 3, _SArgsZero())
    assert shifted["always_co_open"].mean() > 0.9, "the threshold has moved"
    again, _ = m.select_sign_bins(shifted, "always_co_open", "d_group_min",
                                  False, 50)
    np.testing.assert_array_equal(np.sort(open_idx), np.sort(again))


def test_sign_bins_fill_by_rank_when_too_few_agree():
    m = _mod()
    rng = np.random.default_rng(1)
    ct = np.repeat(np.arange(3), 20)
    n, nb = len(ct), 100
    # Only 10 bins are positive in every cell type; the rest are negative.
    strength = np.r_[np.full(10, 0.5), np.full(90, -0.5)]
    D = strength + rng.normal(0, 0.02, (n, nb))
    sg = m.sign_summary(D, ct, 3, _SArgsZero())
    idx, n_agree = m.select_sign_bins(sg, "always_co_open", "d_group_min",
                                      False, 40)
    assert idx.size == 40, "the set is always the requested size"
    assert n_agree == 10, "and reports how many actually agreed"
    assert set(idx[:10].tolist()) == set(range(10)), "agreeing ones come first"


def test_both_sign_beds_are_written_at_the_requested_size(tmp_path):
    import json
    rng = np.random.default_rng(5)
    n_ct, n_per, nb = 3, 30, 300
    ct = np.repeat(np.arange(n_ct), n_per)
    n = len(ct)
    cells = [f"cell{i}" for i in range(n)]
    bins = [f"chr1:{i * 250000}" for i in range(nb)]
    # Everything positive: a threshold would call all 300 co-open and leave
    # the co-closed BED empty.
    D = np.linspace(0.1, 2.0, nb) + rng.normal(0, 0.02, (n, nb))
    mat = tmp_path / "concordance_directional.parquet"
    pd.DataFrame(D, index=cells, columns=bins).to_parquet(mat)
    stab = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells, "bam_path": ["x"] * n,
                  "cell_type": [f"CT{c}" for c in ct],
                  "sample": ["b1", "b2"] * (n // 2)}).to_csv(
        stab, sep="\t", index=False)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix", f"g={mat}",
         "--samples", str(stab), "--cell-type-column", "cell_type",
         "--batch-column", "sample", "--n-bins", "40", "--n-sign-bins", "60",
         "--n-permutations", "40", "--output-dir", str(out),
         "--plot-format", "png"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    op = pd.read_csv(out / "bin_scores_g.always_co_open.bed", sep="\t")
    cl = pd.read_csv(out / "bin_scores_g.always_co_closed.bed", sep="\t")
    assert len(op) == 60 and len(cl) == 60, "both sets, both the asked size"
    # Comparable in size to the marker and invariant sets, which is the point.
    assert len(pd.read_csv(out / "bin_scores_g.selected.bed", sep="\t")) == 40
    assert len(pd.read_csv(out / "bin_scores_g.stable.bed", sep="\t")) == 40
    s = json.loads((out / "marker_bins_summary.json").read_text())["matrices"]["g"]
    assert s["n_always_co_closed"] == 60
    assert s["n_always_co_closed_agreeing"] == 0, "none truly co-closed here"
    assert s["n_always_co_open_agreeing"] == 60
    # The two sets are the opposite ends of the same ranking.
    assert set(op["start"]) & set(cl["start"]) == set()


# ------------------------------------------------ stability versus the level --

def _level_crossed(seed=0, n_ct=4, n_per=30, per_class=25):
    """Four bin classes crossing "flat across cell types" with "level of D",
    so a set that is level-blind can be caught taking both halves."""
    rng = np.random.default_rng(seed)
    ct = np.repeat(np.arange(n_ct), n_per)
    n = len(ct)
    blocks, classes = [], []
    for label, level, spread in (("flat_LOW", -1.5, 0.0),
                                 ("flat_HIGH", 1.5, 0.0),
                                 ("var_LOW", -1.5, 1.0),
                                 ("var_HIGH", 1.5, 1.0)):
        ctm = rng.normal(0, spread, (n_ct, per_class))
        blocks.append(level + ctm[ct] + rng.normal(0, 0.15, (n, per_class)))
        classes += [label] * per_class
    return np.hstack(blocks), ct, np.array(classes)


def _run_level(tmp_path, sub, extra=None):
    X, ct, classes = _level_crossed()
    n = len(ct)
    cells = [f"c{i}" for i in range(n)]
    bins = [f"chr1:{i * 250000}" for i in range(X.shape[1])]
    d = tmp_path / sub
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X, index=cells, columns=bins).to_parquet(
        d / "concordance_directional.parquet")
    pd.DataFrame({"cell_name": cells, "bam_path": ["x"] * n,
                  "cell_type": [f"CT{c}" for c in ct],
                  "sample": ["b1", "b2"] * (n // 2)}).to_csv(
        d / "s.tsv", sep="\t", index=False)
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix",
         f"g={d / 'concordance_directional.parquet'}",
         "--samples", str(d / "s.tsv"), "--cell-type-column", "cell_type",
         "--batch-column", "sample", "--n-bins", "30", "--n-permutations", "40",
         "--output-dir", str(d / "o"), "--plot-format", "png"] + (extra or []),
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    sc = pd.read_csv(d / "o" / "bin_scores_g.tsv", sep="\t")
    sc["class"] = classes
    return sc, r


def test_stability_is_blind_to_the_level_by_design(tmp_path):
    """The answer to 'are the stable bins always highly concordant': no.  Level
    is invisible to the ranking, which runs after per-bin QN."""
    sc, _ = _run_level(tmp_path, "any")
    st = sc[sc["stable"]]
    kinds = set(st["class"])
    assert kinds == {"flat_LOW", "flat_HIGH"}, kinds
    # Both halves, in comparable numbers -- not one or the other.
    frac_low = (st["class"] == "flat_LOW").mean()
    assert 0.25 < frac_low < 0.75, frac_low
    # And the spread statistic genuinely cannot separate them.
    by = sc.groupby("class")["celltype_mean_spread"].mean()
    assert abs(by["flat_HIGH"] - by["flat_LOW"]) < 0.2 * by["var_HIGH"]


def test_bin_mean_raw_is_reported_so_the_level_can_be_checked(tmp_path):
    sc, _ = _run_level(tmp_path, "col")
    assert "bin_mean_raw" in sc.columns
    by = sc.groupby("class")["bin_mean_raw"].mean()
    assert by["flat_HIGH"] > 1.0 and by["flat_LOW"] < -1.0
    # Present in the BEDs too, which is where it is actually needed.
    for kind in ("all", "selected", "stable"):
        bed = pd.read_csv(tmp_path / "col" / "o" / f"bin_scores_g.{kind}.bed",
                          sep="\t")
        assert "bin_mean_raw" in bed.columns, kind


def test_stable_level_high_keeps_only_the_high_half(tmp_path):
    # 20 < the 25 bins in the eligible class, so no fill from outside the pool
    # is needed -- the fill is documented behaviour and would blur the point.
    sc, r = _run_level(tmp_path, "high", ["--stable-level", "high",
                                          "--n-stable-bins", "20"])
    st = sc[sc["stable"]]
    assert set(st["class"]) == {"flat_HIGH"}, set(st["class"])
    assert (st["bin_mean_raw"] > 0).all()
    assert "--stable-level high keeps" in r.stderr


def test_stable_level_low_keeps_only_the_low_half(tmp_path):
    sc, _ = _run_level(tmp_path, "low", ["--stable-level", "low",
                                         "--n-stable-bins", "20"])
    st = sc[sc["stable"]]
    assert set(st["class"]) == {"flat_LOW"}, set(st["class"])
    assert (st["bin_mean_raw"] < 0).all()


def test_stable_level_still_ranks_by_flatness_within_the_pool(tmp_path):
    """The restriction narrows the pool; it does not become the ranking.

    Compared within the BATCH-CLEAN part of the pool, because the batch filter
    takes precedence over the spread ranking -- a flat bin that batch drives is
    not invariant, so it is passed over however flat it looks.
    """
    sc, _ = _run_level(tmp_path, "rank", ["--stable-level", "high",
                                          "--n-stable-bins", "20"])
    med = sc["bin_mean_raw"].median()
    clean = ~(sc["batch_icc"].notna() & (sc["batch_icc"] > 0.1))
    pool = sc[(sc["bin_mean_raw"] > med) & clean]
    st = pool[pool["stable"]]
    rest = pool[~pool["stable"]]
    assert len(st) and len(rest)
    assert st["celltype_mean_spread"].max() <= rest["celltype_mean_spread"].min()


# ------------------------------------------------------------ the CNV check --

def _cnv_case(tmp_path, sub, cnv_drives=True, extra=None, seed=0):
    """Two 'tumour' cell types against two 'line' ones.

    With cnv_drives, the ONLY thing separating the arms is a deleted arm of the
    genome -- the shape of a tumour-vs-line contrast where copy number, not
    coordination, does the separating.
    """
    rng = np.random.default_rng(seed)
    n_ct, n_per, nb = 4, 20, 120
    ct = np.repeat(np.arange(n_ct), n_per)
    n = len(ct)
    tumour = ct < 2
    X = rng.normal(0, 0.4, (n, nb))
    half = nb // 3                                   # the "deleted" region
    # A SMALL affected region either way.  Per-cell rank normalisation is
    # compositional -- pushing most of the genome down pushes the rest up -- so
    # shifting a large block would plant a mirror-image effect in its
    # complement and both fixtures would look confounded.
    if cnv_drives:
        X[tumour, :20] -= 2.0                        # inside the deletion
    else:
        X[tumour, half:half + 20] -= 2.0             # copy-neutral ground
    cells = [f"c{i}" for i in range(n)]
    bins = [f"chr1:{i * 250000}" for i in range(nb)]
    d = tmp_path / sub
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X, index=cells, columns=bins).to_parquet(
        d / "concordance_directional.parquet")
    pd.DataFrame({"cell_name": cells, "bam_path": ["x"] * n,
                  "cell_type": [f"CT{c}" for c in ct],
                  "donor": np.where(tumour, "tumour1", "line1"),
                  "sample": ["b1", "b2"] * (n // 2)}).to_csv(
        d / "s.tsv", sep="\t", index=False)
    # Segmentation: the tumour donors are deleted over the first third only.
    rows = []
    for donor, lost in (("tumour1", True), ("line1", False)):
        rows.append((donor, "chr1", 0, half * 250000, 10, -1.0 if lost else 0.0))
        rows.append((donor, "chr1", half * 250000, nb * 250000, 10, 0.0))
    pd.DataFrame(rows).to_csv(d / "cnv.seg", sep="\t", index=False,
                              header=["ID", "chrom", "start", "end",
                                      "num_mark", "seg.mean"])
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix",
         f"g={d / 'concordance_directional.parquet'}",
         "--samples", str(d / "s.tsv"), "--cell-type-column", "cell_type",
         "--batch-column", "sample", "--n-bins", "20", "--n-permutations", "20",
         "--contrast", "tumour=CT0,CT1", "--cnv-seg", str(d / "cnv.seg"),
         "--cnv-sample-column", "donor",
         "--output-dir", str(d / "o"), "--plot-format", "png"] + (extra or []),
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    import json
    s = json.loads((d / "o" / "marker_bins_summary.json").read_text())
    return s["matrices"]["g"]["contrasts"]["tumour"], r, d


def test_segments_are_mapped_onto_bins_by_overlap():
    """A 250 kb bin routinely spans a segment boundary; the midpoint's segment
    would give it a copy number it only half has."""
    m = _mod()
    bins = ["chr1:0", "chr1:250000"]
    seg = pd.DataFrame({"sample": ["s1"] * 2, "chrom": ["1", "1"],
                        "start": [0, 125000], "end": [125000, 500000],
                        "value": [-2.0, 0.0]})
    samples, mat = m.segment_bin_means(bins, seg)
    assert list(samples) == ["s1"]
    assert mat[0, 0] == pytest.approx(-1.0), "half the bin at -2, half at 0"
    assert mat[0, 1] == pytest.approx(0.0)


def test_a_contrast_that_is_copy_number_is_called_out(tmp_path):
    info, r, _ = _cnv_case(tmp_path, "driven", cnv_drives=True)
    assert "tracks COPY NUMBER" in r.stderr
    assert info["cnv"]["frac_selected_over_delta"] > 0.5


def test_the_warning_keys_on_where_the_bins_are_not_on_the_correlation():
    """Two step functions over complementary regions correlate near-perfectly
    while the effect sits exactly where the copy number does not; a correlation
    criterion alone would flag that clean result."""
    effect = np.r_[np.zeros(40), np.full(80, -2.0)]
    cnv = np.r_[np.full(40, -1.0), np.zeros(80)]
    from scipy.stats import pearsonr
    assert abs(pearsonr(cnv, effect)[0]) > 0.9        # correlation says "yes"
    selected = np.arange(40, 120)                     # ...but nothing selected
    assert np.mean(np.abs(cnv[selected]) > 0.2) == 0  # sits in a CNV region


def test_a_copy_neutral_contrast_is_not_flagged(tmp_path):
    info, r, _ = _cnv_case(tmp_path, "neutral", cnv_drives=False)
    assert "tracks COPY NUMBER" not in r.stderr
    assert info["cnv"]["frac_selected_over_delta"] < 0.3


def test_cnv_max_delta_removes_the_affected_bins_from_the_pool(tmp_path):
    info, r, d = _cnv_case(tmp_path, "filtered", cnv_drives=True,
                           extra=["--cnv-max-delta", "0.5"])
    assert info["cnv"]["n_bins_dropped"] == 40, "the deleted third"
    assert "removed 40 bins from the pool" in r.stderr
    bed = pd.read_csv(d / "o" / "bin_scores_g.contrast_tumour.bed", sep="\t")
    # Nothing selected sits in the deleted region any more.
    assert (bed["start"] >= 40 * 250000).all()


def test_the_cnv_delta_is_reported_per_bin(tmp_path):
    _info, _r, d = _cnv_case(tmp_path, "col", cnv_drives=True)
    sc = pd.read_csv(d / "o" / "bin_scores_g.tsv", sep="\t")
    col = "contrast_tumour_cnv_delta"
    assert col in sc.columns
    assert sc[col].iloc[:40].mean() == pytest.approx(-1.0)
    assert sc[col].iloc[40:].mean() == pytest.approx(0.0)


def test_unmatched_cnv_sample_ids_stop_the_run(tmp_path):
    m = _mod()
    _info, _r, d = _cnv_case(tmp_path, "match", cnv_drives=False)
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix",
         f"g={d / 'concordance_directional.parquet'}",
         "--samples", str(d / "s.tsv"), "--cell-type-column", "cell_type",
         "--batch-column", "sample", "--n-bins", "20", "--n-permutations", "20",
         "--contrast", "tumour=CT0,CT1", "--cnv-seg", str(d / "cnv.seg"),
         "--output-dir", str(d / "o2"), "--plot-format", "png"],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "--cnv-sample-column" in r.stderr


def test_seg_and_headed_formats_both_read():
    m = _mod()
    import io
    headed = io.StringIO("sample\tchrom\tstart\tend\tvalue\ns1\tchr1\t0\t100\t-1\n")
    df = m.read_segments(headed)
    assert list(df.columns) == ["sample", "chrom", "start", "end", "value"]
    assert df["chrom"].iloc[0] == "1", "chr prefix stripped for matching"


# ============================================================ contrast arm ===

def test_contrast_stats_uses_the_named_reference_only():
    """A tumour-vs-monocyte contrast is not tumour-vs-everything.  Folding the
    other cell types into the reference changes both the difference and which
    bins clear the cut, silently."""
    m = _mod()
    X = np.zeros((9, 2))
    X[0:3] = 5.0        # arm
    X[3:6] = 1.0        # reference
    X[6:9] = 100.0      # a third group that must not be in the reference
    X += np.tile([[0.0], [0.1], [-0.1]], (3, 2))   # a little within-group spread
    in_mask = np.array([True] * 3 + [False] * 6)
    ref_mask = np.array([False] * 3 + [True] * 3 + [False] * 3)

    named = m.contrast_stats(X, in_mask, 2, ref_mask)
    everything = m.contrast_stats(X, in_mask, 2)
    assert named["mean_out"][0] == pytest.approx(1.0, abs=0.05)
    assert named["diff"][0] == pytest.approx(4.0, abs=0.1)
    # Against "everything else" the reference mean is dragged up by the third
    # group and the effect flips sign.
    assert everything["diff"][0] < 0


def test_contrast_stats_defaults_to_everything_else():
    m = _mod()
    X = np.array([[3.0], [3.1], [1.0], [1.1]])
    in_mask = np.array([True, True, False, False])
    a = m.contrast_stats(X, in_mask, 2)
    b = m.contrast_stats(X, in_mask, 2, ~in_mask)
    assert a["diff"][0] == pytest.approx(b["diff"][0])


# ------------------------------------------------------- cross-modality ------

def _ctx(diff, fdr, sig, keys=None):
    keys = keys or [f"chr1:{i * 250000}" for i in range(len(diff))]
    return pd.DataFrame({"diff": diff, "fdr": fdr, "sig": sig}, index=keys)


def test_joint_table_labels_which_modality_a_bin_is_aberrant_in():
    m = _mod()
    per = {"gch": _ctx([1.0, 1.0, 0.0, 0.0], [0.01, 0.01, 0.9, 0.9],
                       [True, True, False, False]),
           "hcg": _ctx([1.0, 0.0, 1.0, 0.0], [0.01, 0.9, 0.01, 0.9],
                       [True, False, True, False])}
    tab, a, b = m.joint_contrast_table(per, "x")
    assert list(tab["category"]) == ["both", "gch_only", "hcg_only", "neither"]


def test_joint_table_drops_bins_one_context_never_measured():
    """A bin GCH never saw is not evidence that the aberration is HCG-specific;
    counting it would pad the modality-specific classes with missing data."""
    m = _mod()
    keys = ["chr1:0", "chr1:250000"]
    per = {"gch": _ctx([1.0], [0.01], [True], keys=["chr1:0"]),
           "hcg": _ctx([1.0, 2.0], [0.01, 0.01], [True, True], keys=keys)}
    tab, _, _ = m.joint_contrast_table(per, "x")
    assert list(tab.index) == ["chr1:0"]


def test_joint_table_needs_exactly_two_contexts():
    m = _mod()
    with pytest.raises(ValueError, match="two contexts"):
        m.joint_contrast_table({"gch": _ctx([1.0], [0.01], [True])}, "x")


def test_concordance_counts_direction_only_over_shared_bins():
    """Whether a bin moves the same way in a modality that did not move at all
    is a coin flip, not a result."""
    m = _mod()
    per = {"gch": _ctx([1.0, 1.0, -1.0, 5.0], [0.01] * 3 + [0.9],
                       [True, True, True, False]),
           "hcg": _ctx([1.0, -1.0, -1.0, -5.0], [0.01] * 3 + [0.9],
                       [True, True, True, False])}
    tab, a, b = m.joint_contrast_table(per, "x")
    st = m.joint_concordance(tab, a, b)
    assert st["n_both"] == 3
    assert st["n_both_same_direction"] == 2          # bins 0 and 2
    assert st["frac_both_same_direction"] == pytest.approx(2 / 3)
    assert st["n_gch_only"] == 0 and st["n_hcg_only"] == 0


def test_concordance_with_nothing_shared_does_not_divide_by_zero():
    m = _mod()
    per = {"gch": _ctx([1.0], [0.01], [True]),
           "hcg": _ctx([1.0], [0.9], [False])}
    tab, a, b = m.joint_contrast_table(per, "x")
    st = m.joint_concordance(tab, a, b)
    assert st["n_both"] == 0
    assert np.isnan(st["frac_both_same_direction"])


def test_volcano_writes_a_figure_and_counts_both_directions(tmp_path):
    m = _mod()
    diff = np.array([2.0, -2.0, 0.1, np.nan])
    fdr = np.array([0.001, 0.001, 0.9, 0.5])
    sig = np.array([True, True, False, False])
    up, down = m.volcano_figure(diff, fdr, sig, tmp_path / "v.pdf", "t", 0.05)
    assert (up, down) == (1, 1)
    assert (tmp_path / "v.pdf").stat().st_size > 0


def test_volcano_survives_an_fdr_of_exactly_zero(tmp_path):
    """-log10(0) is inf and would blow up the axis."""
    m = _mod()
    m.volcano_figure(np.array([2.0, 1.0]), np.array([0.0, 0.02]),
                     np.array([True, True]), tmp_path / "v.pdf", "t", 0.05)
    assert (tmp_path / "v.pdf").stat().st_size > 0


def test_the_contrast_is_computed_on_raw_values_not_normalised_ones(tmp_path):
    """Per-bin QN followed by per-cell rank is compositional: a real shift in
    one block pushes every other bin the other way in rank space, so bins that
    are IDENTICAL in every cell pick up a difference between the arms out of
    nothing.  A ranked top-N hid it by taking only the strongest bins; an FDR
    cut does not, and would fill the BED with normalisation artefacts.
    """
    rng = np.random.default_rng(3)
    n_ct, n_per = 3, 20
    ct = np.repeat(np.arange(n_ct), n_per)
    n = len(ct)
    # 40 bins that are constant in every cell, and 40 carrying a real split.
    constant = np.tile(np.linspace(-1, 1, 40), (n, 1)) + rng.normal(0, .02, (n, 40))
    split = np.where((ct < 1)[:, None], 2.0, -2.0) + rng.normal(0, .3, (n, 40))
    X = np.hstack([constant, split])
    cells = [f"cell{i}" for i in range(n)]
    bins = [f"chr1:{i * 250000}" for i in range(X.shape[1])]
    mat = tmp_path / "concordance_directional.parquet"
    pd.DataFrame(X, index=cells, columns=bins).to_parquet(mat)
    stab = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells, "bam_path": ["x"] * n,
                  "cell_type": [f"CT{c}" for c in ct],
                  "sample": ["b1", "b2"] * (n // 2)}).to_csv(
        stab, sep="\t", index=False)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix", f"gch={mat}",
         "--samples", str(stab), "--cell-type-column", "cell_type",
         "--batch-column", "sample", "--n-bins", "20", "--n-permutations", "20",
         "--contrast", "arm=CT0", "--output-dir", str(out),
         "--plot-format", "png"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    bed = pd.read_csv(out / "bin_scores_gch.contrast_arm.bed", sep="\t")
    blk = bed["start"] // 250000 // 40
    n_constant, n_split = int((blk == 0).sum()), int((blk == 1).sum())
    assert n_split >= 35, "the real split should be recovered"
    assert n_constant <= 4, (
        f"{n_constant}/40 constant bins called significant -- the contrast is "
        f"seeing normalised values, where the split makes them differ")


def test_mwu_min_pvalue_is_the_number_of_orderings():
    """A rank test has C(n1+n2, n1) distinct orderings, so its p has a floor."""
    m = _mod()
    assert m.mwu_min_pvalue(3, 3) == pytest.approx(2 / 20)
    assert m.mwu_min_pvalue(4, 4) == pytest.approx(2 / 70)
    assert m.mwu_min_pvalue(16, 10) == pytest.approx(2 / 5311735)
    assert m.mwu_min_pvalue(1, 1) == pytest.approx(1.0)     # clamped
    assert np.isnan(m.mwu_min_pvalue(0, 5))


def test_contrast_stats_mwu_matches_scipy():
    m = _mod()
    from scipy.stats import mannwhitneyu
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    in_mask = np.array([True] * 8 + [False] * 12)
    got = m.contrast_stats(X, in_mask, 3, test="mwu")["p"]
    want = mannwhitneyu(X[in_mask], X[~in_mask], axis=0,
                        alternative="two-sided").pvalue
    np.testing.assert_allclose(got, want)


def test_the_difference_is_of_means_under_either_test():
    """The rank test supplies the p-value, not the effect size: the volcano's
    x-axis stays a difference in D units."""
    m = _mod()
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3))
    in_mask = np.array([True] * 8 + [False] * 12)
    a = m.contrast_stats(X, in_mask, 3, test="mwu")
    b = m.contrast_stats(X, in_mask, 3, test="welch")
    np.testing.assert_allclose(a["diff"], b["diff"])
    assert not np.allclose(a["p"], b["p"])


def test_mwu_is_not_carried_by_a_single_outlying_cell():
    """One cell with an extreme value drags a t; it moves a rank test by one
    rank.  Single-cell values have those cells."""
    m = _mod()
    base = np.array([[0.0], [0.1], [-0.1], [0.05], [-0.05], [0.0]])
    arm = np.vstack([base, [[500.0]]])                  # 7th cell is wild
    ref = base.copy()
    X = np.vstack([arm, ref])
    in_mask = np.array([True] * 7 + [False] * 6)
    welch = m.contrast_stats(X, in_mask, 3, test="welch")["p"][0]
    mwu = m.contrast_stats(X, in_mask, 3, test="mwu")["p"][0]
    assert mwu > 0.3, "the rank test should not be moved by one cell"
    assert mwu > welch


def test_an_unknown_contrast_test_is_refused():
    m = _mod()
    with pytest.raises(ValueError, match="mwu.*welch"):
        m.contrast_stats(np.zeros((4, 2)), np.array([1, 1, 0, 0], bool), 1,
                         test="ttest")


def test_the_contrast_sees_batch_corrected_values(tmp_path):
    """Within-cell-type batch structure is nuisance spread inside each arm.
    Left in, it inflates the standard error and buries real bins; _remove_batch
    takes it out while leaving the between-cell-type means alone, so the
    contrast it feeds is the same comparison with less noise.
    """
    rng = np.random.default_rng(4)
    n_ct, n_per = 3, 20
    ct = np.repeat(np.arange(n_ct), n_per)
    n = len(ct)
    batch = np.tile(["b1", "b2"], n // 2)
    X = rng.normal(0, .25, (n, 200))
    X[:, :40] += np.where((ct == 0)[:, None], 1.2, 0.0)        # the real split
    # A big within-cell-type batch shift on every bin: pure nuisance.
    X += np.where((batch == "b1")[:, None], 2.5, -2.5)

    cells = [f"cell{i}" for i in range(n)]
    bins = [f"chr1:{i * 250000}" for i in range(X.shape[1])]
    mat = tmp_path / "concordance_directional.parquet"
    pd.DataFrame(X, index=cells, columns=bins).to_parquet(mat)
    stab = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells, "bam_path": ["x"] * n,
                  "cell_type": [f"CT{c}" for c in ct],
                  "sample": batch}).to_csv(stab, sep="\t", index=False)

    def run(sub, extra):
        out = tmp_path / sub
        r = subprocess.run(
            [sys.executable, str(_SCRIPT), "--matrix", f"gch={mat}",
             "--samples", str(stab), "--cell-type-column", "cell_type",
             "--batch-column", "sample", "--n-bins", "20",
             "--n-permutations", "20", "--contrast", "arm=CT0",
             "--output-dir", str(out), "--plot-format", "png"] + extra,
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        bed = pd.read_csv(out / "bin_scores_gch.contrast_arm.bed", sep="\t")
        blk = bed["start"] // 250000 // 40
        return int((blk == 0).sum())

    uncorrected = run("plain", [])
    corrected = run("fixed", ["--remove-batch", "within-celltype"])
    assert corrected >= uncorrected, (
        f"batch correction should not lose real bins "
        f"({corrected} vs {uncorrected})")
    assert corrected >= 30, f"only {corrected}/40 planted bins recovered"


# ================================================== genome-wide offset =======

def test_global_shift_report_spots_a_one_sided_contrast():
    """Every bin moving the same way is an offset between the arms, not
    thousands of local aberrations."""
    m = _mod()
    r = m.global_shift_report(np.full(1000, 1.0) + np.random.default_rng(0)
                              .normal(0, 0.05, 1000))
    assert r["frac_bins_same_direction"] > 0.99
    assert r["median_diff"] == pytest.approx(1.0, abs=0.02)
    assert r["median_over_mad"] > 10


def test_global_shift_report_is_quiet_on_a_local_contrast():
    m = _mod()
    rng = np.random.default_rng(0)
    d = rng.normal(0, 1, 1000)
    d[:50] += 4.0                                  # a few genuinely aberrant
    r = m.global_shift_report(d)
    assert r["frac_bins_same_direction"] < 0.7
    assert abs(r["median_diff"]) < 0.3


def test_global_shift_report_ignores_nan():
    m = _mod()
    r = m.global_shift_report(np.array([1.0, 1.0, np.nan, 1.0]))
    assert r["frac_bins_same_direction"] == pytest.approx(1.0)
    assert m.global_shift_report(np.array([np.nan, np.nan])) == {}


def test_center_cells_removes_a_per_cell_offset_not_the_shape():
    m = _mod()
    shape = np.array([0.0, 1.0, -1.0, 2.0])
    X = np.vstack([shape + 10.0, shape - 5.0, shape])
    got = m.center_cells(X)
    # every row now has the same values up to its own median
    for row in got:
        np.testing.assert_allclose(row - row[0], shape - shape[0])
    np.testing.assert_allclose(np.median(got, axis=1), 0.0, atol=1e-12)


def test_center_cells_tolerates_nan():
    m = _mod()
    X = np.array([[1.0, 2.0, np.nan], [11.0, 12.0, 13.0]])
    got = m.center_cells(X)
    assert np.isnan(got[0, 2])
    assert got[0, 0] == pytest.approx(1.0 - 1.5)


def test_a_global_offset_makes_everything_significant_and_says_so(tmp_path):
    """The failure this exists for: a per-cell offset in one arm makes every
    bin differ, the BED comes back holding the genome, and an enrichment on it
    is the genome against itself."""
    rng = np.random.default_rng(11)
    groups = [("tumor1", 30), ("tumor2", 30), ("mono", 28), ("lineA", 20)]
    cells, types = [], []
    for ct, k in groups:
        cells += [f"{ct}_{i}" for i in range(k)]
        types += [ct] * k
    types = np.array(types)
    n_bins = 400
    X = rng.normal(0, 1, (len(cells), n_bins))
    tum = np.char.startswith(types.astype(str), "tumor")
    X[tum] += 1.5                                   # the global offset
    X[np.ix_(tum, np.arange(30))] += 3.0            # 30 genuinely local bins

    bins = [f"chr1:{i * 250000}" for i in range(n_bins)]
    mat = tmp_path / "concordance_directional.parquet"
    pd.DataFrame(X, index=cells, columns=bins).to_parquet(mat)
    stab = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells, "bam_path": ["x"] * len(cells),
                  "cell_type": types,
                  "sample": ["b1", "b2"] * (len(cells) // 2)}).to_csv(
        stab, sep="\t", index=False)

    def run(sub, extra):
        out = tmp_path / sub
        r = subprocess.run(
            [sys.executable, str(_SCRIPT), "--matrix", f"gch={mat}",
             "--samples", str(stab), "--cell-type-column", "cell_type",
             "--batch-column", "sample", "--n-bins", "20",
             "--n-permutations", "20", "--contrast", "t=tumor1,tumor2:mono",
             "--output-dir", str(out), "--plot-format", "png"] + extra,
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        bed = pd.read_csv(out / "bin_scores_gch.contrast_t.bed", sep="\t")
        local = int(((bed["start"] // 250000) < 30).sum())
        return len(bed), local, r.stderr

    n_all, local_all, err = run("plain", [])
    assert n_all > 0.9 * n_bins, "the offset should light up nearly everything"
    assert "genome-wide OFFSET" in err
    assert "--contrast-center cell-median" in err

    n_ctr, local_ctr, err2 = run("centred", ["--contrast-center",
                                             "cell-median"])
    assert n_ctr < 0.2 * n_bins, f"{n_ctr} bins still selected after centring"
    assert local_ctr >= 25, f"only {local_ctr}/30 planted bins recovered"
    assert "genome-wide OFFSET" not in err2


# =========================================== precision-weighted contrast =====

def test_dl_tau2_recovers_the_between_cell_variance():
    """Each estimate carries its own sampling variance, so the spread between
    cells has a known part and an unknown one; tau^2 is the unknown part."""
    m = _mod()
    rng = np.random.default_rng(0)
    n, tau = 800, 0.5
    v = rng.uniform(0.01, 0.5, (n, 1))
    x = rng.normal(0, tau, (n, 1)) + rng.normal(0, np.sqrt(v))
    got = m._dl_tau2(x, v)[0]
    assert got == pytest.approx(tau ** 2, rel=0.25)


def test_dl_tau2_is_zero_when_the_spread_is_all_measurement_error():
    m = _mod()
    rng = np.random.default_rng(1)
    v = np.full((500, 1), 0.2)
    x = rng.normal(0, np.sqrt(0.2), (500, 1))
    assert m._dl_tau2(x, v)[0] == pytest.approx(0.0, abs=0.05)


def test_weighted_arm_is_inverse_variance():
    m = _mod()
    x = np.array([[1.0], [3.0]])
    v = np.array([[1.0], [3.0]])           # the second is 3x noisier
    mu, var = m._weighted_arm(x, v)
    assert mu[0] == pytest.approx((1 / 1 * 1 + 1 / 3 * 3) / (1 + 1 / 3))
    assert var[0] == pytest.approx(1 / (1 + 1 / 3))


def test_weighting_stops_thin_cells_dominating_a_contrast():
    """A handful of very noisy cells can carry an unweighted test; weighted by
    their own variance they cannot."""
    m = _mod()
    rng = np.random.default_rng(2)
    n_good, n_bad = 60, 12
    # Both arms have the same true value.  The 'in' arm has extra thin cells
    # whose estimates are wild.
    v_in = np.concatenate([np.full(n_good, 0.01), np.full(n_bad, 4.0)])[:, None]
    x_in = rng.normal(0.0, np.sqrt(v_in))
    v_out = np.full((n_good, 1), 0.01)
    x_out = rng.normal(0.0, np.sqrt(v_out))
    X = np.vstack([x_in, x_out])
    V = np.vstack([v_in, v_out])
    in_mask = np.array([True] * (n_good + n_bad) + [False] * n_good)
    out_mask = ~in_mask
    w = m.weighted_contrast(X, V, in_mask, out_mask, 3, "precision")
    assert abs(w["diff"][0]) < 0.05, "the thin cells should barely move it"


def test_random_effects_is_more_conservative_than_fixed():
    """Fixed effects assume every cell in an arm shares one true value; when
    they do not, it is anticonservative."""
    m = _mod()
    rng = np.random.default_rng(3)
    n = 80
    v = np.full((2 * n, 1), 0.01)
    x = np.vstack([rng.normal(0.4, 0.5, (n, 1)),      # real spread within arm
                   rng.normal(0.0, 0.5, (n, 1))])
    in_mask = np.array([True] * n + [False] * n)
    fixed = m.weighted_contrast(x, v, in_mask, ~in_mask, 3, "precision")
    re = m.weighted_contrast(x, v, in_mask, ~in_mask, 3, "random-effects")
    assert re["p"][0] > fixed["p"][0]


# ------------------------------------------------------- consistency --------

def test_consistency_keeps_only_bins_moving_the_same_way_in_every_group():
    """A bin significant on one tumour population alone is that population's
    idiosyncrasy, not a shared signature; pooling hides which."""
    m = _mod()
    per = {"t1": {"diff": np.array([1.0, 1.0, -1.0]),
                  "p": np.array([1e-9, 1e-9, 1e-9])},
           "t2": {"diff": np.array([1.0, -1.0, -1.0]),
                  "p": np.array([1e-9, 1e-9, 0.9])}}
    mask, info = m.consistency_mask(per, 0.05)
    assert list(mask) == [True, False, False]
    assert info["n_significant_in_any"] == 3
    assert info["n_consistent"] == 1


def test_consistency_needs_significance_in_all_groups_not_just_agreement():
    m = _mod()
    per = {"a": {"diff": np.array([1.0]), "p": np.array([1e-9])},
           "b": {"diff": np.array([1.0]), "p": np.array([0.5])}}
    mask, _ = m.consistency_mask(per, 0.05)
    assert list(mask) == [False]


def test_consistency_on_no_groups_is_none():
    m = _mod()
    mask, info = m.consistency_mask({}, 0.05)
    assert mask is None and info == {}


def test_coverage_report_flags_a_lopsided_contrast():
    m = _mod()
    counts = np.vstack([np.full((10, 5), 12.0), np.full((10, 5), 60.0)])
    in_mask = np.array([True] * 10 + [False] * 10)
    r = m.coverage_report(counts, in_mask, ~in_mask)
    assert r["median_pairs_in"] == 12 and r["median_pairs_out"] == 60
    assert r["pairs_ratio"] == pytest.approx(0.2)


# ------------------------------------------------------- stratification -----

def _strat_frame(n=60, bins=3, effect=(1.0, 1.0), patient_shift=3.0, seed=0):
    rng = np.random.default_rng(seed)
    ct = np.array(["in"] * n + ["out"] * n)
    pat = np.array([f"p{(i % 2) + 1}" for i in range(2 * n)])
    X = rng.normal(0, 0.4, (2 * n, bins))
    X += np.where((pat == "p1")[:, None], patient_shift, -patient_shift)
    for j, e in enumerate(effect):
        X[(ct == "in") & (pat == "p1"), j] += e
    for j, e in enumerate(effect):
        X[(ct == "in") & (pat == "p2"), j] += e
    return X, ct == "in", ct == "out", pat


def test_stratifying_removes_a_patient_main_effect():
    """Both arms carry the patient's whole contribution, so comparing inside
    a patient cancels it exactly -- including the parts no covariate models."""
    m = _mod()
    X, a, b, pat = _strat_frame(patient_shift=8.0, effect=(1.0, 1.0, 1.0))
    comb, per = m.stratified_contrast(X, pat, a, b, 5, "welch")
    assert comb["n_strata"] == 2
    # the huge patient shift does not reach the estimate
    np.testing.assert_allclose(comb["diff"], 1.0, atol=0.25)


def test_the_stratified_difference_is_the_mean_of_within_stratum_ones():
    m = _mod()
    X, a, b, pat = _strat_frame(effect=(2.0, 2.0, 2.0))
    comb, per = m.stratified_contrast(X, pat, a, b, 5, "welch")
    mean_of_strata = np.mean([per[k]["diff"] for k in per], axis=0)
    np.testing.assert_allclose(comb["diff"], mean_of_strata)


def test_a_stratum_missing_an_arm_is_skipped():
    m = _mod()
    X, a, b, pat = _strat_frame()
    pat = pat.copy()
    pat[a] = "p1"                       # p2 now has no 'in' cells at all
    comb, per = m.stratified_contrast(X, pat, a, b, 5, "welch")
    assert comb is None, "one usable stratum is not a stratified analysis"
    assert list(per) == ["p1"]


def test_stratified_combination_is_not_a_consistency_test():
    """Stouffer asks whether there is an average effect across strata, which a
    bin real in ONE stratum still passes.  Requiring every stratum is the
    different question, and the one 'specific' means."""
    m = _mod()
    rng = np.random.default_rng(2)
    n = 80
    ct = np.array(["in"] * n + ["out"] * n)
    pat = np.array([f"p{(i % 2) + 1}" for i in range(2 * n)])
    X = rng.normal(0, 0.4, (2 * n, 2))
    X[(ct == "in") & (pat == "p1"), 0] += 2.0        # bin 0: p1 only
    X[ct == "in", 1] += 2.0                          # bin 1: both
    comb, per = m.stratified_contrast(X, pat, ct == "in", ct == "out", 5,
                                      "welch")
    assert comb["p"][0] < 0.01, "the combined test passes the p1-only bin"
    mask, info = m.consistency_mask(per, 0.05)
    assert list(mask) == [False, True], "consistency rejects it"


# ----------------------------------------- variance approximated from n -----

def test_variance_from_counts_scales_as_one_over_n():
    """Delta is a difference of proportions, so its sampling variance falls as
    1/n; only the constant is unknown."""
    m = _mod()
    rng = np.random.default_rng(0)
    n_cells, n_bins = 200, 3
    counts = np.tile(np.array([10.0, 40.0, 160.0]), (n_cells, 1))
    X = rng.normal(0, np.sqrt(0.8 / counts))
    a = np.array([True] * 100 + [False] * 100)
    V = m.variance_from_counts(counts, X, a, ~a)
    ratios = np.nanmedian(V, axis=0)
    assert ratios[0] / ratios[1] == pytest.approx(4.0, rel=0.05)
    assert ratios[1] / ratios[2] == pytest.approx(4.0, rel=0.05)


def test_variance_from_counts_recovers_the_constant():
    m = _mod()
    rng = np.random.default_rng(1)
    counts = np.full((400, 1), 25.0)
    X = rng.normal(0, np.sqrt(0.5 / 25.0), (400, 1))
    a = np.array([True] * 200 + [False] * 200)
    V = m.variance_from_counts(counts, X, a, ~a)
    assert np.nanmedian(V) == pytest.approx(0.5 / 25.0, rel=0.2)


def test_variance_from_counts_absorbs_extra_spread_conservatively():
    """Between-cell variation the counts do not explain lands in the constant,
    which inflates the variances rather than understating them."""
    m = _mod()
    rng = np.random.default_rng(2)
    counts = np.full((400, 1), 25.0)
    tight = rng.normal(0, np.sqrt(0.5 / 25.0), (400, 1))
    loose = tight + rng.normal(0, 0.5, (400, 1))
    a = np.array([True] * 200 + [False] * 200)
    v_tight = np.nanmedian(m.variance_from_counts(counts, tight, a, ~a))
    v_loose = np.nanmedian(m.variance_from_counts(counts, loose, a, ~a))
    assert v_loose > 5 * v_tight


def test_counts_variance_is_estimated_inside_each_stratum():
    """Estimated over the whole arm it swallows the between-stratum
    difference -- the patient effect -- and then inflates every within-patient
    comparison by the very variation stratifying removed."""
    m = _mod()
    rng = np.random.default_rng(3)
    n = 60
    ct = np.array(["in"] * n + ["out"] * n)
    pat = np.array([f"p{(i % 2) + 1}" for i in range(2 * n)])
    counts = np.full((2 * n, 2), 30.0)
    X = rng.normal(0, np.sqrt(0.4 / 30.0), (2 * n, 2))
    X += np.where((pat == "p1")[:, None], 5.0, -5.0)     # huge patient effect
    X[ct == "in", 0] += 0.25                             # a real, small effect
    a, b = ct == "in", ct == "out"

    within, _ = m.stratified_contrast(X, pat, a, b, 5, "welch", counts=counts)
    # the same variance estimated once over the whole arm, for comparison
    V_pooled = m.variance_from_counts(counts, X, a, b)
    across, _ = m.stratified_contrast(X, pat, a, b, 5, "welch", V=V_pooled)
    assert within["p"][0] < across["p"][0], (
        "estimating the variance across strata should be the weaker of the "
        "two, since it carries the patient effect as noise")
    assert within["p"][0] < 0.01


# --------------------------------------------------------- bin dropout ------

def test_dropout_report_counts_testable_bins_per_stratum():
    """A weak contrast has two very different causes -- small effect, or most
    of the genome never tested -- and the hit count alone cannot separate
    them."""
    m = _mod()
    X = np.full((12, 4), 1.0)
    a = np.array([True] * 6 + [False] * 6)
    pat = np.array(["p1"] * 3 + ["p2"] * 3 + ["p1"] * 3 + ["p2"] * 3)
    X[:, 1] = np.nan                          # bin 1 unobserved everywhere
    X[3:6, 2] = np.nan                        # bin 2: p2 has no 'in' cells
    r = m.dropout_report(X, a, ~a, 3, pat)
    assert r["n_bins"] == 4
    assert r["n_testable_pooled"] == 3         # bins 0, 2, 3
    assert r["n_testable_p2"] == 2             # bin 2 fails for p2
    assert r["n_testable_all_strata"] == 2     # bins 0 and 3


def test_dropout_report_without_strata_reports_only_the_pooled_count():
    m = _mod()
    X = np.ones((10, 2))
    a = np.array([True] * 5 + [False] * 5)
    r = m.dropout_report(X, a, ~a, 3)
    assert r["n_testable_pooled"] == 2
    assert "n_testable_all_strata" not in r


def test_stratifying_can_only_lose_testable_bins_never_gain():
    m = _mod()
    rng = np.random.default_rng(0)
    X = np.where(rng.random((40, 200)) < 0.5, rng.normal(size=(40, 200)),
                 np.nan)
    a = np.array([True] * 20 + [False] * 20)
    pat = np.array(["p1", "p2"] * 20)
    r = m.dropout_report(X, a, ~a, 3, pat)
    assert r["n_testable_all_strata"] <= r["n_testable_pooled"]


def test_a_degenerate_variance_estimate_does_not_discard_the_bin():
    """One usable cell gives no residual to estimate the constant from.
    Returning NaN would drop the BIN, and in a sparse context that is most of
    the genome."""
    m = _mod()
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (6, 4))
    n = np.full((6, 4), 30.0)
    X[1:3, 1] = np.nan
    n[1:3, 1] = np.nan                    # the 'in' arm keeps one cell here
    a = np.array([True] * 3 + [False] * 3)
    V = m.variance_from_counts(n, X, a, ~a)
    assert np.isfinite(V[a]).any(axis=0).all(), "a bin lost its whole arm"
    # and the stand-in is the arm's usual noise, not something invented
    finite_elsewhere = V[a][:, [0, 2, 3]]
    assert (np.nanmin(finite_elsewhere) <= np.nanmedian(V[a][:, 1])
            <= np.nanmax(finite_elsewhere))


# ------------------------------------------------ companion alignment -------

def _companion_pair(tmp_path, cells, bins, comp_cells, comp_bins):
    mat = pd.DataFrame(np.zeros((len(cells), len(bins))),
                       index=cells, columns=bins)
    comp = pd.DataFrame(np.ones((len(comp_cells), len(comp_bins))),
                        index=comp_cells, columns=comp_bins)
    comp.to_parquet(tmp_path / "concordance_delta_var.parquet")
    return mat


def test_a_stale_companion_is_refused_not_silently_reindexed(tmp_path):
    """Reindexing a mismatched file succeeds and fills the gaps with NaN, so
    every cell it does not cover loses its variance and drops out -- leaving a
    handful of testable bins and nothing to say why."""
    m = _mod()
    cells = [f"c{i}" for i in range(100)]
    bins = [f"chr1:{i}" for i in range(50)]
    mat = _companion_pair(tmp_path, cells, bins, cells[:20], bins)
    with pytest.raises(SystemExit, match="does not match the matrix"):
        m._companion(str(tmp_path / "x.parquet"), None, "g",
                     "concordance_delta_var.parquet", "--delta-var",
                     matrix=mat)


def test_a_companion_missing_bins_is_refused_too(tmp_path):
    m = _mod()
    cells = [f"c{i}" for i in range(50)]
    bins = [f"chr1:{i}" for i in range(100)]
    mat = _companion_pair(tmp_path, cells, bins, cells, bins[:30])
    with pytest.raises(SystemExit, match="30% of its 100 bins|does not match"):
        m._companion(str(tmp_path / "x.parquet"), None, "g",
                     "concordance_delta_var.parquet", "--delta-var",
                     matrix=mat)


def test_a_companion_covering_the_matrix_is_accepted(tmp_path):
    """A superset is fine -- the matrix may have been filtered since."""
    m = _mod()
    cells = [f"c{i}" for i in range(50)]
    bins = [f"chr1:{i}" for i in range(30)]
    mat = _companion_pair(tmp_path, cells, bins,
                          cells + ["extra"], bins + ["chr9:1"])
    got = m._companion(str(tmp_path / "x.parquet"), None, "g",
                       "concordance_delta_var.parquet", "--delta-var",
                       matrix=mat)
    assert got is not None and got.shape[0] == 51


def test_volcano_separates_fdr_passing_bins_that_were_dropped(tmp_path):
    """A bin above the FDR line that a later filter removed must not be drawn
    as 'n.s.' -- a point above the threshold in the not-significant class
    reads as a broken plot, not as a second criterion."""
    m = _mod()
    diff = np.array([2.0, 1.5, 0.1])
    fdr = np.array([0.001, 0.001, 0.9])
    kept = np.array([True, False, False])         # bin 1 dropped by consistency
    passed = np.array([True, True, False])
    up, down = m.volcano_figure(diff, fdr, kept, tmp_path / "v.pdf", "t",
                                0.05, passed_fdr=passed)
    assert (up, down) == (1, 0)
    assert (tmp_path / "v.pdf").stat().st_size > 0


# ------------------------------------------------------- shared grid --------

def _mat(cells, bins, fill=1.0):
    return pd.DataFrame(np.full((len(cells), len(bins)), fill),
                        index=cells, columns=bins)


def test_shared_grid_keeps_only_cells_and_bins_present_in_both():
    """Position i must be the same cell and row j the same bin in both panels,
    or a reader comparing the two positions is comparing nothing."""
    m = _mod()
    a = _mat(["c1", "c2", "c3"], ["b1", "b2", "b3"])
    b = _mat(["c2", "c3", "c4"], ["b2", "b3", "b4"])
    (cells, cols), dropped = m.shared_grid({"a": a, "b": b},
                                           ["b1", "b2", "b3", "b4"])
    assert list(cells) == ["c2", "c3"]
    assert cols == ["b2", "b3"]
    assert dropped["bins_dropped"] == 2
    assert dropped["cells_dropped"] == 2          # c1 and c4


def test_shared_grid_drops_a_bin_measured_in_only_one_modality():
    """Present is not measured: an all-NaN column would draw as a blank
    stripe, which reads as 'no signal' when the truth is 'not measured'."""
    m = _mod()
    a = _mat(["c1", "c2"], ["b1", "b2"])
    b = _mat(["c1", "c2"], ["b1", "b2"])
    b.loc[:, "b2"] = np.nan
    (cells, cols), dropped = m.shared_grid({"a": a, "b": b}, ["b1", "b2"])
    assert cols == ["b1"]
    assert dropped["bins_dropped"] == 1


def test_shared_grid_drops_a_cell_measured_in_only_one_modality():
    m = _mod()
    a = _mat(["c1", "c2", "c3"], ["b1", "b2"])
    b = _mat(["c1", "c2", "c3"], ["b1", "b2"])
    b.loc["c2", :] = np.nan
    (cells, cols), dropped = m.shared_grid({"a": a, "b": b}, ["b1", "b2"])
    assert list(cells) == ["c1", "c3"]
    assert dropped["cells_dropped"] == 1


def test_shared_grid_returns_none_when_nothing_is_shared():
    m = _mod()
    a = _mat(["c1"], ["b1"])
    b = _mat(["c2"], ["b2"])
    grid, dropped = m.shared_grid({"a": a, "b": b}, ["b1", "b2"])
    assert grid is None
    assert dropped["bins_dropped"] == 2


def test_shared_grid_preserves_the_requested_bin_order():
    """The order comes from the clustering, and both panels must follow it."""
    m = _mod()
    a = _mat(["c1"], ["b1", "b2", "b3"])
    b = _mat(["c1"], ["b1", "b2", "b3"])
    (_, cols), _ = m.shared_grid({"a": a, "b": b}, ["b3", "b1", "b2"])
    assert cols == ["b3", "b1", "b2"]


def test_joint_figure_draws_both_panels_on_one_grid(tmp_path):
    m = _mod()
    cells = [f"c{i}" for i in range(12)]
    bins = [f"chr1:{i}" for i in range(6)]
    rng = np.random.default_rng(0)
    a = pd.DataFrame(rng.normal(size=(12, 6)), index=cells, columns=bins)
    b = pd.DataFrame(rng.normal(size=(9, 4)), index=cells[:9],
                     columns=bins[:4])
    tab = pd.DataFrame(
        {"a_diff": [1.0] * 4, "b_diff": [1.0] * 4,
         "a_fdr": [0.01] * 4, "b_fdr": [0.01] * 4,
         "a_sig": [True] * 4, "b_sig": [True] * 4,
         "category": ["both"] * 4}, index=bins[:4])
    ct = pd.Series(["x"] * 12, index=cells)
    m.joint_figure(tab, "a", "b", {"n_both": 4, "frac_both_same_direction": 1.0},
                   {"a": a, "b": b}, ct, tmp_path / "j.pdf", "lab")
    assert (tmp_path / "j.pdf").stat().st_size > 0


def test_joint_figure_uses_the_supplied_cell_type_colours(tmp_path):
    """Columns are cells in cell-type order, and that order is invisible
    without a key: which block is the tumour and which the reference is the
    first thing the figure has to answer."""
    m = _mod()
    cells = [f"c{i}" for i in range(12)]
    bins = [f"chr1:{i}" for i in range(6)]
    rng = np.random.default_rng(0)
    a = pd.DataFrame(rng.normal(size=(12, 6)), index=cells, columns=bins)
    b = pd.DataFrame(rng.normal(size=(12, 6)), index=cells, columns=bins)
    tab = pd.DataFrame(
        {"a_diff": [1.0] * 4, "b_diff": [1.0] * 4,
         "a_fdr": [0.01] * 4, "b_fdr": [0.01] * 4,
         "a_sig": [True] * 4, "b_sig": [True] * 4,
         "category": ["both"] * 4}, index=bins[:4])
    ct = pd.Series(["tumour"] * 6 + ["mono"] * 6, index=cells)
    out = tmp_path / "j.pdf"
    m.joint_figure(tab, "a", "b",
                   {"n_both": 4, "frac_both_same_direction": 1.0},
                   {"a": a, "b": b}, ct, out, "lab",
                   ct_colors={"tumour": "#D1495B", "mono": "#1F6F8B"})
    assert out.stat().st_size > 0


def test_joint_figure_invents_colours_when_none_are_given(tmp_path):
    """--color-column is optional; the key still has to exist."""
    m = _mod()
    cells = [f"c{i}" for i in range(8)]
    bins = [f"chr1:{i}" for i in range(4)]
    rng = np.random.default_rng(1)
    a = pd.DataFrame(rng.normal(size=(8, 4)), index=cells, columns=bins)
    tab = pd.DataFrame(
        {"a_diff": [1.0] * 2, "b_diff": [1.0] * 2,
         "a_fdr": [0.01] * 2, "b_fdr": [0.01] * 2,
         "a_sig": [True] * 2, "b_sig": [True] * 2,
         "category": ["both"] * 2}, index=bins[:2])
    ct = pd.Series(["x"] * 4 + ["y"] * 4, index=cells)
    out = tmp_path / "j.pdf"
    m.joint_figure(tab, "a", "b",
                   {"n_both": 2, "frac_both_same_direction": 1.0},
                   {"a": a, "b": a.copy()}, ct, out, "lab", ct_colors=None)
    assert out.stat().st_size > 0


# ------------------------------------------------- batch negative control --

def _neg_frame(n_cells=60, n_bins=40, batch_shift=0.0, bio_shift=0.0,
               seed=0):
    rng = np.random.default_rng(seed)
    ct = np.array(["A"] * (n_cells // 2) + ["B"] * (n_cells // 2))
    batch = np.array([f"b{i % 2}" for i in range(n_cells)])
    X = rng.normal(0, 1, (n_cells, n_bins))
    X += np.where((batch == "b0")[:, None], batch_shift, -batch_shift)
    X[ct == "A"] += bio_shift
    return X, ct, batch


def test_a_harmless_batch_yields_almost_no_bins():
    """The question a confounding statistic cannot answer: how many bins does
    this pipeline call when the arms differ ONLY by processing?"""
    m = _mod()
    X, ct, batch = _neg_frame(batch_shift=0.01, seed=1)
    got = m.batch_negative_control(X, ct, batch, 3, "welch", 0.05)
    assert got["n_significant"] <= 2, got


def test_a_strong_batch_lights_up_the_whole_genome():
    """And when it does, the biological bins are not separable from it --
    whatever Cramer's V says, and whatever the within-cell-type batch ICC
    filter reports."""
    m = _mod()
    X, ct, batch = _neg_frame(batch_shift=1.5, seed=2)
    got = m.batch_negative_control(X, ct, batch, 3, "welch", 0.05)
    assert got["n_significant"] >= 0.8 * got["n_tested"], got


def test_the_control_is_stratified_so_it_cannot_be_biology():
    """Arms taken from batch, compared WITHIN cell type: a difference it finds
    cannot be a cell-type difference."""
    m = _mod()
    X, ct, batch = _neg_frame(batch_shift=0.0, bio_shift=4.0, seed=3)
    got = m.batch_negative_control(X, ct, batch, 3, "welch", 0.05)
    assert got["n_significant"] <= 2, "a pure cell-type effect must not leak"
    assert got["n_strata"] == 2


def test_one_batch_level_cannot_make_a_control():
    m = _mod()
    X, ct, _ = _neg_frame()
    assert m.batch_negative_control(X, ct, np.array(["b0"] * len(ct)),
                                    3, "welch", 0.05) is None


def test_a_batch_level_with_too_few_cells_is_refused():
    m = _mod()
    X, ct, _ = _neg_frame(n_cells=20)
    batch = np.array(["b0"] * 18 + ["b1"] * 2)
    assert m.batch_negative_control(X, ct, batch, 5, "welch", 0.05) is None


def test_the_control_reports_the_levels_it_used():
    m = _mod()
    X, ct, _ = _neg_frame(n_cells=60)
    batch = np.array(["b0"] * 25 + ["b1"] * 25 + ["rare"] * 10)
    got = m.batch_negative_control(X, ct, batch, 3, "welch", 0.05)
    assert set(got["levels"]) == {"b0", "b1"}, "the two largest levels"
    assert got["n_cells"] == [25, 25]


def test_a_nested_batch_is_reported_as_unmeasurable_not_as_a_number():
    """Where no cell type spans both batches, comparing the batches compares
    the cell types.  Falling back to an unstratified test returns biology
    wearing a batch label -- the exact confusion the control exists to
    resolve."""
    m = _mod()
    rng = np.random.default_rng(7)
    ct = np.array(["A"] * 30 + ["B"] * 30)
    batch = np.where(ct == "A", "e1", "e2")          # perfectly nested
    X = rng.normal(0, 1, (60, 50))
    X[ct == "A"] += 3.0                              # a huge CELL TYPE effect
    got = m.batch_negative_control(X, ct, batch, 3, "welch", 0.05)
    assert got["n_significant"] is None, "must not report a count here"
    assert got["strata"] == []


def test_one_cell_type_spanning_both_batches_is_used_as_a_bridge():
    """A cell type split across two experiments measures the batch effect with
    the biology held constant."""
    m = _mod()
    rng = np.random.default_rng(8)
    ct = np.array(["tum"] * 20 + ["bridge"] * 20 + ["bridge"] * 20)
    batch = np.array(["e1"] * 20 + ["e1"] * 20 + ["e2"] * 20)
    X = rng.normal(0, 1, (60, 60))
    X[batch == "e2"] += 2.0                          # pure batch
    got = m.batch_negative_control(X, ct, batch, 3, "welch", 0.05)
    assert got["strata"] == ["bridge"]
    assert got["n_significant"] > 0.8 * got["n_tested"]


def test_the_bridge_does_not_pick_up_a_cell_type_effect():
    m = _mod()
    rng = np.random.default_rng(9)
    ct = np.array(["tum"] * 20 + ["bridge"] * 20 + ["bridge"] * 20)
    batch = np.array(["e1"] * 20 + ["e1"] * 20 + ["e2"] * 20)
    X = rng.normal(0, 1, (60, 60))
    X[ct == "tum"] += 4.0                            # cell type only, no batch
    got = m.batch_negative_control(X, ct, batch, 3, "welch", 0.05)
    assert got["strata"] == ["bridge"]
    assert got["n_significant"] <= 3, "a cell-type effect must not leak in"


# --------------------------------------------- within-cell permutation ----

def test_permutation_keeps_every_per_cell_property():
    """Offset, spread and coverage all survive; only WHICH bin a value
    belongs to is destroyed.  That is what makes the null contain a
    genome-wide shift."""
    m = _mod()
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (6, 40))
    X[0] += 5.0                       # a cell with its own offset
    X[1, :10] = np.nan                # and one with its own coverage
    got = m.permute_within_cells(X, np.random.default_rng(1))
    np.testing.assert_allclose(np.nanmean(got, axis=1), np.nanmean(X, axis=1))
    np.testing.assert_allclose(np.nanstd(got, axis=1), np.nanstd(X, axis=1))
    np.testing.assert_array_equal(np.isnan(got), np.isnan(X))
    for i in range(X.shape[0]):
        a = np.sort(X[i][np.isfinite(X[i])])
        b = np.sort(got[i][np.isfinite(got[i])])
        np.testing.assert_allclose(a, b)


def test_a_whole_cell_offset_alone_survives_no_permutation_test():
    """The failure this exists for: a genome-wide shift makes the parametric
    p call almost every bin, and there is no bin-specific difference at all."""
    m = _mod()
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (40, 300))
    a = np.arange(40) < 20
    X[a] += 1.0                       # offset only, nothing bin-specific
    got = m.permutation_null(X, a, ~a, 3, "welch", 60)
    q = m._bh_fdr(got["p_empirical"])
    assert int(np.sum(np.isfinite(q) & (q < 0.05))) == 0
    para = m._bh_fdr(m.contrast_stats(X, a, 3, ~a, "welch")["p"])
    assert int(np.sum(np.isfinite(para) & (para < 0.05))) > 200, (
        "the parametric p should call most of the genome here")


def test_a_real_bin_specific_effect_survives_permutation():
    m = _mod()
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (40, 300))
    a = np.arange(40) < 20
    X[a] += 1.0                       # the same nuisance offset
    X[np.ix_(a, np.arange(20))] += 3.0    # plus 20 genuinely different bins
    got = m.permutation_null(X, a, ~a, 3, "welch", 60)
    q = m._bh_fdr(got["p_empirical"])
    hit = np.isfinite(q) & (q < 0.05)
    assert hit[:20].sum() >= 18, "the planted bins should survive"
    assert hit[20:].sum() <= 5, "and little else should"


def test_the_empirical_p_never_reaches_zero():
    """N draws cannot evidence p = 0; the +1 keeps the floor at 1/(N+1)."""
    m = _mod()
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (20, 50))
    a = np.arange(20) < 10
    X[np.ix_(a, [0])] += 50.0         # absurdly extreme
    got = m.permutation_null(X, a, ~a, 3, "welch", 20)
    assert got["p_empirical"][0] == pytest.approx(1.0 / (got["n_null"] + 1))
    assert (np.nan_to_num(got["p_empirical"], nan=1.0) > 0).all()


def test_the_null_is_pooled_over_bins():
    """A per-bin null of 100 permutations cannot resolve a p below 0.01;
    pooling gives n_bins x n_perm draws because permuted bins are
    exchangeable."""
    m = _mod()
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (20, 40))
    a = np.arange(20) < 10
    got = m.permutation_null(X, a, ~a, 3, "welch", 10)
    assert got["n_null"] > 300, got["n_null"]


# ------------------------------------------------- coverage confounding ---

def test_the_effect_is_correlated_with_the_coverage_gap_when_depth_drives_it():
    """The failure a permutation null cannot catch: a coverage gap that
    varies BY BIN is bin-specific, so the shuffle keeps it, and shallower D
    is pulled toward the prior -- technical, and only ever one-directional."""
    m = _mod()
    rng = np.random.default_rng(4)
    n_cells, n_bins = 80, 600
    a = np.arange(n_cells) < 40
    gap = rng.uniform(0.2, 1.0, n_bins)
    depth = np.where(a[:, None], rng.poisson(60 * gap, (n_cells, n_bins)) + 1,
                     rng.poisson(60, (n_cells, n_bins)) + 1)
    X = 0.5 * (depth / (depth + 25.0)) + rng.normal(0, 0.2 / np.sqrt(depth))
    diff = m.contrast_stats(X, a, 3, ~a, "welch")["diff"]
    got = m.coverage_confound(depth, a, ~a, diff)
    assert got["pearson_r_effect_vs_coverage_gap"] > 0.8


def test_no_coverage_gap_gives_no_correlation():
    m = _mod()
    rng = np.random.default_rng(5)
    n_cells, n_bins = 80, 600
    a = np.arange(n_cells) < 40
    depth = np.full((n_cells, n_bins), 50.0)
    X = rng.normal(0.5, 0.1, (n_cells, n_bins))
    X[a] += 0.3                                  # a real effect, equal depth
    diff = m.contrast_stats(X, a, 3, ~a, "welch")["diff"]
    got = m.coverage_confound(depth, a, ~a, diff)
    # Matched bin for bin: there is no gap to correlate with, which is a
    # different statement from "the check ran and found nothing".
    assert got["gap_is_constant"] is True
    assert np.isnan(got["pearson_r_effect_vs_coverage_gap"])


def test_a_gap_that_does_not_track_the_effect_gives_a_low_correlation():
    m = _mod()
    rng = np.random.default_rng(6)
    n_cells, n_bins = 80, 600
    a = np.arange(n_cells) < 40
    # a real gap, but the effect is planted independently of it
    gap = rng.uniform(0.4, 1.0, n_bins)
    depth = np.where(a[:, None], rng.poisson(200 * gap, (n_cells, n_bins)) + 1,
                     rng.poisson(200, (n_cells, n_bins)) + 1)
    X = rng.normal(0.5, 0.05, (n_cells, n_bins))
    X[np.ix_(a, rng.permutation(n_bins)[:60])] += 0.5
    diff = m.contrast_stats(X, a, 3, ~a, "welch")["diff"]
    got = m.coverage_confound(depth, a, ~a, diff)
    assert abs(got["pearson_r_effect_vs_coverage_gap"]) < 0.3


def test_direction_skew_counts_the_sides():
    """Real regional biology goes both ways somewhere; a 99% one-sided result
    is usually a monotone technical gradient."""
    m = _mod()
    diff = np.array([-1.0, -2.0, -3.0, 1.0, 0.5])
    sig = np.array([True, True, True, True, False])
    got = m.direction_skew(diff, sig)
    assert (got["n_down"], got["n_up"]) == (3, 1)
    assert got["frac_one_direction"] == pytest.approx(0.75)


def test_direction_skew_on_nothing_significant():
    m = _mod()
    assert m.direction_skew(np.array([1.0, -1.0]),
                            np.array([False, False])) == {}


def test_coverage_confound_needs_enough_bins():
    m = _mod()
    depth = np.full((10, 4), 50.0)
    a = np.arange(10) < 5
    assert m.coverage_confound(depth, a, ~a, np.zeros(4)) == {}


# --------------------------------------------------------------------------
# Regressing depth out before the permutation null.
#
# The within-cell permutation absorbs any whole-cell effect -- a global offset,
# purity, uniform shallowness -- because shuffling a cell's values across bins
# leaves per-cell summaries untouched.  What it cannot absorb is a confound
# indexed BY BIN, and a bin-varying depth gap between the arms is exactly that.
# D is pulled toward its prior when thin, so a bin where one arm is thinner
# carries a real, reproducible, bin-specific difference that is not biology.
# These tests pin that the pooled depth curve removes it without eating the
# genuine effects sitting alongside it.
# --------------------------------------------------------------------------

def _depth_fixture(seed=0, n_cells=120, n_bins=1200, n_real=40, effect=0.30):
    """Two arms, arm A systematically thinner by a per-bin amount, plus a
    handful of bins where the biology is genuinely different."""
    rng = np.random.default_rng(seed)
    arm = np.zeros(n_cells, bool)
    arm[: n_cells // 2] = True                      # arm A: the thin one
    gap = rng.uniform(0.2, 1.0, n_bins)             # per-bin, so permutation-proof
    depth = rng.gamma(3.0, 20.0, (n_cells, n_bins))
    depth[arm] *= gap                               # A is thinner, bin by bin
    lam = depth / (depth + 25.0)                    # shrinkage toward the prior
    X = 0.5 * lam + rng.normal(0, 0.05, (n_cells, n_bins))
    real = np.zeros(n_bins, bool)
    real[rng.choice(n_bins, n_real, replace=False)] = True
    X[np.ix_(arm, real)] += effect                  # the only true signal
    return X, depth, arm, real


def _called(m, X, counts, arm, n_perm=60, regress=False):
    if regress:
        X, _ = m.regress_out_coverage(X, counts)
    cs = m.contrast_stats(X, arm, 2, out_mask=~arm, test="mwu")
    null = m.permutation_null(X, arm, ~arm, 2, "mwu", n_perm, seed=1)
    q = m._bh_fdr(np.asarray(null["p_empirical"], float))
    return np.asarray(q) < 0.1, np.asarray(cs["diff"])


def test_bin_varying_depth_gap_survives_the_permutation_null():
    """The motivating failure: identical biology in all but 40 bins, yet the
    raw contrast calls hundreds of bins, nearly all in one direction, and the
    permutation null passes them because the confound really is per-bin."""
    m = _mod()
    X, depth, arm, real = _depth_fixture()
    sig, diff = _called(m, X, depth, arm)
    false = int((sig & ~real).sum())
    assert false > 100, f"fixture did not reproduce the artefact ({false})"
    # and it is one-directional, the signature the user saw in the volcano
    down = int((sig & ~real & (diff < 0)).sum())
    assert down > 0.9 * false, "attenuation should push one way"


def test_regressing_depth_out_kills_the_artefact_and_keeps_the_biology():
    m = _mod()
    X, depth, arm, real = _depth_fixture()
    raw_sig, _ = _called(m, X, depth, arm)
    reg_sig, _ = _called(m, X, depth, arm, regress=True)
    raw_false = int((raw_sig & ~real).sum())
    reg_false = int((reg_sig & ~real).sum())
    assert reg_false < 0.05 * raw_false, (
        f"depth regression left {reg_false} of {raw_false} false positives")
    # the 40 planted bins must still be there -- this is the half that makes
    # the correction worth having rather than just conservative
    assert int((reg_sig & real).sum()) >= 0.9 * real.sum()


def test_regression_is_a_no_op_when_depth_does_not_vary():
    """Nothing to regress out means nothing removed: a flat-coverage matrix
    must come back as it went in, not shifted or flattened."""
    m = _mod()
    rng = np.random.default_rng(3)
    X = rng.normal(0.5, 0.1, (60, 400))
    counts = np.full((60, 400), 50.0)
    out, info = m.regress_out_coverage(X, counts)
    assert info["n_strata"] == 0
    np.testing.assert_allclose(out, X)


def test_regression_preserves_a_real_effect_at_equal_depth():
    """When both arms are sequenced the same, a genuine difference must pass
    through untouched -- the curve is fitted on depth, and depth says nothing
    about the arm here."""
    m = _mod()
    rng = np.random.default_rng(4)
    X = rng.normal(0.5, 0.05, (80, 300))
    depth = rng.gamma(3.0, 20.0, (80, 300))         # same law in both arms
    arm = np.zeros(80, bool)
    arm[:40] = True
    X[np.ix_(arm, np.arange(20))] += 0.25
    out, info = m.regress_out_coverage(X, depth)
    assert info["n_strata"] > 1
    before = np.nanmean(X[arm][:, :20]) - np.nanmean(X[~arm][:, :20])
    after = np.nanmean(out[arm][:, :20]) - np.nanmean(out[~arm][:, :20])
    assert after == pytest.approx(before, abs=0.02)


def test_regression_keeps_the_matrix_on_the_D_scale():
    """The grand mean is added back, so downstream --contrast-min-diff and the
    volcano axes still mean what they meant."""
    m = _mod()
    rng = np.random.default_rng(5)
    X = rng.normal(0.5, 0.1, (60, 500))
    depth = rng.gamma(3.0, 20.0, (60, 500))
    out, _ = m.regress_out_coverage(X, depth)
    assert np.nanmean(out) == pytest.approx(np.nanmean(X), abs=1e-9)
    assert np.isnan(out[~np.isfinite(X)]).all() if (~np.isfinite(X)).any() else True


def test_regression_leaves_missing_values_missing():
    m = _mod()
    rng = np.random.default_rng(6)
    X = rng.normal(0.5, 0.1, (40, 200))
    depth = rng.gamma(3.0, 20.0, (40, 200))
    X[rng.random(X.shape) < 0.2] = np.nan
    depth[rng.random(depth.shape) < 0.1] = 0.0      # uncovered
    out, _ = m.regress_out_coverage(X, depth)
    assert np.isnan(out[~np.isfinite(X)]).all()
    # a bin with no reads has no depth to regress against; it must survive
    # unchanged rather than being quietly dropped
    untouched = np.isfinite(X) & (depth <= 0)
    if untouched.any():
        np.testing.assert_allclose(out[untouched], X[untouched])


def test_regress_coverage_without_counts_refuses_with_a_usable_message(tmp_path):
    """Silently skipping the correction would be the worst outcome: the run
    looks corrected and is not.  It must name the flag that fixes it."""
    m = _mod()
    X, ct, batch, classes = _data(seed=1, n_per=30)
    cts = [f"CT{c}" for c in ct]
    samples = [f"cell{i}" for i in range(len(ct))]
    bins = [f"chr1:{i * 250000}" for i in range(X.shape[1])]
    mat = tmp_path / "gch.parquet"
    pd.DataFrame(X, index=samples, columns=bins).to_parquet(mat)
    stab = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": samples, "bam_path": ["x"] * len(ct),
                  "cell_type": cts,
                  "sample": ["batch2" if b == 0 else "scNH" for b in batch]}
                 ).to_csv(stab, sep="\t", index=False)
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix", f"gch={mat}",
         "--samples", str(stab), "--cell-type-column", "cell_type",
         "--batch-column", "sample", "--n-bins", "20", "--n-permutations", "20",
         "--output-dir", str(tmp_path / "out"), "--plot-format", "png",
         "--contrast", "a=CT0", "--contrast-regress-coverage"],
        capture_output=True, text=True)
    assert r.returncode != 0
    # the real guard, not argparse's usage text (which also lists the flag)
    assert "needs the read-pair" in r.stderr


def test_a_stratum_too_thin_to_centre_is_left_alone_not_shifted():
    """A stratum with too few observations to give a stable mean is skipped.
    It must come back untouched -- adding the grand mean to a value that was
    never centred would move it by ~the mean of D while leaving the depth
    effect in place, which is worse than doing nothing."""
    m = _mod()
    rng = np.random.default_rng(11)
    X = rng.normal(0.5, 0.05, (12, 15))
    # Most observations at one identical depth, so the quantile edges collapse
    # onto it and the upper edges come from a sparse tail.  That is what makes
    # a mix of one fat stratum and several thin ones: with a smooth depth
    # distribution, quantile edges guarantee equal occupancy and no stratum is
    # ever thin.
    depth = np.full((12, 15), 10.0)
    tail = np.unravel_index(rng.choice(X.size, 50, replace=False), X.shape)
    depth[tail] = np.linspace(100.0, 5000.0, 50)
    out, info = m.regress_out_coverage(X, depth)
    assert 0.0 < info["frac_corrected"] < 1.0
    untouched = np.isclose(out, X)
    assert untouched.any()
    # what was skipped is the sparse tail, and it came back byte-for-byte --
    # not shifted by the grand mean
    assert (depth[untouched] > 10.0).all()
    np.testing.assert_array_equal(out[untouched], X[untouched])


# --------------------------------------------------------------------------
# Dividing out assay reliability with a reference measured on chrM.
#
# D is a Pearson r between the posteriors at the two ends of a read pair, so
# under a noise model it is a RELIABILITY coefficient, D = S/(S+N) with
# S = e^2 var_A and N = Vbar/n_sites.  On the logit scale the GpC enzyme
# efficiency e is additive (2 log e), so subtracting the logit of a reference
# measured in the same cell cancels it.  chrM is the reference because it
# carries no nucleosomes -- unlike cLADs or replication-timing domains, there
# is no chromatin state there for a tumour to rearrange.
# --------------------------------------------------------------------------

def _corr_axis(p1, p2):
    a = p1 - p1.mean(-1, keepdims=True)
    b = p2 - p2.mean(-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(-1) / den, np.nan)


def _reliability_fixture(seed=0, n_cells=80, n_bins=400, n_real=25,
                         e_lo=0.55, e_hi=0.85, n_sites=6, n_pairs=40,
                         ref_pairs=2000):
    """Two arms whose enzyme differs, plus bins where the biology really does.

    D is BUILT the way the pipeline builds it -- a Pearson r across simulated
    read pairs -- rather than written down directly, so the attenuation under
    test arises from the model instead of being assumed.
    """
    rng = np.random.default_rng(seed)
    arm = np.zeros(n_cells, bool)
    arm[: n_cells // 2] = True
    e = np.where(arm, e_lo, e_hi)                  # arm A: the weaker enzyme
    sd_A = rng.uniform(0.05, 0.25, n_bins)         # the biology, per bin
    real = np.zeros(n_bins, bool)
    real[rng.choice(n_bins, n_real, replace=False)] = True
    sd = np.tile(sd_A, (n_cells, 1))
    sd[np.ix_(arm, real)] *= 2.0                   # a genuine effect in arm A
    A = np.clip(rng.normal(0.5, sd[:, :, None], (n_cells, n_bins, n_pairs)),
                0, 1)
    mu = np.clip(e[:, None, None] * A, 0, 1)
    X = _corr_axis(rng.binomial(n_sites, mu) / n_sites,
                   rng.binomial(n_sites, mu) / n_sites)
    # chrM: same model, var_A fixed (no chromatin to vary), many pairs
    Aref = np.clip(rng.normal(0.5, 0.15, (n_cells, ref_pairs)), 0, 1)
    mur = np.clip(e[:, None] * Aref, 0, 1)
    ref = _corr_axis(rng.binomial(n_sites, mur) / n_sites,
                     rng.binomial(n_sites, mur) / n_sites)
    return X, ref, arm, real


def _perm_called(m, X, arm, n_perm=100, seed=1):
    null = m.permutation_null(X, arm, ~arm, 2, "mwu", n_perm, seed=seed)
    q = m._bh_fdr(np.asarray(null["p_empirical"], float))
    cs = m.contrast_stats(X, arm, 2, out_mask=~arm, test="mwu")
    return (np.isfinite(q) & (q < 0.1)), np.asarray(cs["diff"], float)


def test_an_enzyme_efficiency_gap_survives_the_permutation_null():
    """The motivating failure.  A weaker enzyme attenuates D in EVERY bin,
    but not by the same amount -- the attenuation scales with each bin's own
    var_A -- so it is bin-specific and the permutation keeps it."""
    m = _mod()
    X, ref, arm, real = _reliability_fixture()
    sig, diff = _perm_called(m, X, arm)
    false = int((sig & ~real).sum())
    assert false > 50, f"fixture did not reproduce the artefact ({false})"
    assert int((sig & ~real & (diff < 0)).sum()) > 0.9 * false


def test_the_reference_removes_the_gap_and_recovers_the_real_bins():
    """Both halves matter.  Killing the false positives is easy -- a blunt
    filter does that.  The point of dividing out a MEASURED nuisance rather
    than permuting an unmeasured one away is that the real bins come back."""
    m = _mod()
    X, ref, arm, real = _reliability_fixture()
    raw_sig, _ = _perm_called(m, X, arm)
    Xc, info = m.correct_by_reference(X, ref)
    cor_sig, _ = _perm_called(m, Xc, arm)
    assert info["n_corrected"] == X.shape[0]
    assert int((cor_sig & ~real).sum()) < 0.1 * int((raw_sig & ~real).sum())
    # the power gain, not just the false-positive cut
    assert int((cor_sig & real).sum()) > 2 * int((raw_sig & real).sum())
    assert int((cor_sig & real).sum()) >= 0.8 * real.sum()


def test_a_constant_reference_changes_nothing_that_matters():
    """If every cell has the same reliability there is nothing to divide out,
    and the contrast must come back where it started."""
    m = _mod()
    rng = np.random.default_rng(3)
    X = np.clip(rng.normal(0.4, 0.1, (40, 200)), 0.01, 0.99)
    arm = np.zeros(40, bool)
    arm[:20] = True
    X[np.ix_(arm, np.arange(15))] += 0.2
    out, info = m.correct_by_reference(X, np.full(40, 0.3))
    assert info["n_corrected"] == 40
    before = np.nanmean(X[arm][:, :15]) - np.nanmean(X[~arm][:, :15])
    after = np.nanmean(out[arm][:, :15]) - np.nanmean(out[~arm][:, :15])
    assert after == pytest.approx(before, abs=0.02)


def test_cells_without_a_reference_are_returned_untouched():
    """Correcting some cells and not others leaves two scales in one matrix.
    The untouched ones must at least be EXACTLY untouched -- not round-tripped
    through logit/expit, where the eps clip would move them."""
    m = _mod()
    rng = np.random.default_rng(4)
    X = np.clip(rng.normal(0.4, 0.1, (30, 120)), -0.5, 0.99)
    ref = np.full(30, 0.3)
    ref[:5] = np.nan                       # too few pairs upstream
    ref[5:8] = -0.02                       # degenerate / negative
    out, info = m.correct_by_reference(X, ref)
    assert info["n_corrected"] == 22 and info["n_uncorrected"] == 8
    np.testing.assert_array_equal(out[:8], X[:8])
    assert not np.array_equal(out[8:], X[8:])


def test_missing_bins_stay_missing_through_the_correction():
    m = _mod()
    rng = np.random.default_rng(5)
    X = np.clip(rng.normal(0.4, 0.1, (30, 150)), 0.01, 0.99)
    X[rng.random(X.shape) < 0.25] = np.nan
    out, _ = m.correct_by_reference(X, np.full(30, 0.3))
    np.testing.assert_array_equal(np.isnan(out), np.isnan(X))


def test_a_negative_or_zero_D_is_floored_not_dropped():
    """Sampling noise puts some bins at or below 0, where the logit does not
    exist.  Turning those into NaN would change WHICH bins are testable
    between a corrected and an uncorrected run -- the one difference that
    would make the two incomparable."""
    m = _mod()
    rng = np.random.default_rng(6)
    # A Pearson r whose median sits well above zero with a real tail below
    # it -- the case the floor exists for.  A matrix that is HALF negative
    # is not a reliability coefficient and is refused outright; see
    # test_a_log_odds_matrix_is_refused_not_flattened.
    X = rng.normal(0.30, 0.20, (30, 200))         # ~7% below zero
    assert (X < 0).any()
    out, info = m.correct_by_reference(X, np.full(30, 0.3))
    assert np.isfinite(out).all()
    assert info["frac_at_floor"] > 0
    assert (out >= 0).all() and (out <= 1).all()


def test_a_mismatched_reference_length_is_an_error():
    m = _mod()
    with pytest.raises(ValueError, match="cells"):
        m.correct_by_reference(np.zeros((10, 5)), np.zeros(9))


def _ref_run(tmp_path, ref_table, extra=None):
    """Run the CLI with a reference table, returning the completed process."""
    m = _mod()
    X, ct, batch, classes = _data(seed=1, n_per=30)
    # On the RELIABILITY scale, because that is the only scale this
    # correction is defined on: it takes logit, which clips outside (0, 1).
    # _data is symmetric about zero, and feeding that through used to
    # collapse every bin in a cell onto one value while the run reported
    # success.  expit is monotone, so it puts the matrix in range without
    # changing which cells or which bins differ.
    X = 1.0 / (1.0 + np.exp(-(X - 1.0)))
    cts = [f"CT{c}" for c in ct]
    samples = [f"cell{i}" for i in range(len(ct))]
    bins = [f"chr1:{i * 250000}" for i in range(X.shape[1])]
    mat = tmp_path / "gch.parquet"
    pd.DataFrame(X, index=samples, columns=bins).to_parquet(mat)
    stab = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": samples, "bam_path": ["x"] * len(ct),
                  "cell_type": cts,
                  "sample": ["batch2" if b == 0 else "scNH" for b in batch]}
                 ).to_csv(stab, sep="\t", index=False)
    rpath = tmp_path / "ref.tsv"
    ref_table(samples).to_csv(rpath, sep="\t", index=False)
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix", f"gch={mat}",
         "--samples", str(stab), "--cell-type-column", "cell_type",
         "--batch-column", "sample", "--n-bins", "20", "--n-permutations", "20",
         "--output-dir", str(tmp_path / "out"), "--plot-format", "png",
         "--contrast", "a=CT0",
         "--contrast-reference-d", f"gch={rpath}"] + (extra or []),
        capture_output=True, text=True)


def test_cli_applies_the_reference_and_says_how_many_cells_it_covered(tmp_path):
    r = _ref_run(tmp_path, lambda s: pd.DataFrame(
        {"cell": s, "reference_d": np.linspace(0.2, 0.5, len(s))}))
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "divided out assay reliability" in r.stderr


def test_cli_warns_when_some_cells_have_no_reference(tmp_path):
    """Two scales in one matrix is a real hazard, not a footnote."""
    def table(s):
        d = np.linspace(0.2, 0.5, len(s))
        d[: len(s) // 4] = np.nan
        return pd.DataFrame({"cell": s, "reference_d": d})
    r = _ref_run(tmp_path, table)
    assert r.returncode == 0, r.stderr
    assert "no usable reference" in r.stderr and "two scales" in r.stderr


def test_cli_refuses_a_reference_covering_a_minority_of_cells(tmp_path):
    r = _ref_run(tmp_path, lambda s: pd.DataFrame(
        {"cell": list(s[: len(s) // 4]) + [f"ghost{i}" for i in range(5)],
         "reference_d": 0.3}))
    assert r.returncode != 0
    assert "regenerate it over the same cells" in r.stderr


def test_cli_refuses_a_table_without_the_expected_columns(tmp_path):
    r = _ref_run(tmp_path, lambda s: pd.DataFrame(
        {"cell_name": s, "D": 0.3}))
    assert r.returncode != 0
    assert "reference_concordance.py" in r.stderr


def test_cli_notes_the_overlap_with_cell_median_centering(tmp_path):
    """Both remove a per-cell effect; the run should say the second has
    nothing left to take rather than let the user assume they stack."""
    r = _ref_run(tmp_path, lambda s: pd.DataFrame(
        {"cell": s, "reference_d": np.linspace(0.2, 0.5, len(s))}),
        extra=["--contrast-center", "cell-median"])
    assert r.returncode == 0, r.stderr
    assert "nothing left to take" in r.stderr


def test_the_extractor_output_feeds_the_contrast_unmodified(tmp_path):
    """The two halves are separate programs joined by a TSV, so the column
    names are a real interface.  A rename on either side would pass both
    unit suites and fail only here."""
    ref_script = _SCRIPT.parent / "reference_concordance.py"
    X, ct, batch, classes = _data(seed=1, n_per=30)
    X = 1.0 / (1.0 + np.exp(-(X - 1.0)))     # the reliability scale; see _ref_run
    cts = [f"CT{c}" for c in ct]
    samples = [f"cell{i}" for i in range(len(ct))]
    bins = [f"chr1:{i * 250000}" for i in range(X.shape[1])]
    mat = tmp_path / "gch.parquet"
    pd.DataFrame(X, index=samples, columns=bins).to_parquet(mat)
    stab = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": samples, "bam_path": ["x"] * len(ct),
                  "cell_type": cts,
                  "sample": ["batch2" if b == 0 else "scNH" for b in batch]}
                 ).to_csv(stab, sep="\t", index=False)

    # posteriors for the SAME cells, with correlated ends on chrM
    pdir = tmp_path / "04_posteriors"
    pdir.mkdir()
    rng = np.random.default_rng(0)
    for cell in samples:
        n = 1500
        shared = rng.normal(0, 1, n)
        e1 = 0.6 * shared + 0.8 * rng.normal(0, 1, n)
        e2 = 0.6 * shared + 0.8 * rng.normal(0, 1, n)
        pd.DataFrame({
            "chrom": "chrM", "pos1": rng.integers(0, 16000, n),
            "pos2": rng.integers(0, 16000, n),
            "distance": rng.integers(1, 16000, n), "n1": 6, "n2": 6,
            "posterior1": 1 / (1 + np.exp(-e1)),
            "posterior2": 1 / (1 + np.exp(-e2)),
        }).to_parquet(pdir / f"{cell}.posteriors.parquet")

    ref_tsv = tmp_path / "ref.tsv"
    r1 = subprocess.run(
        [sys.executable, str(ref_script), "--posteriors", str(pdir),
         "--n-pairs", "800", "--output", str(ref_tsv)],
        capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr
    assert "with a usable reference" in r1.stderr

    r2 = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix", f"gch={mat}",
         "--samples", str(stab), "--cell-type-column", "cell_type",
         "--batch-column", "sample", "--n-bins", "20", "--n-permutations", "20",
         "--output-dir", str(tmp_path / "out"), "--plot-format", "png",
         "--contrast", "a=CT0", "--contrast-reference-d", f"gch={ref_tsv}"],
        capture_output=True, text=True)
    assert r2.returncode == 0, f"stdout:\n{r2.stdout}\nstderr:\n{r2.stderr}"
    assert "divided out assay reliability" in r2.stderr
    # every cell had a reference, so nothing was left on the other scale
    assert "no usable reference" not in r2.stderr


# ------------- structural zeros, and why removing the prior gap exposed them --

def test_a_bin_where_only_one_arm_has_coverage_calls_on_coverage():
    """An entry with no read pairs has k = 0 on both sides, so under a
    shared prior the statistic is EXACTLY 0 there.  A bin where one arm
    happens to be covered and the other is not then separates perfectly --
    the covered arm sits wherever its biology puts it, the uncovered arm
    sits on a pile of exact zeros -- and the test reports coverage.

    Under the old separate priors those entries carried the prior gap
    instead: uncovered got gap/alpha_0, covered got (gap + data)/(alpha_0 +
    n) with a LARGER denominator, so the ordering between the two arms was
    scrambled rather than consistent and the artefact stayed buried.
    Removing the gap did not create this; it uncovered it."""
    m = _mod()
    rng = np.random.default_rng(0)
    n, nb = 40, 200
    arm = np.zeros(n, bool)
    arm[:20] = True
    counts = np.full((n, nb), 8.0)
    X = rng.normal(0.0, 0.2, (n, nb))
    art = np.arange(120, 170)                 # only one arm is covered
    counts[np.ix_(arm, art)] = 0.0
    X[np.ix_(arm, art)] = 0.0                 # the exact-zero pile
    X[np.ix_(~arm, art)] = rng.normal(0.5, 0.2, (20, len(art)))

    raw = m.contrast_stats(X, arm, min_cells=5)
    called = np.asarray(raw["p"]) < 0.001
    assert called[art].mean() > 0.9           # essentially all of them

    masked = m.contrast_stats(np.where(counts >= 1, X, np.nan), arm,
                              min_cells=5)
    p = np.asarray(masked["p"])
    assert np.isfinite(p[art]).sum() == 0     # no arm left to compare


def test_masking_leaves_the_real_bins_alone():
    """The point of dropping non-observations is that the genuine calls
    survive; a filter that took those too would be a worse trade than the
    artefact."""
    m = _mod()
    rng = np.random.default_rng(1)
    n, nb = 40, 200
    arm = np.zeros(n, bool)
    arm[:20] = True
    counts = np.full((n, nb), 8.0)
    X = rng.normal(0.0, 0.2, (n, nb))
    real = np.arange(10, 30)
    X[np.ix_(arm, real)] += 1.2
    before = np.asarray(m.contrast_stats(X, arm, min_cells=5)["p"])
    after = np.asarray(m.contrast_stats(
        np.where(counts >= 1, X, np.nan), arm, min_cells=5)["p"])
    assert (before[real] < 0.001).mean() > 0.9
    assert (after[real] < 0.001).mean() > 0.9


def test_the_cli_refuses_min_pairs_without_the_counts(tmp_path):
    r = _ref_run(tmp_path, lambda s: pd.DataFrame(
        {"cell": s, "reference_d": np.linspace(0.2, 0.5, len(s))}),
        extra=["--min-pairs", "1", "--pair-counts", "gch=/nope.parquet"])
    assert r.returncode != 0


def _sparse_zero_matrix(seed=0, n=40, nb=300, n_dense=100):
    """Bin-structured coverage with NO signal at all.

    The first n_dense bins are covered in every cell; the rest in a tenth of
    them.  Uncovered entries are exactly 0, the way decompose_components
    emits them.  The two arms are drawn identically, so every call is a
    false positive by construction.
    """
    rng = np.random.default_rng(seed)
    cov = np.zeros((n, nb), bool)
    cov[:, :n_dense] = True
    cov[:, n_dense:] = rng.random((n, nb - n_dense)) < 0.10
    X = np.where(cov, rng.normal(0.0, 1.0, (n, nb)), 0.0)
    arm = np.zeros(n, bool)
    arm[: n // 2] = True
    return X, cov.astype(float), arm, np.arange(n_dense)


def _imputed_matrix(seed=0, n=40, nb=400, n_dense=100, tint=0.8):
    """A post-RWR matrix: every entry filled, none NaN.

    Measured entries are identical in both arms -- there is no signal here.
    The entries with no read pairs of their own are what RWR put there, and
    RWR borrows across a k-NN graph built from accessibility, so they carry
    the arm's signature by construction.
    """
    rng = np.random.default_rng(seed)
    arm = np.zeros(n, bool)
    arm[: n // 2] = True
    cov = np.zeros((n, nb), bool)
    cov[:, :n_dense] = True
    cov[:, n_dense:] = rng.random((n, nb - n_dense)) < 0.15
    sig = np.where(arm, tint, -tint)[:, None]
    X = np.where(cov, rng.normal(0.0, 1.0, (n, nb)),
                 sig + rng.normal(0.0, 0.05, (n, nb)))
    return X, cov, arm, np.arange(n_dense, nb)


def test_rwr_leaves_no_nan_so_the_permutation_has_nothing_to_hold():
    """permute_within_cells defends the null by keeping non-observations in
    place, and it recognises them by NaN.  A post-RWR component matrix has
    none: every (cell, bin) is filled whatever its real coverage."""
    m = _mod()
    X, _cov, _arm, _sparse = _imputed_matrix()
    assert np.isfinite(X).all()
    out = m.permute_within_cells(X, np.random.default_rng(0))
    assert not np.allclose(out, X)          # everything moved


def test_cell_type_aware_imputation_inflates_the_sparse_bins():
    """The filled values are not neutral.  A sparsely covered bin is mostly
    arm signature; the permuted column at that position holds each cell's
    genome-wide MIX of measured and imputed, which dilutes it -- so the
    observed clears a null that no longer describes it.

    Nothing in this fixture is real: the measured entries are drawn
    identically in both arms."""
    m = _mod()
    X, _cov, arm, sparse = _imputed_matrix()
    r = m.permutation_null(X, arm, ~arm, min_cells=5, test="mwu",
                           n_perm=300, seed=0)
    p = np.asarray(r["p_empirical"])[sparse]
    p = p[np.isfinite(p)]
    assert (p < 0.05).mean() > 0.4          # measured 0.577


def test_holding_the_imputed_entries_fixed_removes_it():
    """The imputed value is still evidence and stays in the observed.  What
    it must not do is MOVE: shuffled, a cell's imputed entries scatter their
    arm signature across every bin position and the null carries the
    genome-wide mix instead of the bin's own share.  Held in place, the
    signature sits in the null exactly as in the observed, and a bin clears
    only by differing more than the imputation explains."""
    m = _mod()
    X, cov, arm, sparse = _imputed_matrix()
    r = m.permutation_null(X, arm, ~arm, min_cells=5, test="mwu",
                           n_perm=300, seed=0, permutable=cov)
    p = np.asarray(r["p_empirical"])[sparse]
    ok = np.isfinite(p)
    assert ok.all()                          # every bin still testable
    assert (p[ok] < 0.05).mean() < 0.15      # measured 0.077, from 0.577


def test_holding_them_fixed_beats_dropping_them():
    """Dropping the imputed entries also works, and costs almost every
    sparse bin -- 6 of 300 still testable.  Holding them fixed keeps all of
    them, which is why --min-pairs gates the null rather than the matrix."""
    m = _mod()
    X, cov, arm, sparse = _imputed_matrix()
    held = m.permutation_null(X, arm, ~arm, min_cells=5, test="mwu",
                              n_perm=300, seed=0, permutable=cov)
    dropped = m.permutation_null(np.where(cov, X, np.nan), arm, ~arm,
                                 min_cells=5, test="mwu", n_perm=300, seed=0)
    n_held = int(np.isfinite(np.asarray(held["p_empirical"])[sparse]).sum())
    n_drop = int(np.isfinite(
        np.asarray(dropped["p_empirical"])[sparse]).sum())
    assert n_held > 10 * n_drop


def test_the_observed_statistic_is_untouched_by_the_gate():
    """--min-pairs must not change what is being tested, only what the null
    is built from.  A flag that quietly altered the estimate as well would
    make the two runs incomparable."""
    m = _mod()
    X, cov, arm, _sparse = _imputed_matrix()
    a = m.permutation_null(X, arm, ~arm, min_cells=5, test="mwu",
                           n_perm=20, seed=0)
    b = m.permutation_null(X, arm, ~arm, min_cells=5, test="mwu",
                           n_perm=20, seed=0, permutable=cov)
    np.testing.assert_allclose(a["stat"], b["stat"], equal_nan=True)


def _planted(n_real=5, seed=0, n=40, nb=400, n_dense=100):
    """Real signal in a few DENSE bins, over the imputed-tint background."""
    X, cov, arm, sparse = _imputed_matrix(seed=seed, n=n, nb=nb,
                                          n_dense=n_dense)
    real = np.arange(n_real)
    X[np.ix_(arm, real)] += 1.5
    return X, cov, arm, real, np.arange(n_real, n_dense), sparse


def test_one_pooled_null_destroys_power_when_coverage_varies():
    """Holding a sparse bin's imputed entries fixed makes its permuted
    statistic close to its observed one -- which is the point -- but it also
    makes it contribute enormous draws to a shared pool.  Measured: sparse
    bins at |t| median 9.3 against a dense-only null whose 99.9th percentile
    is 3.67.  Pooled together, a real effect at |t| 5.3 does not clear and
    power goes to zero everywhere, while the run still reports thousands of
    parametric calls."""
    m = _mod()
    X, cov, arm, real, _dn, _sp = _planted()
    r = m.permutation_null(X, arm, ~arm, min_cells=5, test="mwu",
                           n_perm=300, seed=0, permutable=cov)
    p = np.asarray(r["p_empirical"])
    assert (p[real] < 0.05).mean() == 0.0


def test_stratifying_the_pool_by_coverage_restores_it():
    m = _mod()
    X, cov, arm, real, dense_null, sparse = _planted()
    r = m.permutation_null(X, arm, ~arm, min_cells=5, test="mwu",
                           n_perm=300, seed=0, permutable=cov,
                           pool_strata=m.coverage_strata(cov))
    p = np.asarray(r["p_empirical"])
    assert (p[real] < 0.05).mean() == 1.0           # measured 1.00
    assert (p[dense_null] < 0.05).mean() < 0.08     # measured 0.032
    assert (p[sparse] < 0.05).mean() < 0.10         # measured 0.057
    assert r["n_pool_strata"] >= 2


def test_the_bands_have_to_separate_the_coverage_classes():
    """Quantile edges over a mostly sparse genome fall INSIDE the sparse
    class and put its better-covered bins in with the fully measured ones.
    The pool is poisoned again and power measured 0.00, which is why the
    bands are fixed."""
    m = _mod()
    X, cov, arm, real, _dn, _sp = _planted()
    frac = cov.mean(axis=0)
    leaky = np.clip(np.digitize(
        frac, np.unique(np.quantile(frac, np.linspace(0, 1, 7)))[1:-1]),
        0, 5)
    r = m.permutation_null(X, arm, ~arm, min_cells=5, test="mwu",
                           n_perm=300, seed=0, permutable=cov,
                           pool_strata=leaky)
    assert (np.asarray(r["p_empirical"])[real] < 0.05).mean() == 0.0


def test_a_thin_stratum_falls_back_to_the_whole_pool():
    """Conservative, rather than resolving a tail from a handful of draws."""
    m = _mod()
    X, cov, arm, _real, _dn, _sp = _planted()
    strat = np.zeros(X.shape[1], int)
    strat[:3] = 7                                   # 3 bins on their own
    r = m.permutation_null(X, arm, ~arm, min_cells=5, test="mwu",
                           n_perm=50, seed=0, permutable=cov,
                           pool_strata=strat, min_pool=5_000)
    assert np.isfinite(np.asarray(r["p_empirical"])[:3]).all()


def test_a_null_built_only_from_measured_bins_calls_the_sparse_ones():
    """Restricting the null to bins that were really measured is right about
    WHICH bins are trustworthy, and leaves the untrustworthy ones tested
    against a null that does not describe them.  Their statistic is mostly
    what RWR borrowed from a cell-type-aware graph, so it separates the arms
    by construction and clears a null built from measured bins.

    Measured across tint strengths, with nothing real in the sparse bins:
    0.037 at tint 0, 0.730 at 0.2, 1.000 at 0.8.  The design is correct only
    where the imputation carries no arm signature."""
    m = _mod()

    def measured_only_null(X, cov, arm, n_perm=300):
        g = np.random.default_rng(1)
        obs = np.abs(np.asarray(m.contrast_stats(X, arm, 5)["t"], float))
        D = np.vstack([
            np.abs(np.asarray(m.contrast_stats(
                m.permute_within_cells(X, g, cov), arm, 5)["t"], float))
            for _ in range(n_perm)])
        well = cov.mean(axis=0) >= 0.5
        pool = np.sort(D[:, well][np.isfinite(D[:, well])])
        rank = np.searchsorted(pool, obs, side="left")
        return (pool.size - rank + 1.0) / (pool.size + 1.0)

    rates = {}
    for tint in (0.0, 0.8):
        X, cov, arm, sparse = _imputed_matrix(tint=tint)
        p = measured_only_null(X, cov, arm)[sparse]
        rates[tint] = float((p[np.isfinite(p)] < 0.05).mean())
    assert rates[0.0] < 0.10          # measured 0.037
    assert rates[0.8] > 0.90          # measured 1.000


def test_marking_them_untestable_is_the_other_half_of_that_design():
    """Drop them from the null AND from the calls, rather than from one
    only.  Then nothing is reported that the imputation wrote."""
    m = _mod()
    X, cov, arm, sparse = _imputed_matrix(tint=0.8)
    frac = cov.mean(axis=0)
    Xc = X.copy()
    Xc[:, frac < 0.5] = np.nan
    r = m.permutation_null(Xc, arm, ~arm, min_cells=5, test="mwu",
                           n_perm=300, seed=0, permutable=cov)
    p = np.asarray(r["p_empirical"])
    assert not np.isfinite(p[sparse]).any()      # no call at all
    assert np.isfinite(p[:100]).all()            # the measured ones survive


def test_a_held_entry_does_not_move_and_the_rest_do():
    m = _mod()
    row = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    hold = np.array([[True, True, True, False, False, False]])
    out = m.permute_within_cells(row, np.random.default_rng(0), hold)
    np.testing.assert_array_equal(out[0, 3:], row[0, 3:])
    assert sorted(out[0, :3]) == [1.0, 2.0, 3.0]


def test_structural_zeros_alone_do_not_do_this():
    """The same argument does NOT go through for exact zeros: ties do not
    move a Mann-Whitney.  Recorded so the two are not conflated -- it is the
    cell-type tint that does the damage, not the fill."""
    m = _mod()
    X, cov, arm, sparse = _imputed_matrix(tint=0.0)
    X = np.where(cov, X, 0.0)
    r = m.permutation_null(X, arm, ~arm, min_cells=5, test="mwu",
                           n_perm=300, seed=0)
    p = np.asarray(r["p_empirical"])[sparse]
    p = p[np.isfinite(p)]
    assert (p < 0.05).mean() < 0.12


def test_the_permutation_holds_nan_in_place_but_cannot_hold_a_zero():
    """The exact property the sentinel defeats, pinned so the two stay
    linked: a change to either one has to face this."""
    m = _mod()
    rng = np.random.default_rng(0)
    row = np.array([[1.0, 2.0, np.nan, 3.0, 0.0, np.nan]])
    out = m.permute_within_cells(row, rng)
    assert np.isnan(out[0, 2]) and np.isnan(out[0, 5])
    assert sorted(out[0, [0, 1, 3, 4]]) == [0.0, 1.0, 2.0, 3.0]
    # the zero moved: it is finite, so it is treated as a measurement
    moved = any(m.permute_within_cells(row, np.random.default_rng(s))[0, 4]
                != 0.0 for s in range(8))
    assert moved


def test_the_expected_sign_reference_is_on_D_s_scale_only():
    """C*(2E[F]-1) is D's reference.  Against a log odds ratio it is not a
    rough approximation of the right thing, it is a different quantity that
    happens to be small enough to look harmless: L runs to order 0.1-1
    while the reference is order C, about 1e-3, so subtracting it moves L
    by about a percent and quietly answers a different question."""
    m = _mod()
    rng = np.random.default_rng(0)
    p_bar = rng.uniform(0.2, 0.6, 400)
    C = np.abs(rng.normal(1e-3, 3e-4, 400))
    ref = m.expected_directional(p_bar, C)
    assert m._robust_sd(ref) < 1e-2                  # D scale
    logor = rng.normal(0.0, 0.4, 400)                # a log odds ratio
    assert m._robust_sd(logor) > 20 * m._robust_sd(ref)


def test_a_scale_mismatched_sign_reference_is_refused(tmp_path):
    """Compared by robust scale, because the matrix arrives as a path and
    the script cannot know what statistic it holds.

    --sign-split on is needed to reach it: on 'auto' the sign machinery
    only runs when the filename contains 'directional', which
    concordance_direction_logor.parquet does not, so --sign-reference is
    inert there and was never the cause of anything."""
    m = _mod()
    rng = np.random.default_rng(1)
    n, nb = 30, 200
    cells = [f"cell{i}" for i in range(n)]
    bins = [f"chr1:{i * 250000}" for i in range(nb)]
    d = tmp_path / "ctx"
    d.mkdir()
    # a log-odds-ratio matrix, with D-scale companions beside it
    pd.DataFrame(rng.normal(0, 0.4, (n, nb)), index=cells,
                 columns=bins).to_parquet(d / "gch.parquet")
    pd.DataFrame(np.tile(rng.uniform(0.2, 0.6, nb), (n, 1)), index=cells,
                 columns=bins).to_parquet(d / "concordance_marginal.parquet")
    pd.DataFrame(np.tile(np.abs(rng.normal(1e-3, 3e-4, nb)), (n, 1)),
                 index=cells, columns=bins).to_parquet(
        d / "concordance_coordination.parquet")
    stab = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells, "bam_path": ["x"] * n,
                  "cell_type": [f"CT{i % 3}" for i in range(n)],
                  "batch": ["b0" if i % 2 else "b1" for i in range(n)]}
                 ).to_csv(stab, sep="\t", index=False)
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix", f"gch={d / 'gch.parquet'}",
         "--samples", str(stab), "--cell-type-column", "cell_type",
         "--n-bins", "20", "--sign-reference", "expected",
         "--sign-split", "on", "--batch-column", "batch",
         "--output-dir", str(tmp_path / "out"), "--plot-format", "png"],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "on D's scale" in r.stderr
    assert "--sign-reference zero" in r.stderr


def test_a_threshold_can_be_global_or_named_per_matrix():
    """GCH and HCG do not want the same cut: one is dense enough that a
    coverage filter barely bites, the other is mostly imputed.  A single
    scalar forces the looser of the two on both, and the looser one is the
    one that lets imputation-driven bins through."""
    m = _mod()
    f = lambda specs, name: m.per_matrix(specs, name, int, 0, "--min-pairs")
    assert f(["1"], "gch") == 1 and f(["1"], "hcg") == 1
    assert f(["1", "hcg=5"], "gch") == 1
    assert f(["1", "hcg=5"], "hcg") == 5
    assert f(["gch=0", "hcg=3"], "gch") == 0
    assert f(None, "anything") == 0
    # a later NAME= wins over an earlier one
    assert f(["hcg=2", "hcg=7"], "hcg") == 7


def test_a_float_threshold_casts_and_a_bad_one_is_an_error():
    m = _mod()
    assert m.per_matrix(["0.5", "hcg=0.01"], "hcg", float, 0.0,
                        "--min-bin-frac") == 0.01
    with pytest.raises(SystemExit):
        m.per_matrix(["half"], "gch", float, 0.0, "--min-bin-frac")


def test_pair_counts_are_found_beside_a_matrix_that_was_not_named(tmp_path):
    """--pair-counts hcg=... does NOT exempt gch: _companion falls back to
    the file sitting beside each matrix, and step 5 writes one into every
    context directory."""
    m = _mod()
    d = tmp_path / "ctx"
    d.mkdir()
    cells = [f"c{i}" for i in range(4)]
    bins = [f"chr1:{i * 250000}" for i in range(6)]
    mat = pd.DataFrame(np.zeros((4, 6)), index=cells, columns=bins)
    mat.to_parquet(d / "gch.parquet")
    pd.DataFrame(np.ones((4, 6), int), index=cells,
                 columns=bins).to_parquet(d / "pair_counts_per_bin.parquet")
    got = m._companion(str(d / "gch.parquet"), ["hcg=/elsewhere.parquet"],
                       "gch", "pair_counts_per_bin.parquet", "--pair-counts")
    assert got is not None and got.shape == (4, 6)


# ---------------------------------------- the joint scatter across contexts --

def _joint_tab(n=800, n_both=60, slope=0.7, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1.0, n)
    y = slope * x + rng.normal(0, 0.6, n)
    cat = np.array(["neither"] * n, dtype=object)
    strong = np.argsort(-(np.abs(x) + np.abs(y)))
    cat[strong[:n_both]] = "both"
    cat[strong[n_both:n_both + 40]] = "gch_only"
    cat[strong[n_both + 40:n_both + 80]] = "hcg_only"
    return pd.DataFrame({"gch_diff": x, "hcg_diff": y, "category": cat},
                        index=[f"chr1:{i * 250000}" for i in range(n)])


def _ax():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt.subplots(figsize=(5, 5))


def test_the_joint_scatter_is_two_colours_and_red_is_both():
    """The question the panel answers is whether a bin is aberrant in BOTH
    modalities.  A single-modality bin belongs with the background it is
    being distinguished from, not with a category competing for the eye."""
    m = _mod()
    tab = _joint_tab()
    fig, ax = _ax()
    m.joint_scatter(ax, tab, "gch", "hcg",
                    {"n_both": 60, "frac_both_same_direction": 0.9}, "lab")
    import matplotlib.colors as mc
    cols = [tuple(np.round(c.get_facecolor()[0][:3], 3))
            for c in ax.collections]
    assert len(cols) == 2
    assert tuple(np.round(mc.to_rgb("#D7263D"), 3)) in cols
    assert tuple(np.round(mc.to_rgb("#BFC4C9"), 3)) in cols
    # the red layer holds exactly the bins called in both
    red = [c for c in ax.collections
           if tuple(np.round(c.get_facecolor()[0][:3], 3))
           == tuple(np.round(mc.to_rgb("#D7263D"), 3))][0]
    assert red.get_offsets().shape[0] == 60


def test_the_fit_is_over_every_bin_not_the_significant_ones():
    """A line through the significant subset would be fitted to the points
    selected for being extreme, and its slope would measure that
    selection."""
    m = _mod()
    tab = _joint_tab(slope=0.7)
    fig, ax = _ax()
    fit = m.joint_scatter(ax, tab, "gch", "hcg",
                          {"n_both": 60, "frac_both_same_direction": 0.9},
                          "lab")
    assert fit["n_fit"] == len(tab)
    assert fit["slope"] == pytest.approx(0.7, abs=0.1)
    both = tab[tab["category"] == "both"]
    only_both = np.polyfit(both["gch_diff"], both["hcg_diff"], 1)[0]
    assert abs(only_both - 0.7) > abs(fit["slope"] - 0.7)


def test_the_fit_reports_the_correlation_beside_the_slope():
    """The slope is attenuated -- both axes are estimated with error, and
    error on x biases a least-squares slope toward zero -- so it is
    descriptive and the association claim rests on the correlation."""
    m = _mod()
    fig, ax = _ax()
    fit = m.joint_scatter(ax, _joint_tab(), "gch", "hcg",
                          {"n_both": 60, "frac_both_same_direction": 0.9},
                          "lab")
    assert {"slope", "pearson_r", "spearman_rho", "n_fit"} <= set(fit)
    assert fit["pearson_r"] > 0.5
    txt = " ".join(t.get_text() for t in ax.texts)
    assert "attenuated" in txt and "ρ" in txt


def test_the_axis_names_the_statistic_the_matrix_held():
    m = _mod()
    assert m.statistic_label("x/concordance_coordination.parquet") == "C"
    assert m.statistic_label("x/concordance_direction_logor.parquet") == \
        "log-OR"
    assert "D = C" in m.statistic_label("x/concordance_directional.parquet")
    assert m.statistic_label("x/anything.parquet") is None
    fig, ax = _ax()
    m.joint_scatter(ax, _joint_tab(), "gch", "hcg",
                    {"n_both": 0, "frac_both_same_direction": np.nan}, "lab",
                    stat_labels={"gch": "C", "hcg": "log-OR"})
    assert "(C)" in ax.get_xlabel() and "(log-OR)" in ax.get_ylabel()


def test_a_scatter_with_nothing_in_both_still_draws(tmp_path):
    m = _mod()
    tab = _joint_tab(n_both=0)
    fig, ax = _ax()
    fit = m.joint_scatter(ax, tab, "gch", "hcg",
                          {"n_both": 0, "frac_both_same_direction": np.nan},
                          "lab")
    assert fit["n_fit"] == len(tab)
    assert "no bin significant in both" in ax.get_title()


def test_min_cells_per_group_is_per_matrix_too():
    """GCH keeps most bins at any setting while HCG is mostly imputed, so a
    single scalar forces the looser of the two on both -- and the looser one
    is what lets imputation-driven bins through."""
    m = _mod()
    f = lambda sp, n: m.per_matrix(sp, n, int, 3, "--min-cells-per-group")
    assert f(None, "gch") == 3                    # the documented default
    assert f(["20"], "gch") == 20 and f(["20"], "hcg") == 20
    assert f(["hcg=20"], "gch") == 3              # untouched
    assert f(["hcg=20"], "hcg") == 20
    assert f(["5", "hcg=20"], "gch") == 5


def test_a_per_matrix_setting_does_not_leak_into_the_next(tmp_path):
    """It is resolved by shadowing `args` for one iteration, so the rebuild
    from the untouched base has to actually happen -- otherwise the second
    matrix silently inherits the first's cut."""
    src = _SCRIPT.read_text()
    i = src.index("for mi, spec in enumerate(base_args.matrix):")
    block = src[i:i + 1400]
    assert "args = argparse.Namespace(**vars(base_args))" in block
    # and the loop reads the ORIGINAL list, not the shadowed scalar
    assert "enumerate(base_args.matrix)" in src
    assert "per_matrix(\n            base_args.min_cells_per_group" in block
    # restored afterwards, since everything past the loop is cross-matrix
    j = src.index("if not summary:\n        sys.exit(\"no matrix produced")
    assert "args = base_args" in src[j - 400:j]


def test_two_matrices_get_their_own_cut(tmp_path):
    """End to end: a dense matrix with the default and a sparse one raised,
    in one run."""
    m = _mod()
    X, ct, batch, classes = _data(seed=1, n_per=30)
    X = 1.0 / (1.0 + np.exp(-(X - 1.0)))
    samples = [f"cell{i}" for i in range(len(ct))]
    bins = [f"chr1:{i * 250000}" for i in range(X.shape[1])]
    d = tmp_path / "mats"
    d.mkdir()
    for nm in ("gch", "hcg"):
        pd.DataFrame(X, index=samples, columns=bins).to_parquet(
            d / f"{nm}.parquet")
    stab = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": samples, "bam_path": ["x"] * len(ct),
                  "cell_type": [f"CT{c}" for c in ct],
                  "sample": ["b0" if b == 0 else "b1" for b in batch]}
                 ).to_csv(stab, sep="\t", index=False)
    def run(cut, out):
        return subprocess.run(
            [sys.executable, str(_SCRIPT),
             "--matrix", f"gch={d / 'gch.parquet'}",
             "--matrix", f"hcg={d / 'hcg.parquet'}",
             "--samples", str(stab), "--cell-type-column", "cell_type",
             "--batch-column", "sample", "--n-bins", "20",
             "--min-cells-per-group", cut,
             "--output-dir", str(tmp_path / out), "--plot-format", "png"],
            capture_output=True, text=True)

    # 120 cells: a cut of 25 needs 2*25*3 = 150 and skips the matrix, a cut
    # of 3 needs 18 and does not.
    named = run("hcg=25", "named")
    assert named.returncode == 0, named.stderr
    assert "hcg: only 120 matched cells" in named.stderr
    assert "gch: only 120 matched cells" not in named.stderr

    # the same number as a bare scalar reaches BOTH, which is what the
    # per-matrix form exists to avoid
    both = run("25", "both")
    assert "hcg: only 120 matched cells" in both.stderr
    assert "gch: only 120 matched cells" in both.stderr


# ---------------------------------------------------------------------------
# --null-rank: does either arm depart from its own null?
# ---------------------------------------------------------------------------


def _rank_fixture(tmp_path, seed=0, n=60):
    """Three blocks of bins, and only the first is a real finding.

      0-19   arms differ AND both depart strongly from their own null
      20-39  arms differ by a small constant, NEITHER departs from its null
      40-59  nothing

    Block 20-39 is the failure this flag exists for: at tens of cells per arm
    a constant offset clears FDR comfortably, and the contrast has no way to
    ask whether either arm was doing anything at all.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    nb = 60
    ct = np.repeat(np.arange(4), n // 4)
    tum = ct < 2
    X = np.zeros((n, nb))
    X[:, :20] = np.where(tum[:, None], 1.5, -1.5) + rng.normal(0, .3, (n, 20))
    X[:, 20:40] = (np.where(tum[:, None], 0.10, -0.10)
                   + rng.normal(0, .05, (n, 20)))
    X[:, 40:] = rng.normal(0, .05, (n, 20))
    R = np.zeros((n, nb))
    R[:, :20] = (np.abs(rng.normal(3.5, .5, (n, 20)))
                 * np.where(tum[:, None], 1.0, -1.0))
    R[:, 20:] = rng.normal(0, .4, (n, 40))
    cells = [f"c{i}" for i in range(n)]
    bins = [f"chr1:{i * 250000}" for i in range(nb)]
    mat = tmp_path / "concordance_coordination.parquet"
    pd.DataFrame(X, index=cells, columns=bins).to_parquet(mat)
    pd.DataFrame(R, index=cells, columns=bins).to_parquet(
        tmp_path / "concordance_coordination_rank.parquet")
    s = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells, "bam_path": ["x"] * n,
                  "cell_type": [f"CT{c}" for c in ct],
                  "sample": ["b1", "b2"] * (n // 2)}).to_csv(
        s, sep="\t", index=False)
    return mat, s


def _run_rank(tmp_path, mat, s, tag, *extra):
    out = tmp_path / f"out_{tag}"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix", f"gch={mat}",
         "--samples", str(s), "--cell-type-column", "cell_type",
         "--batch-column", "sample", "--n-bins", "20",
         "--n-permutations", "30", "--contrast", "tumour=CT0,CT1",
         "--volcano", "--output-dir", str(out), "--plot-format", "png",
         *extra], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2500:]
    return out, r.stdout + r.stderr


def test_the_rank_matrix_is_found_beside_the_matrix_it_belongs_to(tmp_path):
    mod = _mod()
    assert mod.rank_filename_for("/x/concordance_direction_logor.parquet") == \
        "concordance_direction_logor_rank.parquet"
    assert mod.rank_filename_for("/x/concordance_coordination.parquet") == \
        "concordance_coordination_rank.parquet"
    # Pairing a coordination matrix with a direction rank would gate one
    # statistic on another's evidence, so the name is derived, not fixed.
    assert mod.rank_filename_for("/x/concordance_coordination.parquet") != \
        mod.rank_filename_for("/x/concordance_direction_logor.parquet")


def test_departure_is_per_arm_and_ignores_unmeasured_cells(tmp_path):
    """An unmeasured cell is not evidence of absence.  Counting it as
    non-departing would penalise exactly the sparse bins that most need an
    honest answer."""
    mod = _mod()
    R = np.array([[3.0, 0.1, np.nan],
                  [2.0, 0.2, 4.0],
                  [0.5, 0.0, 5.0],
                  [np.nan, 0.1, 0.3]])
    a = np.array([True, True, False, False])
    d = mod.rank_departure(R, a, ~a, 1.3)
    np.testing.assert_allclose(d["rank_frac_a"], [1.0, 0.0, 1.0])
    np.testing.assert_allclose(d["rank_frac_b"], [0.0, 0.0, 0.5])
    np.testing.assert_array_equal(d["rank_n_a"], [2, 2, 1])
    # max over arms, not min: a bin coordinated in the reference and flat in
    # the tumour is the finding, not a failure of it.
    np.testing.assert_allclose(d["rank_frac_max"], [1.0, 0.0, 1.0])


def test_the_gate_drops_bins_where_neither_arm_departs_from_null(tmp_path):
    """Without it the run calls twice as many bins, and half of them are a
    difference in nothing."""
    mat, s = _rank_fixture(tmp_path)
    _, log_off = _run_rank(tmp_path, mat, s, "off", "--null-rank-min-frac", "0")
    _, log_on = _run_rank(tmp_path, mat, s, "on", "--null-rank-min-frac", "0.5")

    def called(log):
        line = [l for l in log.splitlines() if "up /" in l and "down ->" in l]
        assert line, log
        return int(line[0].split()[2].replace(",", ""))

    assert called(log_off) == 40
    assert called(log_on) == 20
    assert "no arm departs from its own drawn null" in log_on
    assert "no arm departs" not in log_off


def test_the_dropped_bins_stay_visible_rather_than_being_hidden(tmp_path):
    """Applied to `sig`, not by NaN-ing the column, so a bin that cleared FDR
    and failed the gate shows on the volcano as 'FDR only, not kept'.  How
    many calls rested on nothing is worth seeing."""
    mat, s = _rank_fixture(tmp_path)
    out, log = _run_rank(tmp_path, mat, s, "vis", "--null-rank-min-frac", "0.5")
    assert "significant but filtered" in log
    tested = pd.read_csv(out / "bin_scores_gch.contrast_tumour.tested.bed",
                         sep="\t")
    assert len(tested) == 60          # still tested, just not called


def test_the_departure_rates_are_reported_even_when_nothing_is_gated(
        tmp_path):
    """'The arms differ' and 'either arm is doing something' are separate
    claims; a table carrying only the first invites the second to be
    assumed."""
    mat, s = _rank_fixture(tmp_path)
    out, log = _run_rank(tmp_path, mat, s, "rep", "--null-rank-min-frac", "0")
    t = pd.read_csv(out / "bin_scores_gch.tsv", sep="\t")
    for c in ("contrast_tumour_rank_frac_a", "contrast_tumour_rank_frac_b",
              "contrast_tumour_rank_frac_max", "contrast_tumour_rank_mean_a",
              "contrast_tumour_rank_mean_b"):
        assert c in t.columns, c
    idx = t["bin"].str.split(":").str[1].astype(int) // 250_000
    real = t.loc[idx < 20, "contrast_tumour_rank_frac_max"]
    hollow = t.loc[(idx >= 20) & (idx < 40), "contrast_tumour_rank_frac_max"]
    assert real.mean() > 0.9
    assert hollow.mean() < 0.2
    assert "departs from its own drawn null" in log


def test_the_rank_companion_heatmap_uses_the_concordance_layout(tmp_path):
    """It answers what the main heatmap cannot: is the block structure a
    departure from the null, or the raw statistic's scale tracking depth and
    openness across cell types."""
    mat, s = _rank_fixture(tmp_path)
    out, log = _run_rank(tmp_path, mat, s, "hm")
    made = {p.name for p in out.glob("*nullrank*")}
    assert "marker_heatmap_gch_nullrank_fixedorder.png" in made
    assert "marker_heatmap_gch_nullrank_reclustered.png" in made
    assert "null-rank companions" in log


def test_the_rank_heatmap_is_never_z_scored(tmp_path):
    """The rank already has a calibrated zero -- 'at its own null'.  Z-scoring
    each bin across cells would discard it and manufacture contrast in a bin
    where every cell sits at the null, which is the one thing this panel
    exists to rule out."""
    import inspect
    src = inspect.getsource(_mod().main)
    i = src.index('"nullrank"')
    call = src[i:i + 260]
    assert "symmetric=True" in call
    assert "zscore=True" not in call


def test_asking_for_a_rank_that_is_not_there_is_an_error_not_a_skip(tmp_path):
    mat, s = _rank_fixture(tmp_path)
    (tmp_path / "concordance_coordination_rank.parquet").unlink()
    out = tmp_path / "out_missing"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--matrix", f"gch={mat}",
         "--samples", str(s), "--cell-type-column", "cell_type",
         "--batch-column", "sample", "--n-bins", "10",
         "--n-permutations", "20", "--contrast", "tumour=CT0,CT1",
         "--null-rank-heatmaps", "on",
         "--output-dir", str(out), "--plot-format", "png"],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "--null-rank" in r.stderr


def test_the_rank_is_not_put_through_the_per_cell_corrections(tmp_path):
    """It is already referenced to each cell's own null, which is what
    --contrast-center cell-median approximates; centring it again would
    subtract an offset that is not there."""
    import inspect
    src = inspect.getsource(_mod().main)
    nrc = src.index("NRc = NR_df.reindex")
    centre = src.index("Xc = center_cells(Xc)")
    assert nrc < centre
    # center_cells / correct_by_reference / regress_out_coverage all act on
    # Xc; none of them may touch NRc.
    for fn in ("center_cells(", "correct_by_reference(",
               "regress_out_coverage("):
        for m in re.finditer(re.escape(fn), src):
            seg = src[max(0, m.start() - 120):m.start()]
            assert "NRc" not in seg.split("\n")[-1], fn
