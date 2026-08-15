"""Cell-type marker-bin selection + the out-of-selection control.

The statistical trap this figure must avoid is circular analysis: pick the bins
most variable across cell types, then 'show' they separate cell types.  That is
guaranteed on noise.  The defences pinned here are (1) selection filters AGAINST
batch, so sample-driven bins do not enter, and (2) the claim is validated on the
bins that were NOT selected -- if those separate cell types too, it is real.
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
    assert (out / "marker_heatmap_gch_mean_reclustered.png").exists()
    assert (out / "marker_heatmap_gch_mean_reclustered_batchcorrected.png"
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
    for suffix in ("fixedorder", "reclustered", "reclustered_batchcorrected"):
        assert (out / f"marker_heatmap_gch_compartment_{suffix}.png").exists()


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
    assert block(contrast).idxmax() == 4, "block 4 is the planted split"
    assert (contrast["contrast_tumour_fdr"] < 0.05).all()
    assert (contrast["contrast_tumour_diff"] > 0).all(), "in-set is the high arm"

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
