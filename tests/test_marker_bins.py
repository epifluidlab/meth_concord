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
