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
    mat = tmp_path / "gch.parquet"
    pd.DataFrame(X, index=samples,
                 columns=[f"chr1:{i * 250000}" for i in range(X.shape[1])]
                 ).to_parquet(mat)
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
    scores = pd.read_csv(out / "bin_scores_gch.tsv", sep="\t")
    assert scores["selected"].sum() == 30
    # the out-of-selection control must fire: non-selected bins separate types
    import json
    s = json.loads((out / "marker_bins_summary.json").read_text())["matrices"]["gch"]
    assert s["nonselected_perm_p"] is not None
    assert s["nonselected_perm_p"] < 0.05
    assert s["median_celltype_icc_selected"] > 0.5
