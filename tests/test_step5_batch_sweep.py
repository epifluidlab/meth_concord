"""Step-5 (RWR-basis) batch sweep.

This axis cannot be swept from a finished Δ matrix the way the Step-7 ones can:
correcting the accessibility matrix changes the k-NN graph, so it changes what
RWR imputes and therefore the concordance values.  Each setting needs its own
Step 5 — which is affordable only because steps 1-3 do not depend on the
correction and are replayed via ``precomputed``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance import cli, concordance


def _cells(tmp_path, n_cells=24, n_bins=20, seed=0):
    rng = np.random.default_rng(seed)
    cells = [f"c{i}" for i in range(n_cells)]
    grp = np.repeat(["A", "B", "C"], n_cells // 3)
    batch = np.array(["b1", "b2"] * (n_cells // 2))
    paths = {}
    for i, c in enumerate(cells):
        lvl = (0.2 + 0.3 * (grp[i] == "B") + 0.5 * (grp[i] == "C")
               + 0.15 * (batch[i] == "b2"))
        recs = []
        for b in range(n_bins):
            base = b * 5000
            for k in range(12):
                pp = lambda: float(np.clip(rng.normal(lvl, 0.1), 0, 1))  # noqa
                recs.append({"chrom": "chr1", "pos1": base + 100 + k * 3,
                             "pos2": base + 3000 + k * 3, "n1": 5, "n2": 5,
                             "posterior1": pp(), "posterior2": pp(),
                             "distance": 2900})
                recs.append({"chrom": "chr1", "pos1": base + 200 + k * 3,
                             "pos2": base + 220 + k * 3, "n1": 5, "n2": 5,
                             "posterior1": pp(), "posterior2": pp(),
                             "distance": 20})
        p = tmp_path / f"{c}.parquet"
        pd.DataFrame(recs).to_parquet(p)
        paths[c] = str(p)
    samples = pd.DataFrame({"cell_name": cells, "group": grp,
                            "exp_batch": batch,
                            "plate_batch": ["p1", "p2"] * (n_cells // 2)})
    return paths, samples


_KW = dict(bin_size=5000, min_distance=100, min_gch_sites=2,
           min_pairs_per_bin=1, use_rwr=True, rwr_k=3, eb_min_obs=2,
           n_threads=1, random_state=42)


def test_precomputed_reproduces_the_full_run_exactly(tmp_path):
    """The seam that makes the sweep affordable must not change the answer."""
    paths, _ = _cells(tmp_path)
    kw = dict(_KW, output_dir=str(tmp_path / "o"))
    full = concordance.compute_signed_concordance_matrix(
        paths, save_outputs=False, **kw)
    cache = {k: full[k] for k in ("raw_counts", "accessibility_raw",
                                  "cell_names", "sorted_bins",
                                  "uses_expected")}
    again = concordance.compute_signed_concordance_matrix(
        paths, precomputed=cache, save_outputs=False, **kw)
    a = full["delta_signed"]
    b = again["delta_signed"].reindex(index=a.index, columns=a.columns)
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_cache_carries_everything_needed_to_skip_steps_1_to_3(tmp_path):
    paths, _ = _cells(tmp_path)
    res = concordance.compute_signed_concordance_matrix(
        paths, save_outputs=False, **dict(_KW, output_dir=str(tmp_path / "o")))
    for key in ("raw_counts", "accessibility_raw", "cell_names",
                "sorted_bins", "uses_expected"):
        assert key in res, key
    # accessibility_raw must be PRE-correction, or a swept setting would be
    # applied on top of the first run's correction.
    assert res["accessibility_raw"].shape[0] == len(res["cell_names"])


def _args(tmp_path, methods="none,center,zscore", cols="exp_batch,plate_batch"):
    a = cli.parse_args(["--samples", "x", "--reference", "y",
                        "--output-dir", str(tmp_path),
                        "--step5-batch-sweep", methods,
                        "--batch-column", cols,
                        "--concordance-method", "signed", "--n-threads", "1"])
    a.min_cell_coverage = 0.0
    return a


def test_sweep_covers_every_method_and_column_subset(tmp_path):
    paths, samples = _cells(tmp_path)
    kw = dict(_KW, output_dir=str(tmp_path / "o"))
    first = concordance.compute_signed_concordance_matrix(
        paths, save_outputs=False, **kw)
    cli._step5_batch_sweep(first, paths, None, kw, samples,
                           _args(tmp_path), samples.set_index("cell_name")["group"],
                           str(tmp_path))
    tab = pd.read_csv(tmp_path / "step5_batch_sweep.tsv", sep="\t")
    # none(1) + center x 3 subsets + zscore x 3 subsets
    assert len(tab) == 7
    assert (tab["error"].isna() | (tab["error"] == "")).all(), tab["error"].tolist()
    assert tab["knn_purity"].notna().all()
    assert tab["knn_purity"].is_monotonic_decreasing


def test_sweep_needs_batch_column(tmp_path, caplog):
    paths, samples = _cells(tmp_path)
    kw = dict(_KW, output_dir=str(tmp_path / "o"))
    first = concordance.compute_signed_concordance_matrix(
        paths, save_outputs=False, **kw)
    a = _args(tmp_path)
    a.batch_column = None
    cli._step5_batch_sweep(first, paths, None, kw, samples, a,
                           samples.set_index("cell_name")["group"],
                           str(tmp_path))
    assert not (tmp_path / "step5_batch_sweep.tsv").exists()


def test_flag_defaults_off():
    a = cli.parse_args(["--samples", "x", "--reference", "y",
                        "--output-dir", "z"])
    assert a.step5_batch_sweep is None


def test_step7_resolution_is_shared_not_duplicated():
    """The sweep scores through _resolve_step7_params so a swept setting is
    judged under the same normalization the real Step 7 applies."""
    a = cli.parse_args(["--samples", "x", "--reference", "y",
                        "--output-dir", "z", "--concordance-method", "signed"])
    got = cli._resolve_step7_params(a)
    assert got["matrix_norm"] == "quantile" and got["per_cell_norm"] == "rank"
    assert got["skip_top_svd"] == 1 and got["n_svd"] == 10
