"""Label-agreement metrics used to caption the UMAPs.

The point of these metrics is to RANK representations, so the tests pin
ordering (separated > mixed) and the one property that motivates reporting
ASW and kNN-purity together: a fragmented-but-clean labelling scores high on
purity and poorly on ASW.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance import metrics


def _blobs(sep: float, n=40, seed=0):
    """Two labelled gaussian blobs ``sep`` apart in 2-D."""
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0, 1, (n, 2)),
                   rng.normal(sep, 1, (n, 2))])
    y = pd.Series(["A"] * n + ["B"] * n)
    return X, y


def test_separated_beats_mixed_on_every_metric():
    far, y = _blobs(12.0)
    near, _ = _blobs(0.0)
    m_far = metrics.label_agreement(far, y)
    m_near = metrics.label_agreement(near, y)
    for key in ("silhouette", "knn_purity", "ari", "nmi"):
        assert m_far[key] > m_near[key], key
    assert m_far["knn_purity"] > 0.95
    assert m_far["ari"] > 0.95
    # Overlapping blobs sit at chance, not at zero.
    assert m_near["knn_purity"] == pytest.approx(0.5, abs=0.12)


def test_knn_chance_is_the_imbalance_baseline():
    # 90/10 split: random neighbours already give .9^2 + .1^2 = 0.82.
    y = pd.Series(["A"] * 90 + ["B"] * 10)
    assert metrics.knn_chance(y) == pytest.approx(0.82)
    # Ten balanced classes -> chance 0.1, so a purity of 0.5 is real signal.
    assert metrics.knn_chance(pd.Series(list(range(10)) * 10)) == \
        pytest.approx(0.1)


def test_fragmented_labels_keep_purity_but_lose_silhouette():
    """The reason both are reported: splitting a label into two distant
    islands leaves it locally pure but drops its ASW sharply.

    Stated as a CONTRAST against the same data unfragmented — an absolute ASW
    threshold would just be pinning this fixture's geometry (the other,
    compact label averages the score back up).
    """
    rng = np.random.default_rng(1)
    y = pd.Series(["A"] * 40 + ["B"] * 40)
    a = rng.normal(0, 0.3, (40, 2))
    compact = np.vstack([a, rng.normal(8, 0.3, (40, 2))])
    split = np.vstack([a, rng.normal(-8, 0.3, (20, 2)),
                       rng.normal(8, 0.3, (20, 2))])
    m_compact = metrics.label_agreement(compact, y, k=5)
    m_split = metrics.label_agreement(split, y, k=5)
    assert m_compact["knn_purity"] > 0.95 and m_split["knn_purity"] > 0.95, \
        "both are locally pure — fragmentation is invisible to kNN-purity"
    assert m_split["silhouette"] < m_compact["silhouette"] - 0.2, \
        "but ASW registers the split"


def test_unlabelled_cells_are_dropped_not_treated_as_a_class():
    X, y = _blobs(12.0)
    y_missing = y.copy()
    y_missing.iloc[:10] = None
    m = metrics.label_agreement(X, y_missing)
    assert m["n_cells"] == 70          # 80 - 10 dropped
    assert m["n_groups"] == 2          # NOT 3
    assert m["knn_purity"] > 0.9


def test_degenerate_inputs_give_nan_not_an_exception():
    X, y = _blobs(5.0)
    for labels in (None, pd.Series(["A"] * len(y)), pd.Series([None] * len(y))):
        m = metrics.label_agreement(X, labels)
        assert np.isnan(m["knn_purity"]), labels
    assert metrics.format_metrics(metrics.label_agreement(X, None)) == ""


def test_format_metrics_is_a_single_readable_line():
    X, y = _blobs(12.0)
    line = metrics.format_metrics(metrics.label_agreement(X, y), "10 SVD comps")
    for token in ("ASW", "kNN-purity", "chance", "ARI", "NMI", "10 SVD comps"):
        assert token in line, token
    assert "\n" not in line


def test_umap_writes_metrics_tsv_and_captions_the_figure(tmp_path):
    pytest.importorskip("umap")
    from nome_hic_concordance import visualization
    X, y = _blobs(12.0)
    cells = [f"c{i}" for i in range(X.shape[0])]
    mat = pd.DataFrame(X, index=cells, columns=["b1", "b2"])
    labels = pd.Series(y.to_numpy(), index=cells)
    visualization.run_umap_and_plot(
        mat, labels, output_prefix="m", output_dir=str(tmp_path),
        n_threads=1, random_state=0,
    )
    out = pd.read_csv(tmp_path / "m_metrics.tsv", sep="\t")
    assert set(out["space"]) == {"feature", "umap2d"}
    assert (out["knn_purity"] > 0.9).all()
    assert (tmp_path / "m.png").exists()


def test_comparison_reuses_the_shared_implementation():
    """comparison.py must not carry a second copy that can drift."""
    from nome_hic_concordance import comparison
    assert comparison._silhouette is metrics.silhouette
    assert comparison._knn_purity is metrics.knn_purity
