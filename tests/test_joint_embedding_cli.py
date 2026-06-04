"""End-to-end smoke test for scripts/joint_multistate_embedding.py.

Validates the J3 wiring (tensor build → per-channel embedding → channel AUC →
scores CSV) on synthetic data. UMAP/tensorly are absent in CI, so the script
must still succeed and write the embedding + channel-AUC tables, skipping the
plot/tensor with a message."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / \
    "joint_multistate_embedding.py"
_COLS = ["chrom", "pos1", "pos2", "n1", "n2", "posterior1", "posterior2"]


def _write_cell(gdir, hdir, cell, seed, group):
    rng = np.random.default_rng(seed)
    g, h = [], []
    for ci, chrom in enumerate(("chr1", "chr2")):
        p = 1_000_000 + 10_000 * np.arange(40)
        # weak group-specific regional pattern in accessibility/methylation
        shift = 0.15 if group == "A" else -0.15
        a = np.clip(0.5 + (shift if ci == 0 else -shift) + 0.2 * rng.random(40),
                    0.05, 0.95)
        m = np.clip(0.5 - (shift if ci == 0 else -shift) + 0.2 * rng.random(40),
                    0.05, 0.95)
        for i in range(40):
            for k in (2, 3, 4):
                if i + k < 40:
                    g.append((chrom, p[i], p[i + k], 5, 5, a[i], a[i + k]))
                    nc = 5 if ci == 0 else 0
                    h.append((chrom, p[i], p[i + k], nc, nc,
                              m[i] if nc else np.nan, m[i + k] if nc else np.nan))
    pd.DataFrame(g, columns=_COLS).to_parquet(
        gdir / f"{cell}.posteriors.parquet", index=False)
    pd.DataFrame(h, columns=_COLS).to_parquet(
        hdir / f"{cell}.posteriors.parquet", index=False)


def _run(tmp_path, extra):
    gdir = tmp_path / "gch"; hdir = tmp_path / "hcg"; gdir.mkdir(); hdir.mkdir()
    cells, groups = [], []
    for i in range(18):
        grp = "A" if i % 2 == 0 else "B"
        _write_cell(gdir, hdir, f"C{i:02d}", seed=i, group=grp)
        cells.append(f"C{i:02d}"); groups.append(grp)
    samples = tmp_path / "samples.tsv"
    pd.DataFrame({"cell_name": cells, "group": groups}).to_csv(
        samples, sep="\t", index=False)
    (tmp_path / "out").mkdir()
    prefix = tmp_path / "out" / "emb"
    cmd = [sys.executable, str(_SCRIPT),
           "--gch-dir", str(gdir), "--hcg-dir", str(hdir),
           "--samples", str(samples), "--granularity", "chrom",
           "--min-distance", "10000", "--max-distance", "50000",
           "--min-cpg-sites", "1", "--min-region-frac", "0.0",
           "--k-per-channel", "3", "--n-folds", "3", "--n-threads", "1",
           "--output-prefix", str(prefix)] + extra
    return subprocess.run(cmd, capture_output=True, text=True), prefix


def test_embedding_cli_end_to_end(tmp_path):
    r, prefix = _run(tmp_path, ["--feature", "delta"])
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "per-channel component AUC" in r.stdout
    # UMAP/tensor absent in CI -> skipped gracefully, run still succeeds
    assert "UMAP skipped" in r.stdout

    scores = Path(f"{prefix}.embedding_scores.csv")
    chauc = Path(f"{prefix}.channel_auc.csv")
    assert scores.exists() and chauc.exists()
    sdf = pd.read_csv(scores, index_col=0)
    assert sdf.shape[0] == 18 and "group" in sdf.columns
    adf = pd.read_csv(chauc)
    assert set(adf["channel"]) >= {"open", "closed", "poised",
                                   "active_methylated", "joint_all"}
    assert np.isfinite(adf["auc"]).all()


def test_embedding_cli_resid_full(tmp_path):
    r, prefix = _run(tmp_path, ["--feature", "resid_full", "--tensor"])
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    # tensorly absent -> tensor step skipped, run still succeeds
    assert "tensor decomposition skipped" in r.stdout or \
           "channel factor" in r.stdout
    assert Path(f"{prefix}.embedding_scores.csv").exists()
