"""End-to-end smoke test for scripts/joint_coordination_existence.py.

Validates the orchestration wiring (prior scan, parallel map, confound ladder,
table, CSV, batch + per-channel) on synthetic data — not the science (the AUCs
are ~chance on random data; process_cell's numerics are tested separately)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / \
    "joint_coordination_existence.py"


def _write_cell(gdir, hdir, cell, seed, hcg_observed_chr2=False):
    rng = np.random.default_rng(seed)
    g_rows, h_rows = [], []
    for ci, chrom in enumerate(("chr1", "chr2")):
        n_loci = 30
        p = 1_000_000 + 10_000 * np.arange(n_loci)
        a = 0.05 + 0.9 * rng.random(n_loci)
        m = 0.05 + 0.9 * rng.random(n_loci)
        for i in range(n_loci):
            for k in (2, 3):
                if i + k < n_loci:
                    g_rows.append((chrom, p[i], p[i + k], 5, 5, a[i], a[i + k]))
                    # methylation observed on chr1; on chr2 only if requested
                    n_cpg = 5 if (ci == 0 or hcg_observed_chr2) else 0
                    mm1 = m[i] if n_cpg else np.nan
                    mm2 = m[i + k] if n_cpg else np.nan
                    h_rows.append((chrom, p[i], p[i + k], n_cpg, n_cpg, mm1, mm2))
    cols = ["chrom", "pos1", "pos2", "n1", "n2", "posterior1", "posterior2"]
    pd.DataFrame(g_rows, columns=cols).to_parquet(
        gdir / f"{cell}.posteriors.parquet", index=False)
    pd.DataFrame(h_rows, columns=cols).to_parquet(
        hdir / f"{cell}.posteriors.parquet", index=False)


def test_joint_cli_end_to_end(tmp_path):
    gdir = tmp_path / "gch"; hdir = tmp_path / "hcg"
    gdir.mkdir(); hdir.mkdir()
    cells, groups, batches = [], [], []
    for i in range(18):
        cell = f"C{i:02d}"
        _write_cell(gdir, hdir, cell, seed=i)
        cells.append(cell)
        groups.append("A" if i % 2 == 0 else "B")
        batches.append("b1" if i % 3 == 0 else "b2")
    samples = tmp_path / "samples.tsv"
    pd.DataFrame({"cell_name": cells, "group": groups, "batch": batches}) \
        .to_csv(samples, sep="\t", index=False)

    prefix = tmp_path / "out" / "j"
    (tmp_path / "out").mkdir()
    cmd = [sys.executable, str(_SCRIPT),
           "--gch-dir", str(gdir), "--hcg-dir", str(hdir),
           "--samples", str(samples),
           "--granularities", "chrom",
           "--min-distance", "10000", "--max-distance", "50000",
           "--min-gpc-sites", "2", "--min-cpg-sites", "1",
           "--n-perm", "3", "--n-folds", "3", "--n-threads", "1",
           "--per-channel", "--batch-column", "batch",
           "--output-prefix", str(prefix)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"

    # Headline + QC lines printed
    assert "joint(4ch)" in r.stdout
    assert "global priors" in r.stdout
    assert "QC:" in r.stdout

    csv = Path(f"{prefix}.joint_existence.csv")
    assert csv.exists(), r.stdout
    df = pd.read_csv(csv)
    feats = set(df["feature"])
    assert {"joint", "gch", "hcg"}.issubset(feats)
    # per-channel rows present
    assert any(f.startswith("joint:") for f in feats)
    assert "joint:open" in feats and "joint:active_methylated" in feats
    # the three main feature sets produced finite AUCs at the +cov rung
    main = df[df["feature"].isin(["joint", "gch", "hcg"])]
    assert np.isfinite(main["resid_full_auc"]).all()
    # batch control column populated (since --batch-column given)
    assert np.isfinite(main["resid_full_batch_auc"]).all()
    # joint has 8 features (2 chroms × 4 channels) at chrom granularity
    assert int(df[df["feature"] == "joint"]["n_features"].iloc[0]) == 8


def test_joint_cli_no_optional_flags(tmp_path):
    """Runs without --per-channel / --batch-column / --output-prefix."""
    gdir = tmp_path / "gch"; hdir = tmp_path / "hcg"
    gdir.mkdir(); hdir.mkdir()
    cells, groups = [], []
    for i in range(12):
        cell = f"C{i:02d}"
        _write_cell(gdir, hdir, cell, seed=100 + i, hcg_observed_chr2=True)
        cells.append(cell); groups.append("A" if i % 2 == 0 else "B")
    samples = tmp_path / "samples.tsv"
    pd.DataFrame({"cell_name": cells, "group": groups}) \
        .to_csv(samples, sep="\t", index=False)
    cmd = [sys.executable, str(_SCRIPT),
           "--gch-dir", str(gdir), "--hcg-dir", str(hdir),
           "--samples", str(samples), "--granularities", "chrom",
           "--min-distance", "10000", "--max-distance", "50000",
           "--min-cpg-sites", "1", "--n-perm", "0", "--n-folds", "3",
           "--n-threads", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "joint(4ch)" in r.stdout
