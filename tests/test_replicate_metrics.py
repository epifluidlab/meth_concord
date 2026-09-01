"""Cross-replicate reproducibility metrics and the diagnostic CLI.

The central claim these exist to support: ordinary kNN purity CANNOT
distinguish "cell types align across replicates" from "each replicate
recovered them separately".  Both give within-replicate purity 1.000.  The
first test asserts exactly that, because if it ever stopped being true the
whole metric would be unnecessary.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance.replicate_metrics import (batch_separation, purity,
                                                    svd_scores)

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "replicate_reproducibility.py")


def _two_worlds(seed=0, n_t=5, per=60):
    """Same cells and labels; types either aligned across replicates or
    duplicated per replicate."""
    rng = np.random.default_rng(seed)
    ct, rp, aligned, split = [], [], [], []
    for t in range(n_t):
        c = rng.normal(0, 1, 2) * 12
        for b in ("A", "B"):
            pts = c + rng.normal(0, 0.5, (per, 2))
            aligned.append(pts)
            split.append(pts + (np.array([40.0, 0.0]) if b == "B" else 0.0))
            ct += [f"T{t}"] * per
            rp += [b] * per
    return np.vstack(aligned), np.vstack(split), np.array(ct), np.array(rp)


def test_ordinary_purity_cannot_tell_the_two_apart_but_cross_can():
    """The reason this module exists."""
    aligned, split, ct, rp = _two_worlds()
    pa, ps = purity(aligned, ct, rp, k=25), purity(split, ct, rp, k=25)
    assert pa["within"] > 0.99 and ps["within"] > 0.99   # identical, and 1.0
    assert pa["ratio"] > 0.95                            # aligned
    assert ps["ratio"] < 0.5                             # duplicated
    assert abs(pa["chance"] - ps["chance"]) < 1e-12


def test_batch_separation_is_gameable_and_purity_is_not():
    """Noise overlaps the replicates perfectly while destroying the types --
    which is why batch separation is reported beside purity, never instead."""
    aligned, split, ct, rp = _two_worlds()
    noise = np.random.default_rng(1).normal(0, 1, aligned.shape)
    assert batch_separation(split, rp) > batch_separation(noise, rp)
    assert purity(noise, ct, rp, k=25)["ratio"] > 0.9     # looks "aligned"
    assert purity(noise, ct, rp, k=25)["within"] < 0.5    # but types are gone


def test_purity_survives_a_replicate_with_too_few_cells():
    aligned, _, ct, rp = _two_worlds()
    rp = rp.copy()
    rp[:] = "A"
    rp[0] = "B"                       # one cell in the second replicate
    p = purity(aligned, ct, rp, k=25)
    assert np.isfinite(p["within"])


def test_svd_scores_uses_the_pipelines_own_space():
    """per-cell rank then SVD: raw values would score a different space from
    the one the pipeline embeds."""
    rng = np.random.default_rng(2)
    M = pd.DataFrame(rng.normal(0, 1, (40, 30)))
    S = svd_scores(M, n_svd=5)
    assert S.shape == (40, 5)
    scaled = svd_scores(M * 100.0, n_svd=5)      # rank is scale-invariant
    assert np.allclose(np.abs(S), np.abs(scaled), atol=1e-6)


# ------------------------------------------------------------------- CLI --

@pytest.fixture
def cli_inputs(tmp_path):
    aligned, split, ct, rp = _two_worlds(per=30)
    cells = [f"c{i}" for i in range(len(ct))]
    rng = np.random.default_rng(5)
    nb = 40
    # a matrix that reproduces, and one that does not
    good = pd.DataFrame(aligned @ rng.normal(0, 1, (2, nb)), index=cells)
    bad = pd.DataFrame(split @ rng.normal(0, 1, (2, nb)), index=cells)
    good.to_parquet(tmp_path / "good.parquet")
    bad.to_parquet(tmp_path / "bad.parquet")
    pd.DataFrame({"cell": cells, "UMAP1": aligned[:, 0],
                  "UMAP2": aligned[:, 1]}).to_csv(
        tmp_path / "coords.tsv", sep="\t", index=False)
    sp = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells, "group": ct, "exp_batch": rp}).to_csv(
        sp, sep="\t", index=False)
    return tmp_path, sp


def _run(args, ok=True):
    r = subprocess.run([sys.executable, str(_SCRIPT)] + args,
                       capture_output=True, text=True)
    if ok:
        assert r.returncode == 0, r.stderr[-2000:]
    return r


def test_score_ranks_the_reproducible_matrix_above_the_other(cli_inputs):
    d, sp = cli_inputs
    r = _run(["score", str(sp), "--replicate-col", "exp_batch",
              "--celltype-col", "group",
              "--matrix", f"good={d / 'good.parquet'}",
              "--matrix", f"bad={d / 'bad.parquet'}", "--n-svd", "5"])
    rows = {ln.split()[0]: float(ln.split()[3])
            for ln in r.stdout.splitlines()
            if ln.strip().startswith(("good", "bad"))}
    assert rows["good"] > rows["bad"], r.stdout


def test_score_accepts_umap_coordinates(cli_inputs):
    d, sp = cli_inputs
    r = _run(["score", str(sp), "--replicate-col", "exp_batch",
              "--celltype-col", "group",
              "--matrix", f"umap={d / 'coords.tsv'}"])
    assert "umap" in r.stdout


def test_probe_reports_location_scale_and_concentration(cli_inputs):
    d, sp = cli_inputs
    r = _run(["probe", str(sp), "--replicate-col", "exp_batch",
              "--celltype-col", "group",
              "--matrix", f"bad={d / 'bad.parquet'}", "--n-svd", "5"])
    for token in ("LOCATION", "SCALE", "80% of the replicate separation",
                  "AUC"):
        assert token in r.stdout, token


def test_probe_refuses_coordinates(cli_inputs):
    d, sp = cli_inputs
    r = _run(["probe", str(sp), "--replicate-col", "exp_batch",
              "--celltype-col", "group",
              "--matrix", f"umap={d / 'coords.tsv'}"], ok=False)
    assert r.returncode != 0 and "parquet" in r.stderr


def test_a_missing_column_lists_what_is_available(cli_inputs):
    d, sp = cli_inputs
    r = _run(["score", str(sp), "--replicate-col", "nope",
              "--celltype-col", "group",
              "--matrix", f"good={d / 'good.parquet'}"], ok=False)
    assert "nope" in r.stderr and "exp_batch" in r.stderr


def test_matrix_spec_must_be_named(cli_inputs):
    d, sp = cli_inputs
    r = _run(["score", str(sp), "--replicate-col", "exp_batch",
              "--celltype-col", "group",
              "--matrix", str(d / "good.parquet")], ok=False)
    assert "NAME=PATH" in r.stderr
