"""Step 5's link-shuffle E and drawn null: each contact keeps its left end;
E is the anchor with the mean long-range read end of one half of its
stratum's OTHER cells, in the same bin at a matched distance, and the null
one such end per draw from the other half -- assigned for the whole stratum
at once (nome_hic_concordance.link_shuffle)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nome_hic_concordance import concordance as cc
from nome_hic_concordance import link_shuffle as ls


def _cell(path, rows):
    df = pd.DataFrame(rows, columns=["chrom", "pos1", "pos2", "posterior1",
                                     "posterior2"])
    df["n1"] = np.int32(3)
    df["n2"] = np.int32(3)
    df["distance"] = (df["pos2"] - df["pos1"]).abs().astype(np.int64)
    df.to_parquet(path)
    return str(path)


def test_o_e_and_the_null_are_built_on_the_assigned_contacts(tmp_path):
    rows = [("chr1", 100_000 + i * 30_000, 105_000 + i * 30_000, 0.9, 0.8)
            for i in range(4)]                       # contacts, rows 0-3
    rows.append(("chr1", 405_000, 400_000, 0.3, 0.7))  # row 4: left end is pos2
    rows += [("chr1", 100_000 + i * 7_000, 100_300 + i * 7_000, 0.05, 0.05)
             for i in range(10)]                     # short pairs
    path = _cell(tmp_path / "c.parquet", rows)
    link = {"row": np.arange(5),
            "level": np.array([0, 1, 0, -1, 0]),
            "m": np.array([0.5, 0.25, 0.5, np.nan, 0.5]),
            "donor_p": np.array([[0.2, 0.4, 0.6, np.nan, 0.0],
                                 [1.0, 0.0, 1.0, np.nan, 1.0]])}
    _, out = cc._compute_cell_dm_expected(
        path, "c", 1_000_000, 1_000, 2, mode="pooled", null_draws=2,
        link_assign=link)
    r = out["chr1:0"]
    # row 3 has no donor: out of O, E and the null alike
    assert r["n_linked"] == r["n_dm"] == 4
    assert r["k_open"] == pytest.approx(3 * 0.9 * 0.8 + 0.7 * 0.3)
    assert r["k_dm_open"] == pytest.approx(0.9 * 1.25 + 0.7 * 0.5)
    assert r["k_dm_closed"] == pytest.approx(0.1 * 1.75 + 0.3 * 0.5)
    for d, want in ((0, 0.9 * 1.2), (1, 0.9 * 2 + 0.7)):
        ko, _, _, kn = cc._null_keys(d)
        assert r[kn] == 4 and r[ko] == pytest.approx(want)
    assert (r["n_ls_own_bin"], r["n_ls_widened"], r["n_ls_none"]) == (3, 1, 1)


def test_the_assignment_finds_every_contact_step5_scores(tmp_path):
    """read_cell's contacts are exactly step 5's linked pairs -- same bin,
    in band, both ends called -- or a contact would be dropped as donorless
    when its assignment was only misfiled."""
    rng = np.random.default_rng(0)
    paths = []
    for c in range(6):
        rows = []
        for _ in range(200):
            a = int(rng.integers(0, 4_000_000))
            b = a + int(rng.choice([300, 30_000, 400_000, 1_500_000]))
            rows.append(("chr1", a, b, float(rng.random()),
                         float(rng.random())))
        paths.append(_cell(tmp_path / f"c{c}.parquet", rows))
    cells = [ls.read_cell(p, 20_000, 2, 1_000_000, 1_000_000) for p in paths]
    out, summary = ls.assign_all(cells, ["S"] * 6,
                                 ls.log_dbins(20_000, 1_000_000, 4),
                                 n_draws=1, seed=0, bin_size=1_000_000)
    none = 0.0
    total = 0.0
    for i, p in enumerate(paths):
        _, res = cc._compute_cell_dm_expected(
            p, f"c{i}", 1_000_000, 20_000, 2, mode="pooled", null_draws=1,
            max_distance=1_000_000, link_assign=out[i])
        none += sum(e["n_ls_none"] for e in res.values())
        total += sum(e["n_ls_own_bin"] + e["n_ls_widened"] + e["n_ls_none"]
                     for e in res.values())
    assert total == summary["contacts"] > 0
    assert none == summary["none"]


def test_step5_runs_a_link_shuffle_null_end_to_end(tmp_path, caplog):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "_null_rank_fixture", Path(__file__).with_name("test_null_rank.py"))
    nr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nr)
    paths = nr.null_rank_paths(tmp_path, n_cells=8, n_bins=20)
    strata = pd.Series({c: "AB"[i % 2] for i, c in enumerate(sorted(paths))})
    with caplog.at_level("INFO"):
        res = cc.compute_signed_concordance_matrix(
            paths, bin_size=10_000, min_distance=500, min_gch_sites=2,
            baseline="distance-pooled", dm_prior="shared",
            output_dir=str(tmp_path / "out"), save_outputs=False,
            use_rwr=False, n_threads=1, eb_min_obs=50, emit_null=True,
            null_draws=2, null_type="link-shuffle", null_strata=strata)
    assert res["null_components"] is not None
    msgs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("Link-shuffle E and null") for m in msgs)
    assert any(m.startswith("Link-shuffle: of ") for m in msgs)
    assert not any("own read pairs" in m for m in msgs)


def test_the_null_type_is_checked():
    with pytest.raises(ValueError, match="null_type"):
        cc.compute_signed_concordance_matrix({}, null_type="shuffle")
