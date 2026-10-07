"""Step 5's link-shuffle drawn null: each contact keeps its left end and its
partner becomes a long-range read end from ANOTHER cell of the stratum, in
the same bin at a matched distance; E is built from short-read-pair ends."""

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


def test_contact_ends_stay_out_of_e_and_donors_come_from_the_pool(tmp_path):
    rows = [("chr1", 100_000 + i * 30_000, 105_000 + i * 30_000, 0.95, 0.95)
            for i in range(20)]                      # contacts: methylated
    rows += [("chr1", 100_000 + i * 7_000, 100_300 + i * 7_000, 0.05, 0.05)
             for i in range(80)]                     # short pairs: not
    path = _cell(tmp_path / "c.parquet", rows)
    pool = ls.build_pool({1: (np.arange(100_000, 900_000, 1_000, np.int64),
                              np.full(800, 0.5))})
    _, link = cc._compute_cell_dm_expected(
        path, "c", 1_000_000, 1_000, 2, mode="pooled", null_draws=2,
        link_pool=pool, cell_id=0, chrom_code={"chr1": 0})
    _, plain = cc._compute_cell_dm_expected(path, "c", 1_000_000, 1_000, 2,
                                            mode="pooled")
    rl, rp = link["chr1:0"], plain["chr1:0"]
    assert rl["k_dm_open"] / rl["n_dm"] < 0.01       # 0.05^2: short ends only
    assert rp["k_dm_open"] / rp["n_dm"] > 0.05       # contact ends in it
    for d in range(2):                               # every contact, a donor
        ko, kc, km, kn = cc._null_keys(d)
        assert rl[kn] == 20
        assert rl[ko] == pytest.approx(20 * 0.95 * 0.5)
    assert rl["n_ls_none"] == 0


def test_a_cell_never_donates_to_itself(tmp_path):
    rows = [("chr1", 100_000, 105_000, 0.95, 0.95)]
    rows += [("chr1", 100_000 + i * 7_000, 100_300 + i * 7_000, 0.05, 0.05)
             for i in range(40)]
    path = _cell(tmp_path / "c.parquet", rows)
    # the only donor near the partner is this cell's own (id 0)
    pool = ls.build_pool({0: (np.array([105_000], np.int64), np.array([0.95]))})
    _, out = cc._compute_cell_dm_expected(
        path, "c", 1_000_000, 1_000, 2, mode="pooled", null_draws=1,
        link_pool=pool, cell_id=0, chrom_code={"chr1": 0})
    rec = out["chr1:0"]
    assert rec[cc._null_keys(0)[3]] == 0 and rec["n_ls_none"] == 1


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
    assert any(m.startswith("Drawn null (link-shuffle)") for m in msgs)
    assert not any("own read pairs" in m for m in msgs)


def test_the_null_type_is_checked():
    with pytest.raises(ValueError, match="null_type"):
        cc.compute_signed_concordance_matrix({}, null_type="shuffle")
