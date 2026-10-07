"""ep_coordination_existence.py --null: the 6b ladder against a drawn null.

At 1-3 pairs per region Δ's SPREAD depends on the methylation level, and
regressing out level, level^2 and coverage removes only its average, so a
random forest separates cell types that differ in level alone.  The drawn
null -- every contact replaced by a non-contact pair of the cell's own ends,
same region, matched distance -- carries exactly that and no coordination.
These tests pin both directions: with no coordination the real Δ is just one
more null draw; with planted coordination it clears every draw.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ep_coordination_existence",
    Path(__file__).resolve().parents[1] / "scripts" / "ep_coordination_existence.py")
ep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ep)


def make_cells(out, coord=0.0, per=30, n_types=3, chroms=3, chrom_mb=20,
               contacts=2.0, seed=0):
    """Posteriors for cell types that differ in methylation level in 30% of
    1 Mb regions.  ``coord`` > 0 also couples the two ends of a contact --
    in the regions whose index mod n_types is the cell's type."""
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base = rng.uniform(0.6, 0.9, (chroms, chrom_mb))
    level = np.repeat(base[None], n_types, 0)
    diff = rng.random((chroms, chrom_mb)) < 0.3
    level[:, diff] = np.clip(
        level[:, diff] + rng.normal(0, 0.25, (n_types, int(diff.sum()))),
        0.05, 0.95)
    rows = []
    for t in range(n_types):
        for c in range(per):
            rec = []
            for ch in range(chroms):
                for r in range(chrom_mb):
                    m, lo = level[t, ch, r], r * 1_000_000

                    def end():
                        return 0.95 if rng.random() < m else 0.05

                    own = (ch * chrom_mb + r) % n_types == t
                    for _ in range(rng.poisson(contacts)):
                        a = lo + int(rng.integers(0, 1_000_000))
                        b = a + int(rng.integers(20_000, 400_000))
                        pa = end()
                        pb = pa if own and rng.random() < coord else end()
                        rec.append((f"chr{ch + 1}", a, b, pa, pb))
                    for _ in range(rng.poisson(12)):      # the E end pool
                        a = lo + int(rng.integers(0, 1_000_000))
                        rec.append((f"chr{ch + 1}", a,
                                    a + int(rng.integers(200, 900)),
                                    end(), end()))
            df = pd.DataFrame(rec, columns=["chrom", "pos1", "pos2",
                                            "posterior1", "posterior2"])
            df["n1"] = np.int32(3)
            df["n2"] = np.int32(3)
            df.to_parquet(out / f"t{t}_c{c}.posteriors.parquet")
            rows.append((f"t{t}_c{c}", f"T{t}"))
    pd.DataFrame(rows, columns=["cell_name", "group"]).to_csv(
        out / "samples.tsv", sep="\t", index=False)
    return out


def run(d, classifier="rf", draws=4, extra=()):
    prefix = str(Path(d) / "o")
    with contextlib.redirect_stdout(io.StringIO()):
        ep.main(["--input-dir", str(d), "--samples", str(Path(d) / "samples.tsv"),
                 "--min-distance", "20000", "--max-distance", "2000000",
                 "--min-gch-sites", "1", "--granularities", "1",
                 "--classifier", classifier, "--min-region-frac", "0.5",
                 "--n-perm", "0", "--n-threads", "1", "--null",
                 "--null-draws", str(draws), "--output-prefix", prefix,
                 *extra])
    t = pd.read_csv(prefix + ".ep_existence.csv")
    real = float(t.loc[t["data"] == "real",
                       "concordance_resid_full_auc"].iloc[0])
    null = t.loc[t["data"] == "drawn_null",
                 "concordance_resid_full_auc"].to_numpy(float)
    return real, null, t


# ------------------------------------------------------------- behaviour --


def test_with_no_coordination_the_real_is_one_more_null_draw(tmp_path):
    """Cell types differ only in level, and the forest still separates them
    (well above 0.5) -- but so does every null draw."""
    real, null, _ = run(make_cells(tmp_path, coord=0.0, seed=0))
    assert null.size == 4
    assert real - null.mean() < 0.1
    assert real <= null.max() + 0.02


def test_planted_coordination_clears_every_null_draw(tmp_path):
    real, null, _ = run(make_cells(tmp_path, coord=0.9, seed=1))
    assert real > null.max()
    assert real - null.mean() > 0.1


# ---------------------------------------------------------------- pieces --


def test_a_null_pair_comes_from_the_contacts_region_and_distance():
    dbins = ep._null_dbins(20_000, 2_000_000)
    b_reg = np.array([7, 7, 7, 9])
    b_dist = np.array([25_000, 900_000, 1_250_000, 25_000])
    pick, other = ep._draw_null(np.array([7, 7, 9, 5]),
                                np.array([1_300_000, 26_000, 30_000, 26_000]),
                                b_reg, b_dist, dbins,
                                np.random.default_rng(0))
    assert pick[0] == 2 and not other[0]           # region 7, 1.12-1.5 Mb bin
    assert pick[1] == 0 and not other[1]           # region 7, ~25 kb bin
    assert pick[2] == 3 and other[2]               # region 9, no match -> any
    assert pick[3] == -1                           # region 5 has no pool


def test_read_pairs_are_recognised_in_either_order():
    keys = ep._pair_keys([100, 5_000], [9_000, 80_000])
    assert ep._is_pair([100, 5_000, 100], [9_000, 80_000, 80_000],
                       keys).tolist() == [True, True, False]
    assert not ep._is_pair([1], [2], None).any()


def test_each_region_gets_as_many_null_pairs_as_contacts(tmp_path):
    d = make_cells(tmp_path, per=2, chroms=1, chrom_mb=10, contacts=3)
    path = sorted(d.glob("*.posteriors.parquet"))[0]
    _, out, _, null_part = ep.process_cell(
        path, [1], 20_000, 2_000_000, 1, 4000, 0, None, True, 3)
    stats = null_part["stats"]["1"]
    assert stats[2] == 0                            # every contact got one
    n_null = null_part["out"]["1"]["n"]
    assert n_null.shape == (3, out["1"]["keys"].size)
    for draw in n_null:                             # same regions, same keys
        assert np.array_equal(draw, out["1"]["n"])


def test_without_null_nothing_changes_shape(tmp_path):
    d = make_cells(tmp_path, per=1, chroms=1, chrom_mb=5)
    path = sorted(d.glob("*.posteriors.parquet"))[0]
    cell, out, _, null_part = ep.process_cell(path, [1], 20_000, 2_000_000,
                                              1, 4000, 0, None)
    assert null_part is None and set(out) == {"1"}
    o = out["1"]
    assert o["keys"].size == o["delta"].size == o["marg"].size == o["n"].size


def test_the_csv_tags_the_draws_and_the_matrices_are_written(tmp_path):
    d = make_cells(tmp_path, per=8, chroms=2, chrom_mb=10)
    real, null, t = run(d, classifier="lr", draws=2)
    assert t["data"].tolist() == ["real", "drawn_null", "drawn_null"]
    assert t["draw"].tolist() == [-1, 0, 1]
    for stem in ("delta", "delta_null", "delta_mcov", "delta_mcov_null"):
        m = pd.read_parquet(d / f"o.1mb.{stem}.parquet")
        assert m.shape[1] > 0 and str(m.columns[0]).startswith("chr")


def test_a_null_draw_computes_only_what_it_is_compared_on(tmp_path):
    """Two forests per draw by default; every column with the flag."""
    d = make_cells(tmp_path, per=8, chroms=2, chrom_mb=10)
    _, _, core = run(d, classifier="lr", draws=1)
    null = core[core["data"] == "drawn_null"].iloc[0]
    assert np.isfinite(null["concordance_resid_full_auc"])
    assert np.isnan(null["concordance_auc"]) and np.isnan(null["missingness_auc"])
    _, _, full = run(d, classifier="lr", draws=1,
                     extra=("--null-full-ladder",))
    null = full[full["data"] == "drawn_null"].iloc[0]
    assert np.isfinite(null["concordance_auc"])
    assert null["concordance_resid_full_auc"] == pytest.approx(
        core[core["data"] == "drawn_null"].iloc[0]["concordance_resid_full_auc"])
