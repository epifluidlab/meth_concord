"""End-posterior distance correlation.

The site-level script correlates single 0/1 calls, which at one cytosine on one
molecule is mostly Bernoulli noise.  This one correlates the per-END posteriors,
each already an average over that end's informative sites.

The fixture plants a known correlation through a shared latent per pair, so the
tests check that it is recovered, that the nulls destroy it, and that all three
measures are compared against the SAME synthetic pairs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "end_posterior_distance_correlation.py")


def _mod():
    spec = importlib.util.spec_from_file_location("epd", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["epd"] = m
    spec.loader.exec_module(m)
    return m


def _fixture(tmp_path, n_cells=8, rho_g=0.9, rho_h=0.4, seed=0):
    """Both contexts read off the SAME read pairs, as the pipeline emits them."""
    rng = np.random.default_rng(seed)
    for ctx in ("GCH", "HCG"):
        (tmp_path / ctx).mkdir(parents=True, exist_ok=True)
    cells = [f"c{i}" for i in range(n_cells)]
    for c in cells:
        rows = []
        for b in range(40):
            base = b * 20000
            for k in range(15):
                d = int(rng.integers(500, 15000))
                rows.append((base + k * 40, base + k * 40 + d, d))
        recs = {"GCH": [], "HCG": []}
        for (p1, p2, d) in rows:
            shared = rng.normal(0, 1)
            for ctx, rho in (("GCH", rho_g), ("HCG", rho_h)):
                f = lambda: float(1 / (1 + np.exp(                     # noqa
                    -(shared * rho + rng.normal(0, 0.6)))))
                recs[ctx].append({"chrom": "chr1", "pos1": p1, "pos2": p2,
                                  "n1": 6, "n2": 6, "distance": d,
                                  "posterior1": f(), "posterior2": f()})
        for ctx in ("GCH", "HCG"):
            pd.DataFrame(recs[ctx]).to_parquet(
                tmp_path / ctx / f"{c}.posteriors.parquet")
    sp = tmp_path / "s.tsv"
    pd.DataFrame({"cell_name": cells,
                  "group": np.repeat(["A", "B"], n_cells // 2),
                  "exp_batch": ["b1", "b2"] * (n_cells // 2),
                  "plate_batch": ["p1", "p2"] * (n_cells // 2)}).to_csv(
        sp, sep="\t", index=False)
    return str(sp)


def _run(tmp_path, extra=None):
    m = _mod()
    sp = _fixture(tmp_path)
    out = tmp_path / "out"
    argv = ["--posteriors", f"GCH={tmp_path / 'GCH'}",
            "--posteriors", f"HCG={tmp_path / 'HCG'}",
            "--samples", sp, "--min-pairs", "50", "--plot-format", "png",
            "--output-dir", str(out)] + (extra or [])
    assert m.main(argv) == 0
    return pd.read_csv(out / "end_posterior_dist_all.tsv", sep="\t"), out


# ------------------------------------------------------------- correctness --

def test_recovers_the_planted_correlation_and_its_ordering(tmp_path):
    res, _ = _run(tmp_path)
    ok = res[res["gg_linked_n"] > 0]
    gg = ok["gg_linked_r"].median()
    hh = ok["hh_linked_r"].median()
    assert gg > 0.5, "the strongly coupled context must show a clear r"
    assert hh > 0.15
    assert gg > hh, "rho_g > rho_h was planted, so gg must exceed hh"


def test_the_null_destroys_the_pairing_for_every_measure(tmp_path):
    res, _ = _run(tmp_path)
    ok = res[res["gg_null_n"] > 0]
    for m in ("gg", "hh", "gh"):
        assert abs(ok[f"{m}_null_r"].median()) < 0.1, m
        assert (ok[f"{m}_linked_r"].median()
                - ok[f"{m}_null_r"].median()) > 0.1, m


def test_all_three_measures_get_a_null(tmp_path):
    """Regression: an earlier version built per-context nulls only, so the
    cross-context panel had no null and its Δ was undefined."""
    res, _ = _run(tmp_path)
    for m in ("gg", "hh", "gh"):
        assert (res[f"{m}_null_n"] > 0).any(), f"{m} has no null at any distance"


def test_cross_context_needs_both_contexts_on_the_same_pair(tmp_path):
    m = _mod()
    sp = _fixture(tmp_path)
    g = pd.read_parquet(tmp_path / "GCH" / "c0.posteriors.parquet")
    h = pd.read_parquet(tmp_path / "HCG" / "c0.posteriors.parquet")
    ren = {"posterior1": "p1", "posterior2": "p2"}
    j = m.joined_pairs({"GCH": g.rename(columns=ren), "HCG": h.rename(columns=ren)})
    assert j is not None and len(j) == len(g)
    for c in ("p1_g", "p2_g", "p1_h", "p2_h"):
        assert c in j.columns
    # Only one context available -> no cross-context frame at all.
    assert m.joined_pairs({"GCH": g.rename(columns=ren)}) is None


def test_end_table_carries_both_contexts_sorted_by_position(tmp_path):
    m = _mod()
    df = pd.DataFrame({"pos1": [300, 100], "pos2": [400, 200],
                       "p1_g": [0.1, 0.2], "p2_g": [0.3, 0.4],
                       "p1_h": [0.5, 0.6], "p2_h": [0.7, 0.8]})
    pos, vals, pi = m.end_table(df, {"g": ("p1_g", "p2_g"),
                                     "h": ("p1_h", "p2_h")})
    assert list(pos) == [100, 200, 300, 400], "sorted for binary search"
    assert set(vals) == {"g", "h"} and len(vals["g"]) == 4
    # The two ends of one pair keep the same pair index, so a null can reject
    # re-creating a real ligation.
    assert pi[list(pos).index(100)] == pi[list(pos).index(200)]


# ------------------------------------------------------------------ options --

def test_group_split_emits_a_table_per_group(tmp_path):
    _, out = _run(tmp_path, ["--group-column", "group"])
    for label in ("all", "A", "B"):
        assert (out / f"end_posterior_dist_{label}.tsv").exists(), label
        assert (out / f"end_posterior_dist_{label}.png").exists(), label


def test_pooled_null_with_batch_matching_runs(tmp_path):
    res, _ = _run(tmp_path, ["--null-scope", "pooled",
                             "--null-match-column", "exp_batch,plate_batch"])
    ok = res[res["gg_null_n"] > 0]
    assert len(ok) > 0
    assert abs(ok["gg_null_r"].median()) < 0.15


def test_locus_matched_null_runs(tmp_path):
    res, _ = _run(tmp_path, ["--null-type", "locus-matched",
                             "--null-window", "50000"])
    assert (res["gg_null_n"] > 0).any()


def test_unknown_match_column_fails_loudly(tmp_path):
    m = _mod()
    sp = _fixture(tmp_path)
    with pytest.raises(SystemExit, match="--null-match-column"):
        m.main(["--posteriors", f"GCH={tmp_path / 'GCH'}",
                "--posteriors", f"HCG={tmp_path / 'HCG'}",
                "--samples", sp, "--null-match-column", "nope",
                "--output-dir", str(tmp_path / "o2")])


def test_posteriors_spec_must_be_ctx_equals_dir(tmp_path):
    m = _mod()
    sp = _fixture(tmp_path)
    with pytest.raises(SystemExit, match="CTX=DIR"):
        m.main(["--posteriors", "justapath", "--samples", sp,
                "--output-dir", str(tmp_path / "o3")])


# ------------------------------------------------------- shared stats module --

def test_pearson_helpers_handle_continuous_and_binary_alike():
    from nome_hic_concordance import distance_stats as ds
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    y = 0.8 * x + rng.normal(0, 0.4, 500)
    acc = ds.new_acc(1)
    ds.accumulate(acc, np.zeros(500, int), x, y)
    r, n = ds.corr_from_acc(acc, min_pairs=10)
    assert n[0] == 500
    assert r[0] == pytest.approx(np.corrcoef(x, y)[0, 1], abs=1e-9)


def test_constant_bin_is_masked_not_reported_as_zero():
    from nome_hic_concordance import distance_stats as ds
    acc = ds.new_acc(1)
    ds.accumulate(acc, np.zeros(50, int), np.ones(50), np.arange(50.0))
    r, _ = ds.corr_from_acc(acc, min_pairs=10)
    assert np.isnan(r[0]), "no variance means undefined, not uncorrelated"
