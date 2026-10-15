"""end_posterior_distance_correlation.py --haplotypes: each copy against
donors of the same cell.

Synthetic cells, chr1 and chrX, every read pair tagged with its haplotype.
On X the inactive copy (hap1) is methylated low and the active one high;
on chr1 the copies are alike.  Without coordination within a copy the
linked pairs agree only because both ends sit on one copy: against donors
of the same cell and copy that is nothing, against donors of either copy
it is the whole of X's excess.  With coordination planted on X's copies,
X stands above the autosomes copy by copy."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "end_posterior_distance_correlation.py")
L = 20_000_000


def _mod():
    spec = importlib.util.spec_from_file_location("epd_units", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["epd_units"] = m
    spec.loader.exec_module(m)
    return m


def _fixture(root: Path, n_cells=20, per=1200, x_coord=0.0, seed=0):
    rng = np.random.default_rng(seed)
    for d in ("GCH", "HCG", "haps"):
        (root / d).mkdir(parents=True, exist_ok=True)
    cells = [f"GM_{i}" for i in range(n_cells)]
    logit = {("chrX", 1): -1.5, ("chrX", 2): 1.5,
             ("chr1", 1): 0.0, ("chr1", 2): 0.0}
    for c in cells:
        recs, haps = [], []
        for (chrom, hap), mu in logit.items():
            p1 = rng.integers(0, L - 3_000_000, per)
            d = (10 ** rng.uniform(2.5, 6.5, per)).astype(np.int64)
            p2 = p1 + d
            z = (rng.normal(0, 1, per) * x_coord if chrom == "chrX"
                 else np.zeros(per))
            post = [1 / (1 + np.exp(-(mu + z + rng.normal(0, 0.8, per))))
                    for _ in range(2)]
            for i in range(per):
                recs.append((chrom, int(p1[i]), int(p2[i]), 6, 6, int(d[i]),
                             float(post[0][i]), float(post[1][i])))
                haps.append((chrom, int(p1[i]), int(p2[i]), hap))
        t = pd.DataFrame(recs, columns=["chrom", "pos1", "pos2", "n1", "n2",
                                        "distance", "posterior1",
                                        "posterior2"])
        for ctx in ("GCH", "HCG"):
            t.to_parquet(root / ctx / f"{c}.posteriors.parquet")
        pd.DataFrame(haps, columns=["chrom", "pos1", "pos2", "hap"]
                     ).to_parquet(root / "haps" / f"{c}.haplotypes.parquet")
    sp = root / "s.tsv"
    pd.DataFrame({"cell_name": cells, "group": "GM"}).to_csv(
        sp, sep="\t", index=False)
    return sp


def _run(root: Path, out: Path):
    m = _mod()
    sp = _fixture(root, x_coord=_run.x_coord)
    argv = ["--posteriors", f"GCH={root / 'GCH'}",
            "--posteriors", f"HCG={root / 'HCG'}", "--samples", str(sp),
            "--group-column", "group", "--haplotypes", str(root / "haps"),
            "--xi", "GM=1", "--genome", "none", "--chrx-par", "none",
            "--chrx-escape", "none", "--null-type", "distance-matched",
            "--null-scope", "cell", "--min-distance", "300",
            "--max-distance", "3000000", "--bins-per-decade", "4",
            "--unit-bands", "300,30000,3000000", "--min-pairs", "30",
            "--n-bootstrap", "200", "--fisher-z", "none", "--per-cell", "off",
            "--n-threads", "1", "--plot-format", "png", "--output-dir",
            str(out)]
    assert m.main(argv) == 0
    t = pd.read_csv(out / "end_posterior_units_GM.tsv", sep="\t")
    return t[t["measure"] == "gg"].set_index(["comparison", "band_lo"])


def test_without_coordination_x_excess_is_the_copies_differing(tmp_path):
    _run.x_coord = 0.0
    t = _run(tmp_path / "d", tmp_path / "o")
    for b in (300, 30000):
        # against donors of either copy X's pairs agree -- one copy each
        r = t.loc[("chrX both copies", b)]
        assert r["value"] > 0.2 and r["lo"] > 0
        r = t.loc[("copy part: chrX − autosomes", b)]
        assert r["value"] > 0.2 and r["lo"] > 0
        # against donors of their own copy, nothing beyond the autosomes
        for nm in ("chrX same copy", "chrX − autosomes, same copy",
                   "autosomes both copies"):
            assert abs(t.loc[(nm, b), "value"]) < 0.05, nm
    assert (tmp_path / "o" / "end_posterior_units_GM.png").exists()
    # one group per unit, each with its own curves
    assert (tmp_path / "o" / "end_posterior_dist_GM_chrX_Xi.tsv").exists()


def test_coordination_within_x_copies_stands_above_the_autosomes(tmp_path):
    _run.x_coord = 1.0
    t = _run(tmp_path / "d", tmp_path / "o")
    for b in (300, 30000):
        r = t.loc[("chrX − autosomes, same copy", b)]
        assert r["value"] > 0.3 and r["lo"] > 0
        # "same copy" is the copies' mean, not their pooled correlation
        assert abs(t.loc[("chrX same copy", b), "value"] - (
            t.loc[("chrX Xi", b), "value"]
            + t.loc[("chrX Xa", b), "value"]) / 2) < 1e-9
