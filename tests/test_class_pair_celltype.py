"""class_pair_celltype.py: is concordance cell-type specific, pair by pair?

Synthetic per-cell rows of an annotation_class_concordance run: three cell
types, three chromHMM states.  Type B is more concordant everywhere (+1 pp
in every pair, the genome-wide too); type C only between promoters and
enhancers (+4 pp).  The controls carry nothing.  B's difference is no
pair's own -- it goes with the genome-wide value -- C's is promoter x
enhancer's."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "class_pair_celltype",
    Path(__file__).resolve().parents[1] / "scripts" / "class_pair_celltype.py")
cpc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cpc)

STATES = ("promoter", "enhancer", "polycomb")


def make_run(out: Path, per=80, seed=0):
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True)
    pairs = [("all regions", "all regions")] + [
        (a, b) for i, a in enumerate(STATES) for b in STATES[i:]]
    rows, cov = [], []
    for t in "ABC":
        for k in range(per):
            cell = f"{t}{k}"
            glob = 0.02 + (0.01 if t == "B" else 0.0) + rng.normal(0, 0.003)
            cov.append((cell, f"b{k % 2}", 0.5 + rng.normal(0, 0.02)))
            for a, b in pairs:
                spec = 0.04 if t == "C" and {a, b} == {"promoter",
                                                        "enhancer"} else 0.0
                d = glob + spec + rng.normal(0, 0.008)
                ctl = rng.normal(0, 0.008)
                mu = 0.5 + rng.normal(0, 0.02)
                rows.append(("GCH", "chromhmm", cell, t, a, b, "all 1kb-10Mb",
                             int(rng.integers(50, 150)), True, d, ctl,
                             d * 0.8, ctl * 0.8, np.sign(d) * d,
                             np.sign(ctl) * ctl, mu, mu))
    pd.DataFrame(rows, columns=[
        "context", "system", "cell", "cell_type", "anchor", "partner",
        "band", "pairs", "enough_pairs", "delta", "control", "signed_rel",
        "signed_rel_control", "signed_sgn", "signed_sgn_control",
        "mu_anchor", "mu_partner"]).to_csv(out / "percell.tsv.gz", sep="\t",
                                           index=False)
    pd.DataFrame(cov, columns=["cell", "exp_batch", "level_GCH"]).to_csv(
        out / "cell_covariates.tsv", sep="\t", index=False)
    return out


def test_a_pair_of_its_own_and_a_type_concordant_everywhere(tmp_path):
    d = make_run(tmp_path / "run")
    o = tmp_path / "out"
    cpc.main(["--run-dir", str(d), "--batch-columns", "exp_batch",
              "--sets", "regulatory=promoter,enhancer",
              "--sets", "repressive=polycomb", "--n-threads", "2",
              "--plot-format", "png", "--output-dir", str(o)])
    v = pd.read_csv(o / "class_pair_variation.tsv", sep="\t").set_index(
        "pair")
    pe = "promoter × enhancer"
    # every pair differs between types (B everywhere), the controls do not
    assert (v["delta_q"] < 0.05).all()
    assert (v["control_delta_q"] > 0.05).all()
    # beyond the cell's genome-wide value only promoter x enhancer
    sp = v["delta_specific_q"].drop("all regions × all regions")
    assert sp[pe] < 1e-6 and (sp.drop(pe) > 0.05).all()
    m = pd.read_csv(o / "class_pair_type_means.tsv", sep="\t")
    m = m[(m["statistic"] == "delta") & (m["pair"] == pe)].set_index(
        "cell_type")
    assert m.loc["C", "minus_all_regions"] > 0.03
    assert abs(m.loc["B", "minus_all_regions"]) < 0.005
    a = pd.read_csv(o / "class_pair_auc.tsv", sep="\t")
    a = a[a["statistic"].isin(["delta", "level"])].set_index(
        ["group", "features", "classifier"])["auc"]
    for clf in ("lr", "rf"):
        assert a[(pe, "concordance", clf)] > 0.7 > a[(pe, "control", clf)]
        assert a[("regulatory", "concordance", clf)] > 0.7
        assert a[("polycomb × polycomb", "control", clf)] < 0.65
    for ctx_stat in ("GCH_delta", "GCH_signed_rel", "GCH_signed_sgn"):
        assert (o / f"class_pair_celltype_{ctx_stat}.png").exists()
