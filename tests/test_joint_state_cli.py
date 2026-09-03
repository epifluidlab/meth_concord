"""scripts/joint_state_extract.py end to end, and the manifest that gates it.

The manifest checks are worth more than they look: every one of them is a
mistake that otherwise produces a complete run with quietly wrong contents --
cells dropped from figures, per-group outputs overwriting each other, or a
cell type that cannot be told apart from its batch.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pysam = pytest.importorskip("pysam")

from tests.test_joint_state_core import (              # noqa: E402
    REF_NAME, _make_reference, _sites, _write_bam)

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "joint_state_extract.py"


def _mod():
    spec = importlib.util.spec_from_file_location("jse_cli", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["jse_cli"] = m
    spec.loader.exec_module(m)
    return m


def _cpp_or_skip():
    try:
        from nome_hic_concordance import _joint_state_core  # noqa: F401
    except Exception:                                        # pragma: no cover
        pytest.skip("htslib extension not built")


# ------------------------------------------------------------- the manifest ---

from nome_hic_concordance.samples_manifest import (   # noqa: E402
    ManifestError, load_manifest, safe_name)


def _manifest(tmp, rows, header="name\tbam\tgroup\tcolor\tbatch"):
    p = tmp / "s.tsv"
    p.write_text(header + "\n" + "\n".join("\t".join(r) for r in rows) + "\n")
    return p


def _bam(tmp, name="c1"):
    ref = _make_reference(600, seed=1)
    return str(_write_bam(tmp / f"{name}.bam", ref,
                          [(0, 400, "CT", "CT", 0.5, 0.5)], read_len=80))


def test_group_order_must_match_the_manifest_both_ways(tmp_path):
    b = _bam(tmp_path)
    p = _manifest(tmp_path, [["c1", b, "A", "#111111", "b1"],
                             ["c2", b, "B", "#222222", "b2"]])
    with pytest.raises(ManifestError) as e:
        load_manifest(str(p), group_order="A,C")
    msg = str(e.value)
    assert "'C'" in msg, "a group-order entry with no cells must be named"
    assert "'B'" in msg, "a manifest group left out of group-order must be named"


def test_every_problem_is_reported_at_once(tmp_path):
    """Failing on the first would mean one re-run per mistake."""
    p = _manifest(tmp_path, [["c1", "/no/such.bam", "A", "#111", "b1"],
                             ["c1", "/no/such.bam", "B", "#222", "b1"]])
    with pytest.raises(ManifestError) as e:
        load_manifest(str(p), group_order="A,B,Z")
    assert len(e.value.problems) >= 3, e.value.problems


def test_a_group_with_two_colors_is_an_error(tmp_path):
    b = _bam(tmp_path)
    p = _manifest(tmp_path, [["c1", b, "A", "#111111", "b1"],
                             ["c2", b, "A", "#999999", "b2"]])
    with pytest.raises(ManifestError, match="different"):
        load_manifest(str(p))


def test_hex_colors_survive_the_comment_stripper(tmp_path):
    """'#FF9A4C' would truncate the row under pandas' comment='#', blanking
    that field and every column to its right."""
    b = _bam(tmp_path)
    p = _manifest(tmp_path, [["c1", b, "A", "#FF9A4C", "b1"]])
    m = load_manifest(str(p))
    assert m.colors == {"A": "#FF9A4C"}
    assert m.table.loc[0, "batch"] == "b1", "columns right of the colour survived"


def test_duplicate_names_are_refused(tmp_path):
    b = _bam(tmp_path)
    p = _manifest(tmp_path, [["c1", b, "A", "#111111", "b1"],
                             ["c1", b, "A", "#111111", "b2"]])
    with pytest.raises(ManifestError, match="duplicat"):
        load_manifest(str(p))


def test_unindexed_bams_are_refused(tmp_path):
    b = _bam(tmp_path)
    Path(b + ".bai").unlink()
    p = _manifest(tmp_path, [["c1", b, "A", "#111111", "b1"]])
    with pytest.raises(ManifestError, match="index"):
        load_manifest(str(p))


def test_slashes_in_group_names_are_sanitized_for_paths(tmp_path):
    """The real order has three: IT-L2/3, IT-L5/6, PC/EC."""
    b = _bam(tmp_path)
    p = _manifest(tmp_path, [["c1", b, "IT-L2/3", "#111111", "b1"],
                             ["c2", b, "PC/EC", "#222222", "b2"]])
    m = load_manifest(str(p), group_order="IT-L2/3,PC/EC")
    assert m.safe_names == {"IT-L2/3": "IT-L2_3", "PC/EC": "PC_EC"}
    assert m.group_order == ["IT-L2/3", "PC/EC"], "order keeps the real names"
    assert "group_safe_names" in m.metadata()


def test_two_groups_sanitizing_to_one_path_is_an_error(tmp_path):
    """Otherwise the second group's per-group outputs overwrite the first's."""
    b = _bam(tmp_path)
    p = _manifest(tmp_path, [["c1", b, "IT-L2/3", "#111111", "b1"],
                             ["c2", b, "IT-L2_3", "#222222", "b2"]])
    with pytest.raises(ManifestError, match="sanitize"):
        load_manifest(str(p))


def test_a_group_confined_to_one_batch_is_reported(tmp_path, caplog):
    b = _bam(tmp_path)
    p = _manifest(tmp_path, [["c1", b, "A", "#111111", "b1"],
                             ["c2", b, "A", "#111111", "b2"],
                             ["c3", b, "B", "#222222", "b1"]])
    m = load_manifest(str(p))
    assert m.confounded_groups() == ["B"]
    with caplog.at_level("WARNING"):
        m.report()
    assert "SINGLE batch" in caplog.text


def test_safe_name_never_returns_an_empty_path_component():
    assert safe_name("///") == "unnamed"


# --------------------------------------------------------------- end to end ---

@pytest.fixture(scope="module")
def run(tmp_path_factory):
    _cpp_or_skip()
    d = tmp_path_factory.mktemp("cli")
    ref = _make_reference(4000, seed=9)
    import random
    rng = random.Random(2)
    cells = []
    for i in range(3):
        pairs = [(rng.randrange(0, 1500), rng.randrange(2000, 3800),
                  rng.choice(["CT", "GA"]), rng.choice(["CT", "GA"]),
                  rng.choice([0.0, 0.4, 0.9]), rng.choice([0.0, 0.4, 0.9]))
                 for _ in range(120)]
        cells.append(str(_write_bam(d / f"cell{i}.bam", ref, pairs)))

    beds = {}
    for mark in ("GCH", "HCG"):
        pos, base = _sites(ref, mark)
        f = d / f"{mark}.bed"
        with open(f, "w") as fh:
            for p_, b_ in zip(pos.tolist(), base.tolist()):
                fh.write(f"{REF_NAME}\t{p_}\t{p_ + 1}\t.\t0\t"
                         f"{'+' if b_ == ord('C') else '-'}\n")
        # load_context_from_bed reads through tabix, so the BED has to be
        # bgzipped and indexed exactly as the real site files are.
        pysam.tabix_compress(str(f), str(f) + ".gz", force=True)
        pysam.tabix_index(str(f) + ".gz", preset="bed", force=True)
        beds[mark] = str(f) + ".gz"

    man = d / "samples.tsv"
    man.write_text("name\tbam\tgroup\tcolor\tbatch\n" + "".join(
        f"cell{i}\t{cells[i]}\tG{i % 2}\t#{'aabbcc' if i % 2 == 0 else 'ddeeff'}"
        f"\tb{i % 2}\n" for i in range(3)))

    out = d / "out"
    m = _mod()
    rc = m.main([
        "--samples", str(man), "--group-order", "G0,G1",
        "--context-bed", f"GCH:{beds['GCH']},HCG:{beds['HCG']}",
        "-o", str(out), "--threads", "1", "--min-distance", "0",
        "--bin-size", "1000", "--min-obs", "5", "--strata-per-decade", "2",
    ])
    assert rc == 0
    return {"out": out, "mod": m, "ref": ref, "man": str(man),
            "beds": beds, "cells": cells}


def test_counts_are_sharded_by_chromosome(run):
    shards = sorted((run["out"] / "02j_read_end_joint_counts").glob("chrom=*/*.parquet"))
    assert shards, "no shard written"
    assert all(p.parent.name == f"chrom={REF_NAME}" for p in shards)
    assert {p.stem for p in shards} == {"cell0", "cell1", "cell2"}


def test_raw_counts_are_persisted_next_to_every_posterior(run):
    """A.3.2: Part B's error correction is unambiguous with raw counts and
    needs an empirical diagnostic without them."""
    d = pd.read_parquet(run["out"] / "03j_read_end_posteriors")
    for mark in ("gch", "hcg"):
        for col in (f"k_{mark}", f"n_{mark}", f"theta_{mark}",
                    f"var_{mark}", f"mu_{mark}", f"centred_{mark}"):
            assert col in d.columns, f"missing {col}"
    assert (d["k_hch"] <= d["n_hch"]).all()


def test_every_end_has_a_positive_variance(run):
    """The n == 1 trap, on real extracted data rather than a constructed one."""
    d = pd.read_parquet(run["out"] / "03j_read_end_posteriors")
    assert (d["n_hcg"] == 1).sum() > 0, "the fixture must contain n == 1 ends"
    for mark in ("gch", "hcg"):
        assert (d[f"var_{mark}"] > 0).all()
        assert np.isfinite(d[f"var_{mark}"]).all()


def test_centred_values_are_location_only_by_default(run):
    d = pd.read_parquet(run["out"] / "03j_read_end_posteriors")
    for mark in ("gch", "hcg"):
        assert np.allclose(d[f"centred_{mark}"],
                           d[f"theta_{mark}"] - d[f"mu_{mark}"])


def test_both_marks_share_one_grid(run):
    """A.3.3: two baselines on different grids introduce a spurious covariance
    between the centred marks."""
    d = pd.read_parquet(run["out"] / "03j_read_end_posteriors")
    prior = pd.read_parquet(run["out"] / "03j_prior.parquet")
    assert len(prior) == prior["grid_cell"].nunique()
    # Where a cell is unusable, BOTH marks sit on their global prior.
    unusable = set(prior.loc[~prior["usable"], "grid_cell"])
    sub = d[d["grid_cell"].isin(unusable)]
    if len(sub):
        for mark in ("gch", "hcg"):
            assert sub[f"mu_{mark}"].nunique() == 1


def test_conversion_is_marked_not_dropped(run):
    """A.3.5: k_hch/n_hch must survive so the threshold can be revisited
    without re-extraction."""
    d = pd.read_parquet(run["out"] / "03j_read_end_posteriors")
    assert "conversion_pass" in d.columns
    assert "conversion_rate" in d.columns
    meta = json.loads((run["out"] / "_metadata.json").read_text())
    assert meta["min_conversion_applied"].startswith("marked")


def test_metadata_records_the_normalization_choice(run):
    """A.3.4 asks for it explicitly -- it is the interaction most likely to
    break the analysis, and hardest to attribute after the fact."""
    meta = json.loads((run["out"] / "_metadata.json").read_text())
    assert meta["gch_norm"] == "location"
    assert meta["bin_size"] == 1000
    assert meta["manifest"]["group_order"] == ["G0", "G1"]
    assert "hch" in meta and "MD" in meta["hch"]


def test_pairs_share_a_distance_and_a_grid_stratum(run):
    d = pd.read_parquet(run["out"] / "03j_read_end_posteriors")
    g = d.groupby(["cell_id", "pair_idx"])["contact_dist"].nunique()
    assert (g == 1).all(), "the two ends of a contact must agree on its span"


def test_rerunning_the_baseline_alone_reproduces_it(run):
    """--steps baseline must not need a BAM, and must land on the same fit."""
    m, out = run["mod"], run["out"]
    before = pd.read_parquet(out / "03j_prior.parquet")
    rc = m.main([
        "--samples", run["man"], "--group-order", "G0,G1",
        "--context-bed", f"GCH:{run['beds']['GCH']},HCG:{run['beds']['HCG']}",
        "-o", str(out), "--threads", "1",
        "--min-distance", "0", "--bin-size", "1000", "--min-obs", "5",
        "--strata-per-decade", "2", "--steps", "baseline",
    ])
    assert rc == 0
    after = pd.read_parquet(out / "03j_prior.parquet")
    pd.testing.assert_frame_equal(before, after)


def test_thread_count_does_not_change_the_output(run, tmp_path):
    """A.4 requires it, and the pooled prior fit is where it could leak in."""
    m = run["mod"]
    outs = []
    for threads in (1, 3):
        o = tmp_path / f"t{threads}"
        rc = m.main([
            "--samples", run["man"], "--group-order", "G0,G1",
            "--context-bed", f"GCH:{run['beds']['GCH']},HCG:{run['beds']['HCG']}",
            "-o", str(o), "--threads", str(threads),
            "--min-distance", "0", "--bin-size", "1000", "--min-obs", "5",
            "--strata-per-decade", "2",
        ])
        assert rc == 0
        outs.append(pd.read_parquet(o / "03j_prior.parquet"))
    pd.testing.assert_frame_equal(outs[0], outs[1])


def test_a_single_mark_run_is_refused(run, tmp_path):
    m = run["mod"]
    with pytest.raises(SystemExit, match="two marks"):
        m.main(["--samples", run["man"], "--contexts", "GCH",
                "--context-bed", f"GCH:{run['beds']['GCH']}",
                "-o", str(tmp_path / "x"), "--group-order", "G0,G1"])
