"""End-to-end multi-group run with the bootstrap on.

Every previous end-to-end test wrote a SINGLE group ('all'), so nothing ever
executed the group loop twice.  That is the only way to catch state leaking
from one iteration into the next -- which is exactly how a `keys` rebind inside
the FDR block survived review: the 'all' figure was written, the name was
clobbered, and the first real group died with KeyError: 'gg', after the entire
parse had completed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = (Path(__file__).resolve().parent.parent / "scripts"
           / "site_distance_correlation.py")
_FIX = Path("/tmp/sdc")


def _samples(tmp_path, n_rep=4):
    """8 cells over 2 groups — enough cells for the cell-level bootstrap."""
    src = _FIX / "s.tsv"
    if not src.exists():
        pytest.skip("synthetic fixture BAM not present")
    rows = [ln.rstrip("\n").split("\t") for ln in src.read_text().splitlines()]
    hdr, body = rows[0], rows[1:]
    out = ["\t".join(hdr)]
    for i in range(n_rep):
        for r in body:
            r = list(r)
            r[0] = f"{r[0]}_r{i}"
            out.append("\t".join(r))
    p = tmp_path / "samples.tsv"
    p.write_text("\n".join(out) + "\n")
    return p


def _run(tmp_path, *extra):
    pytest.importorskip("pysam")
    if not (_FIX / "GCH.bed.gz").exists():
        pytest.skip("synthetic fixture context BEDs not present")
    out_dir = tmp_path / "out"
    cmd = [sys.executable, str(_SCRIPT),
           "--samples", str(_samples(tmp_path)),
           "--context-bed",
           f"GCH:{_FIX / 'GCH.bed.gz'},HCG:{_FIX / 'HCG.bed.gz'}",
           "--output-dir", str(out_dir),
           "--backend", "htslib", "--n-threads", "2",
           "--min-pairs", "500", "--plot-format", "png", *extra]
    return subprocess.run(cmd, capture_output=True, text=True), out_dir


def test_group_loop_runs_past_the_first_group(tmp_path):
    """The regression: 'all' used to succeed and the next group crashed."""
    r, out_dir = _run(tmp_path, "--group-column", "group",
                      "--n-bootstrap", "200")
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "KeyError" not in r.stderr
    for stem in ("all", "A", "B"):
        assert (out_dir / f"site_distance_corr_{stem}.png").exists(), \
            f"missing figure for group {stem}\n{r.stdout}"
        assert (out_dir / f"site_distance_corr_{stem}.tsv").exists()


def test_every_group_reaches_the_summary(tmp_path):
    import json
    r, out_dir = _run(tmp_path, "--group-column", "group",
                      "--n-bootstrap", "200")
    assert r.returncode == 0, r.stderr
    summary = json.loads(
        (out_dir / "site_distance_corr_summary.json").read_text())
    assert set(summary["groups"]) == {"all", "A", "B"}
    for g in summary["groups"].values():
        assert g["n_read_pairs"] > 0


def test_multi_group_without_bootstrap_also_works(tmp_path):
    """--n-bootstrap 0 skips the FDR block entirely — the other branch."""
    r, out_dir = _run(tmp_path, "--group-column", "group", "--n-bootstrap", "0")
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    for stem in ("all", "A", "B"):
        assert (out_dir / f"site_distance_corr_{stem}.png").exists()


def test_fdr_q_column_is_written_per_group(tmp_path):
    r, out_dir = _run(tmp_path, "--group-column", "group",
                      "--n-bootstrap", "200")
    assert r.returncode == 0, r.stderr
    for stem in ("all", "A", "B"):
        head = (out_dir / f"site_distance_corr_{stem}.tsv"
                ).read_text().splitlines()[0].split("\t")
        assert "gg_delta_q" in head and "gg_delta_p" in head, stem
