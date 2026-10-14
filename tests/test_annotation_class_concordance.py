"""annotation_class_concordance.py: long-range concordance by annotation
class, per cell, against the same links reshuffled.

Synthetic cells: 2 Mb domains alternating A (accessibility 0.6) and B (0.25);
contacts prefer partners in their anchor's compartment, so the pooled
contacts carry A/B; coordination is planted in A-A contacts only, with a
per-cell strength."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "annotation_class_concordance",
    Path(__file__).resolve().parents[1] / "scripts"
    / "annotation_class_concordance.py")
acc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(acc)
xci = importlib.import_module("nome_hic_concordance.xci")

DOM = 2_000_000
CHROMS = {"chr1": 20_000_000, "chr2": 20_000_000}


def is_a(pos):
    return (np.asarray(pos) // DOM) % 2 == 0


def make_data(out, n_types=3, per=25, coord=None, offset=None, n_contacts=2500,
              n_short=1500, seed=0, coord_in="A", levels=(0.6, 0.25)):
    """``coord(t, c)``: the chance an A-A contact's partner copies its anchor
    in cell c of type t.  ``offset(t, c)`` is added to every posterior of
    that cell AFTER the calls are drawn, so it changes nothing but level."""
    rng = np.random.default_rng(seed)
    out = Path(out)
    post = out / "GCH"
    post.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in range(n_types):
        for c in range(per):
            chrom = rng.choice(list(CHROMS), n_contacts)
            a_pos = rng.integers(0, 17_000_000, n_contacts)
            d = np.exp(rng.uniform(np.log(1_000), np.log(3_000_000),
                                   n_contacts)).astype(np.int64)
            b_pos = a_pos + d
            # contacts prefer their own compartment
            move = (is_a(a_pos) != is_a(b_pos)) & (rng.random(n_contacts) < 0.7)
            b_pos[move] += DOM
            level = lambda x: np.where(is_a(x), *levels)  # noqa: E731
            pa = np.where(rng.random(n_contacts) < level(a_pos), 0.9, 0.1)
            pb = np.where(rng.random(n_contacts) < level(b_pos), 0.9, 0.1)
            if coord is not None:
                aa = (is_a(a_pos) & is_a(b_pos) if coord_in == "A"
                      else ~is_a(a_pos) & ~is_a(b_pos))
                copy = aa & (rng.random(n_contacts) < coord(t, c))
                pb[copy] = pa[copy]
            s_chr = rng.choice(list(CHROMS), n_short)
            s_pos = rng.integers(0, 19_000_000, n_short)
            s_end = s_pos + rng.integers(200, 800, n_short)
            sa = np.where(rng.random(n_short) < level(s_pos), 0.9, 0.1)
            sb = np.where(rng.random(n_short) < level(s_end), 0.9, 0.1)
            df = pd.DataFrame({
                "chrom": np.concatenate([chrom, s_chr]),
                "pos1": np.concatenate([a_pos, s_pos]),
                "pos2": np.concatenate([b_pos, s_end]),
                "posterior1": np.concatenate([pa, sa]),
                "posterior2": np.concatenate([pb, sb])})
            df["n1"] = np.int32(3)
            df["n2"] = np.int32(3)
            df["k1"] = np.round(df["posterior1"] * 3).astype(np.int32)
            df["k2"] = np.round(df["posterior2"] * 3).astype(np.int32)
            if offset is not None:
                df[["posterior1", "posterior2"]] += offset(t, c)
            df["distance"] = (df["pos2"] - df["pos1"]).abs()
            df.to_parquet(post / f"t{t}_c{c}.posteriors.parquet")
            rows.append((f"t{t}_c{c}", f"T{t}", f"b{c % 2}"))
    pd.DataFrame(rows, columns=["cell_name", "group", "batch"]).to_csv(
        out / "samples.tsv", sep="\t", index=False)
    # annotations: PMD = B, HMD = A; RT early in A; chromHMM states per domain
    tiles = [(c, s, s + 100_000) for c, n in CHROMS.items()
             for s in range(0, n, 100_000)]
    t_ = pd.DataFrame(tiles, columns=["chrom", "start", "end"])
    a_ = is_a(t_["start"])
    pd.DataFrame({"c": t_["chrom"], "s": t_["start"], "e": t_["end"],
                  "score": 0.1, "cls": np.where(a_, "HMD", "PMD"),
                  "lab": "neither"}).to_csv(out / "pmd.bed", sep="\t",
                                            header=False, index=False)
    for name, hi_in_a in (("early", True), ("late", False)):
        pd.DataFrame({"c": t_["chrom"], "s": t_["start"], "e": t_["end"],
                      "v": np.where(a_ == hi_in_a, 2.0, 0.5)}).to_csv(
            out / f"{name}.bedGraph", sep="\t", header=False, index=False)
    seg = [(c, s, s + 50_000) for c, n in CHROMS.items()
           for s in range(0, n, 50_000)]
    sg = pd.DataFrame(seg, columns=["c", "s", "e"])
    alt = (sg["s"] // 50_000) % 2 == 0
    sg["state"] = np.where(is_a(sg["s"]),
                           np.where(alt, "1_mTSS1", "2_mEnhA1"),
                           np.where(alt, "3_mHET1", "4_mQuies1"))
    sg.to_csv(out / "chromhmm.bed", sep="\t", header=False, index=False)
    # "open sites" every 20 kb
    st = [(c, s, s + 1) for c, n in CHROMS.items()
          for s in range(10_000, n, 20_000)]
    pd.DataFrame(st).to_csv(out / "open_sites.bed", sep="\t", header=False,
                            index=False)
    pd.DataFrame({"state": ["1", "2", "3", "4"],
                  "group": ["TSS", "enhancers", "HET", "quiescent"],
                  "mneumonics": ["mTSS1", "mEnhA1", "mHET1", "mQuies1"]}
                 ).to_csv(out / "groups.tsv", sep="\t", index=False)
    return out


def run(d, systems="ab,rt,chromhmm,pmd", extra=()):
    o = Path(d) / "out"
    acc.main(["--posteriors", f"GCH={Path(d) / 'GCH'}",
              "--samples", str(Path(d) / "samples.tsv"),
              "--systems", systems,
              "--rt-early", str(Path(d) / "early.bedGraph"),
              "--rt-late", str(Path(d) / "late.bedGraph"),
              "--rt-window", "100000",
              "--chromhmm", str(Path(d) / "chromhmm.bed"),
              "--chromhmm-groups", str(Path(d) / "groups.tsv"),
              "--pmd", str(Path(d) / "pmd.bed"),
              "--ab-resolution", "500000", "--ab-min-contacts", "1000",
              "--min-pairs", "10", "--n-boot", "100", "--n-threads", "2",
              "--batch-column", "batch", "--n-locus-perm", "10",
              "--open-sites", str(Path(d) / "open_sites.bed"),
              "--celltype-auc-trees", "40",
              "--plot-format", "png", "--output-dir", str(o), *extra])
    return (pd.read_csv(o / "summary.tsv", sep="\t"),
            pd.read_csv(o / "percell.tsv.gz", sep="\t"), o)


def _row(s, system, k1, k2, band="all"):
    r = s[(s["system"] == system) & (s["anchor"] == k1)
          & (s["partner"] == k2) & s["band"].str.startswith(band)]
    assert len(r) == 1, (system, k1, k2, band)
    return r.iloc[0]


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    d = make_data(tmp_path_factory.mktemp("planted"),
                  coord=lambda t, c: 0.1 + 0.8 * (c % 5) / 4)
    return run(d)


# -------------------------------------------------------------- behaviour --


def test_planted_coordination_shows_in_its_class_pair_only(planted):
    s, _, _ = planted
    for system, inside, outside in (("pmd", ("HMD", "HMD"), ("PMD", "PMD")),
                                    ("rt", ("early", "early"),
                                     ("late", "late")),
                                    ("ab", ("A", "A"), ("B", "B"))):
        r = _row(s, system, *inside)
        assert r["delta_lo"] > 0 and r["delta"] > 0.05, system
        o = _row(s, system, *outside)
        assert o["delta_lo"] < 0 < o["delta_hi"] or abs(o["delta"]) < 0.01
        for k in (inside, outside):            # the control is centred
            c = _row(s, system, *k)
            assert c["control_lo"] < 0 < c["control_hi"] or abs(
                c["control"]) < 0.01


def test_cell_to_cell_variation_is_found_where_planted(planted):
    s, _, _ = planted
    r = _row(s, "pmd", "HMD", "HMD")
    assert r["excess_sd"] > 0.05
    assert r["split_half_r"] > 0.3 and abs(r["control_split_half_r"]) < 0.4
    o = _row(s, "pmd", "PMD", "PMD")
    assert o["excess_sd"] < 0.5 * r["excess_sd"]


def test_chromhmm_groups_and_the_figure(planted):
    s, pc, o = planted
    chm = s[s["system"] == "chromhmm"]
    assert set(chm["anchor"]) <= {"promoter", "enhancer", "heterochromatin",
                                  "quiescent"}
    r = _row(s, "chromhmm", "promoter", "enhancer")
    assert r["delta"] > 0
    assert (o / "fig_GCH_pmd.png").exists()
    assert (o / "main_figure.png").exists()
    for tag in ("rel", "sign", "even"):
        assert (o / f"fig_GCH_pmd_signed_{tag}.png").exists()
        assert (o / f"main_figure_signed_{tag}.png").exists()
    assert set(pc.columns) >= {"cell", "cell_type", "delta", "control",
                               "mu_anchor", "enough_pairs"}


def test_ab_is_called_per_cell_type_and_oriented(planted):
    _, _, o = planted
    tab = pd.read_csv(o / "compartments_ab.tsv", sep="\t")
    for t, g in tab[tab["ab"].notna() & (tab["ab"] != "")].groupby(
            "cell_type"):
        mid = g["start"] + 250_000
        agree = np.mean((g["ab"] == "A") == is_a(mid))
        assert agree > 0.9, t
    qc = pd.read_csv(o / "compartments_qc.tsv", sep="\t")
    assert (qc["r_with_reference"] > 0.5).all()


def test_no_coordination_gives_nothing_to_find(tmp_path):
    s, _, _ = run(make_data(tmp_path, per=20, seed=3), systems="pmd")
    for k in (("HMD", "HMD"), ("PMD", "PMD")):
        r = _row(s, "pmd", *k)
        assert abs(r["delta"]) < 0.02
        assert r["excess_sd"] < 0.6 * r["cell_sd"]


def test_a_cells_own_level_does_not_count_as_concordance(tmp_path):
    """Centred on its own marginal, a cell shifted up or down in every
    posterior scores exactly as before -- real and reshuffled alike."""
    base = run(make_data(tmp_path / "a", per=10, seed=5,
                         coord=lambda t, c: 0.5), systems="pmd")[1]
    shifted = run(make_data(tmp_path / "b", per=10, seed=5,
                            coord=lambda t, c: 0.5,
                            offset=lambda t, c: 0.03 * t - 0.02 * (c % 3)),
                  systems="pmd")[1]
    key = ["cell", "anchor", "partner", "band"]
    m = base.merge(shifted, on=key, suffixes=("", "_s"))
    assert len(m) == len(base) > 0
    for col in ("C_obs", "delta", "control"):
        assert np.allclose(m[col], m[col + "_s"], atol=1e-5), col


# ----------------------------------------------------------------- pieces --


def test_pmd_reads_column_5_or_the_common_labels(tmp_path):
    p = tmp_path / "pmd.bed"
    pd.DataFrame([("chr1", 0, 100, 0.1, "PMD", "neither"),
                  ("chr1", 100, 200, 0.9, "HMD", "commonHMD"),
                  ("chr1", 200, 300, 0.1, "PMD", "commonPMD"),
                  ("chr1", 300, 400, 0.5, "NA", "neither")]).to_csv(
        p, sep="\t", header=False, index=False)
    iv = acc.read_pmd(p)
    assert iv.lookup(["chr1"] * 4, [50, 150, 250, 350]).tolist() == [1, 0, 1,
                                                                     -1]
    common = acc.read_pmd(p, common_only=True)
    assert common.lookup(["1"] * 3, [50, 150, 250]).tolist() == [-1, 0, 1]


def test_intervals_lookup_edges_and_chrom_names():
    iv = acc.Intervals("x", ("a", "b"), ["chr1", "chr1", "2"], [0, 100, 0],
                       [100, 200, 50], [0, 1, 1])
    got = iv.lookup(["1", "chr1", "chr1", "chr2", "chr3"],
                    [99, 100, 200, 10, 10])
    assert got.tolist() == [0, 1, -1, 1, -1]


# -------------------------------------------------------------- symmetry --


def test_both_ends_are_reshuffled_from_their_own_side_and_class():
    """Two cells, one contact each, each the other's only donor: the right
    end's donor lies to the right in the right end's class and bin, the
    left end's to the left in the left end's -- mirrored, not borrowed."""
    S = acc._ls.GENOME_STRIDE
    g_lo = np.array([1_000_000, 1_010_000], np.int64) + S
    g_hi = np.array([1_050_000, 1_060_000], np.int64) + S
    ok, right, left, _ = acc.reshuffle(
        g_lo, g_hi, g_hi - g_lo, np.array([0, 1]), np.array([0, 0]),
        np.array([1, 1]), np.array([0.4, 0.3]), np.array([-0.1, -0.2]),
        np.array(["s", "s"]), 2, acc._ls.log_dbins(1000, 10_000_000, 4),
        250_000, 2, seed=0)
    assert ok.all()
    assert right[:, 0].tolist() == [-0.2, -0.2]   # cell 1's right end
    assert left[:, 0].tolist() == [0.3, 0.3]      # cell 1's left end
    assert right[:, 1].tolist() == [-0.1, -0.1]
    assert left[:, 1].tolist() == [0.4, 0.4]


def test_pairs_are_unordered_unless_asked(planted, tmp_path):
    s, _, _ = planted
    pmd = s[s["system"] == "pmd"]
    pairs = set(zip(pmd["anchor"], pmd["partner"]))
    assert ("HMD", "PMD") in pairs and ("PMD", "HMD") not in pairs
    d = make_data(tmp_path, per=8, seed=2)
    s2, _, _ = run(d, systems="pmd", extra=("--ordered-pairs",))
    pairs = set(zip(s2["anchor"], s2["partner"]))
    assert ("HMD", "PMD") in pairs and ("PMD", "HMD") in pairs
    assert s2["ordered"].all() and not s["ordered"].any()


def test_a_plain_bed_is_inside_versus_outside(tmp_path):
    p = tmp_path / "lad.bed"
    pd.DataFrame([("chr1", 100, 200), ("chr1", 150, 300),
                  ("chr1", 500, 600)]).to_csv(p, sep="\t", header=False,
                                               index=False)
    iv = acc.read_annotation("LAD", p)
    assert iv.classes == ("LAD", "non-LAD")
    got = iv.lookup(["chr1"] * 5 + ["chr2"], [50, 120, 250, 400, 550, 10])
    assert got.tolist() == [1, 0, 0, 1, 0, -1]
    q = tmp_path / "lab.bed"
    pd.DataFrame([("chr1", 0, 100, "x"), ("chr1", 100, 200, "y")]).to_csv(
        q, sep="\t", header=False, index=False)
    lab = acc.read_annotation("st", q)
    assert lab.classes == ("x", "y")
    assert lab.lookup(["chr1"] * 3, [50, 150, 250]).tolist() == [0, 1, -1]


# ---------------------------------------------------------------- signed --


def test_signed_concordance_is_positive_when_open_and_negative_when_closed(
        planted, tmp_path):
    """Coordination in mostly-open A (level 0.6) is co-open: S > 0.  The
    same in mostly-closed B (level 0.25) is co-closed: S < 0.  The control,
    the same formula on reshuffled links alone, is centred either way."""
    s, pc, _ = planted
    r = _row(s, "pmd", "HMD", "HMD")
    assert r["F"] > 0.5 and r["signed_lo"] > 0
    assert r["signed_control_lo"] < 0 < r["signed_control_hi"] or abs(
        r["signed_control"]) < 0.005
    closed, _, _ = run(make_data(tmp_path, per=20, seed=4, coord_in="B",
                                 coord=lambda t, c: 0.6), systems="pmd")
    r = _row(closed, "pmd", "PMD", "PMD")
    assert r["F"] < 0.5 and r["signed_hi"] < 0
    assert abs(r["signed"]) > 5 * abs(r["signed_control"])
    assert {"F", "F_null", "signed", "signed_control"} <= set(pc.columns)


def test_against_the_genome_a_rare_open_state_still_reads_co_open(tmp_path):
    """GCH-like: open in 15% of A and 3% of B, so every class is mostly
    closed and 2F - 1 calls coordination co-closed wherever it is.  Read
    against the cell's genome-wide co-open share, coordination in the
    (relatively) open A is co-open and in the closed B co-closed."""
    gch = dict(per=20, n_contacts=4000, levels=(0.15, 0.03),
               coord=lambda t, c: 0.6)
    s, _, o = run(make_data(tmp_path / "a", seed=6, coord_in="A", **gch),
                  systems="pmd")
    r = _row(s, "pmd", "HMD", "HMD")
    assert r["signed_hi"] < 0                     # absolute: "co-closed"
    assert r["signed_rel_lo"] > 0                 # against the genome: co-open
    assert r["F_ref"] < r["F"] < 0.5
    assert (o / "fig_GCH_pmd_signed_rel.png").exists()
    s, _, _ = run(make_data(tmp_path / "b", seed=7, coord_in="B", **gch),
                  systems="pmd")
    r = _row(s, "pmd", "PMD", "PMD")
    assert r["signed_rel_hi"] < 0                 # closed B: co-closed
    for k in (("HMD", "HMD"), ("PMD", "PMD")):    # the control is centred
        c = _row(s, "pmd", *k)
        assert (c["signed_rel_control_lo"] < 0 < c["signed_rel_control_hi"]
                or abs(c["signed_rel_control"]) < 0.002)


def test_the_weight_is_two_f_minus_one_against_an_even_split():
    F = np.array([0.1, 0.5, 0.8])
    assert np.allclose(acc._weight(F, 0.5), 2 * F - 1)
    assert acc._weight(0.2, 0.2) == 0
    assert acc._weight(0.3, 0.1) > 0 > acc._weight(0.05, 0.1)


def test_the_sign_version_keeps_delta_c_whole(planted):
    """ΔC x sign(w): |ΔC| unchanged, the direction that of ΔC x w."""
    s, pc, _ = planted
    r = s[np.isfinite(s["signed_rel"]) & (s["signed_rel"] != 0)]
    assert np.allclose(np.abs(r["signed_sgn"]), np.abs(r["delta"]))
    assert (np.sign(r["signed_sgn"]) == np.sign(r["signed_rel"])).all()
    # (a class pair's rows; all regions mixes the pairs' signs)
    c = pc[np.isfinite(pc["signed_rel"]) & (pc["signed_rel"] != 0)
           & (pc["anchor"] != acc.ALL_REGIONS)]
    assert np.allclose(np.abs(c["signed_sgn"]), np.abs(c["delta"]))
    h = _row(s, "pmd", "HMD", "HMD")
    assert h["signed_sgn_lo"] > 0 and h["signed_sgn"] >= h["signed_rel"]


def test_a_donor_stays_in_its_partners_bin_on_every_chromosome():
    """The 250 kb position bin is counted from each chromosome's start: on
    chromosome 7 a donor 1 kb past the partner's bin edge is refused."""
    S = acc._ls.GENOME_STRIDE
    c = 7 * S
    g_lo = np.array([1_200_000, 1_190_000], np.int64) + c
    g_hi = np.array([1_249_000, 1_251_500], np.int64) + c   # bins 4 and 5
    ok, right, _, _ = acc.reshuffle(
        g_lo, g_hi, g_hi - g_lo, np.array([0, 1]), np.array([0, 0]),
        np.array([0, 0]), np.array([0.1, 0.2]), np.array([0.3, 0.4]),
        np.array([0, 0]), 1, acc._ls.log_dbins(1000, 10_000_000, 4),
        250_000, 0, seed=0, names=["s"])
    # each right end's only other-cell donor sits in the other bin
    assert not ok.any() and np.isnan(right).all()


def test_window_means_match_a_per_base_average(tmp_path):
    """The running-integral window mean equals averaging the covered bases
    one by one -- from a bedGraph and from the same data as a bigWig."""
    rng = np.random.default_rng(0)
    edges = np.sort(rng.choice(np.arange(1, 9_999), 60, replace=False))
    st, en = edges[0::2], edges[1::2]                  # gaps between them
    val = rng.uniform(0, 5, st.size)
    bg = tmp_path / "t.bedGraph"
    pd.DataFrame({"c": "chr1", "s": st, "e": en, "v": val}).to_csv(
        bg, sep="\t", header=False, index=False)
    w0 = np.array([0, 500, 2_000, 9_000, 9_990])
    w1 = np.array([1_000, 2_500, 2_001, 10_000, 9_995])
    base = np.full(10_000, np.nan)
    for s_, e_, v_ in zip(st, en, val):
        base[s_:e_] = v_
    want = np.array([np.nanmean(base[a:b]) if np.isfinite(base[a:b]).any()
                     else np.nan for a, b in zip(w0, w1)])
    code = acc._chrom_code("chr1")
    got = acc._window_means(acc._track(bg), code, w0, w1)
    assert np.allclose(got, want, equal_nan=True)
    pyBigWig = pytest.importorskip("pyBigWig")
    bw_path = str(tmp_path / "t.bw")
    bw = pyBigWig.open(bw_path, "w")
    bw.addHeader([("chr1", 10_000)])
    bw.addEntries(["chr1"] * st.size, st.tolist(), ends=en.tolist(),
                  values=val.tolist())
    bw.close()
    got_bw = acc._window_means(acc._track(bw_path), code, w0, w1)
    assert np.allclose(got_bw, want, equal_nan=True, rtol=1e-5)


# ------------------------------------------------------ cell covariates --


def test_cell_cycle_covariates_from_a_cells_read_pairs(tmp_path):
    rows = ([("chr1", 0, 10_000)] * 4                 # < 25 kb: not counted
            + [("chr1", 0, 100_000)] * 6              # near
            + [("chr1", 0, 5_000_000)] * 3            # mitotic band
            + [("chr1", 0, 20_000_000)])              # far
    df = pd.DataFrame(rows, columns=["chrom", "pos1", "pos2"])
    df.to_parquet(tmp_path / "c.parquet")
    rt = acc.Intervals("rt", ("early", "late"), ["chr1", "chr1"],
                       [0, 1_000_000], [1_000_000, 30_000_000], [0, 1])
    _, cov = acc._contact_pass(tmp_path / "c.parquet", 1000, 250_000, set(),
                               rt)
    assert cov["contacts_25kb"] == 10
    assert cov["frac_near"] == pytest.approx(0.6)
    assert cov["frac_mitotic"] == pytest.approx(0.3)
    # 14 left ends at 0 (early), 4 + 6 right ends early, 4 late
    assert cov["repli_score"] == pytest.approx(24 / 28)


def test_a_covariate_that_drives_concordance_explains_its_variation():
    """Within each type ΔC follows % mitotic plus sampling noise; the
    control is the noise alone.  % mitotic is found, nothing else is, and
    most of the REAL within-type variation is explained."""
    rng = np.random.default_rng(0)
    n = 300
    cells = [f"c{i}" for i in range(n)]
    types = np.repeat(["T0", "T1", "T2"], n // 3)
    cov = pd.DataFrame({"cell": cells, "cell_type": types,
                        "log10_contacts": rng.normal(4, 0.2, n),
                        "frac_near": rng.uniform(0.4, 0.6, n),
                        "frac_mitotic": rng.uniform(0.05, 0.2, n),
                        "repli_score": rng.uniform(0.45, 0.55, n)})
    signal = 0.4 * (cov["frac_mitotic"] - 0.125)
    pc = pd.DataFrame({"context": "GCH", "system": "ab", "cell": cells,
                       "cell_type": types, "anchor": "A", "partner": "A",
                       "band": "all 1kb-10Mb", "enough_pairs": True,
                       "delta": signal + rng.normal(0, 0.005, n)
                       + np.where(types == "T1", 0.01, 0),
                       "control": rng.normal(0, 0.005, n)})
    ctx_cov = {"GCH": pd.DataFrame({"cell": cells,
                                    "level": rng.uniform(0.2, 0.3, n)})}
    a = acc.covariate_association(pc, cov, ctx_cov, stats=("delta",))
    r = a.iloc[0]
    assert r["rho_frac_mitotic"] > 0.6 and r["q_frac_mitotic"] < 1e-6
    for c in ("log10_contacts", "frac_near", "repli_score", "level"):
        assert abs(r[f"rho_{c}"]) < 0.2
    assert r["explained_share"] > 0.7
    assert abs(r["control_R2_adj"]) < 0.05


def test_the_covariate_outputs_are_written(planted):
    _, _, o = planted
    cov = pd.read_csv(o / "cell_covariates.tsv", sep="\t")
    assert {"frac_near", "frac_mitotic", "repli_score", "log10_contacts",
            "level_GCH", "frac_short", "sites_mean_GCH", "decisive_GCH",
            "frac_called_GCH", "compartment_strength", "batch"} <= set(
                cov.columns)
    assert cov["compartment_strength"].notna().any()
    assert cov["repli_score"].between(0, 1).all()
    a = pd.read_csv(o / "covariate_association.tsv", sep="\t")
    assert {"rho_frac_mitotic", "q_repli_score", "explained_share",
            "bio_share", "control_R2_adj"} <= set(a.columns)
    assert {"batch_batch_p", "R2_adj_all", "R2_adj_batch"} <= set(a.columns)
    d = a[a["statistic"] == "delta"]
    assert {"cell_share", "locus_share", "noise_share", "perm_p",
            "explained_cell_share"} <= set(d.columns)
    for st in ("delta", "delta_r", "signed_rel", "signed_sgn"):
        assert a.loc[a["statistic"] == st, "cell_share"].notna().any(), st
    lr = pd.read_csv(o / "locus_reassignment.tsv", sep="\t")
    assert {"delta", "delta_r", "signed_rel", "signed_sgn"} <= set(
        lr["statistic"])
    assert (o / "covariates_GCH_delta_r.png").exists()
    # the bar's components: Shapley shares adding up to the explained part
    same = a[(a["anchor"] == a["partner"]) & a["R2_adj_all"].notna()
             & a["band"].astype(str).str.startswith("all")]
    sh = [c for c in a.columns if c.startswith("shap_")]
    assert {"shap_technical", "shap_1D baseline", "shap_scale",
            "shap_3D contacts", "shap_labeling efficiency",
            "shap_substructure", "shap_batch"} <= set(sh)
    assert {"rho_ps_slope", "rho_pair_share", "rho_mu_pair_1"} <= set(
        a.columns)
    assert cov["ps_slope"].between(-3, 0.5).all()
    tot = same[sh].sum(1, min_count=1)
    np.testing.assert_allclose(tot, same["R2_adj_all"].clip(lower=0),
                               atol=1e-6)
    assert a.loc[a["anchor"] != a["partner"], sh].isna().all().all()
    assert {"open_site_level", "sub_pc1", "sub_pc5"} <= set(cov.columns)
    # over all cells: noise + loci + every group (cell type one) + the rest
    tot = a[(a["anchor"] == a["partner"]) & a["tot_noise"].notna()]
    parts = ["tot_noise", "tot_loci", "tot_residual"] + [
        c for c in a.columns if c.startswith("tot_") and c not in (
            "tot_noise", "tot_loci", "tot_residual")]
    assert "tot_cell type" in parts and "residual_q" in a
    s_ = tot[parts].sum(1)
    assert ((s_ > 0.97) | (tot["residual_var"] < 0)).all()
    assert (o / "variance_total_GCH_delta.png").exists()
    allb = a[a["band"].astype(str).str.startswith("all")
             & (a["anchor"] == a["partner"]) & (a["statistic"] == "delta")]
    assert allb["R2_cv_nonlinear"].notna().any()
    assert (allb["tot_residual_nonlinear"] <= allb["tot_residual"] + 1e-9
            ).all()
    cc = pd.read_csv(o / "cross_class_correlation.tsv", sep="\t")
    assert {"delta", "signed_rel"} <= set(cc["statistic"])
    assert cc["r_resid"].notna().any()
    ch = pd.read_csv(o / "cell_chromosome.tsv", sep="\t")
    assert {"cell_x_chromosome_share", "perm_p"} <= set(ch.columns)
    assert cov["open_site_level"].between(0.2, 0.7).all()
    assert (o / "cell_covariates_extra.png").exists()
    assert (o / "cross_class_correlation.tsv").exists()
    assert (o / "cross_class_GCH.png").exists()
    assert (o / "covariates_GCH_delta.png").exists()
    assert (o / "covariates_GCH_delta_by_band.png").exists()
    assert (o / "cell_cycle_covariates.png").exists()
    assert a["band"].astype(str).str.startswith("all").any()
    assert (~a["band"].astype(str).str.startswith("all")).any()


# ------------------------------------------------- what cells, what loci --


def _loci_and_cells(cell_effect, seed=0):
    """Contacts of 3 types x 100 cells, 150 each, at loci whose concordance
    differs a lot (sd 0.6, against noise sd 0.5) -- plus, optionally, a
    per-cell offset."""
    rng = np.random.default_rng(seed)
    n_cells, per = 300, 150
    cell = np.repeat(np.arange(n_cells), per)
    ctype = np.repeat(np.arange(3), n_cells // 3)
    locus_effect = rng.normal(0, 0.6, 5000)[rng.integers(0, 5000, cell.size)]
    noise = rng.normal(0, 0.5, cell.size)
    d = locus_effect + noise + cell_effect * rng.normal(0, 1, n_cells)[cell]
    c = rng.normal(0, 0.5, cell.size)
    z = np.zeros(cell.size, np.int64)
    return acc.locus_reassignment(cell, ctype, z, z, {"d": d, "c": c}, 1, 10,
                                  40, np.random.default_rng(seed + 1))


def test_locus_sampling_alone_is_not_a_cell_difference():
    r = _loci_and_cells(0.0).iloc[-1]                 # the 'all' band
    assert r["locus_share"] > 0.3                     # loci differ ...
    assert r["cell_share"] < 0.1 and r["perm_p"] > 0.05   # ... cells don't


def test_a_real_cell_difference_survives_reassignment():
    r = _loci_and_cells(0.15).iloc[-1]
    assert r["cell_share"] > 0.3 and r["perm_p"] < 0.05


def _signed_parts(cell_effect, seed=0):
    """As _loci_and_cells, with the parts of every statistic: the
    correlation scale, and a co-open share well above the cells'
    genome-wide one, so that ΔC x w carries ΔC's variation."""
    rng = np.random.default_rng(seed)
    n_cells, per = 300, 150
    cell = np.repeat(np.arange(n_cells), per)
    ctype = np.repeat(np.arange(3), n_cells // 3)
    locus = rng.normal(0, 0.6, 5000)[rng.integers(0, 5000, cell.size)]
    d = (locus + rng.normal(0, 0.5, cell.size)
         + cell_effect * rng.normal(0, 1, n_cells)[cell])
    c = rng.normal(0, 0.5, cell.size)
    s12 = rng.uniform(0.15, 0.25, n_cells)[cell]
    parts = {"d": d, "c": c, "r": d / 2 / s12, "rc": c / 2 / s12,
             "oo": rng.uniform(0.6, 0.8, cell.size),
             "oc": rng.uniform(0.1, 0.3, cell.size),
             "no2": rng.uniform(0.3, 0.5, cell.size),
             "nc2": rng.uniform(0.3, 0.5, cell.size)}
    z = np.zeros(cell.size, np.int64)
    fg = rng.uniform(0.45, 0.55, n_cells)
    return acc.locus_reassignment(cell, ctype, z, z, parts, 1, 10, 30,
                                  np.random.default_rng(seed + 1), fg=fg,
                                  fgn=fg)


def test_the_locus_split_covers_every_statistic():
    for eff, cell_hi in ((0.0, False), (0.15, True)):
        lr = _signed_parts(eff)
        a = lr[lr["band_index"] == 1].set_index("statistic")
        assert set(a.index) == {"delta", "delta_r", "signed_rel",
                                "signed_sgn"}
        for st in a.index:
            shares = a.loc[st, ["noise_share", "locus_share", "cell_share"]]
            assert shares.sum() == pytest.approx(1, abs=0.05)
            if cell_hi:
                assert a.loc[st, "cell_share"] > 0.25, st
                assert a.loc[st, "perm_p"] < 0.05, st
            else:
                assert a.loc[st, "cell_share"] < 0.12, st
                assert a.loc[st, "locus_share"] > 0.3, st


def test_compartment_strength_is_excess_same_compartment_pairing():
    res = 250_000
    maps = {"__all__": acc.Intervals("ab", ("A", "B"), ["chr1"] * 40,
                                     np.arange(40) * res,
                                     (np.arange(40) + 1) * res,
                                     np.arange(40) // 4 % 2)}   # 1 Mb domains
    rng = np.random.default_rng(0)
    b1 = rng.integers(0, 20, 400).astype(np.int32)
    same = (b1 + 8).astype(np.int32)                  # 2 Mb on: same type
    rand = (b1 + rng.integers(4, 20, 400)).astype(np.int32)
    code = np.full(400, acc._chrom_code("chr1"), np.int64)
    cells = [(code, b1, same, {}), (code, b1, rand, {})]
    s_ = acc.compartment_strength(cells, ["x", "x"], maps, res, 1_000_000)
    assert s_[0] > 0.4 and abs(s_[1]) < 0.1


def test_cross_class_correlation_finds_one_per_cell_factor():
    rng = np.random.default_rng(0)
    n = 200
    cells = [f"c{i}" for i in range(n)]
    types = np.repeat(["T0", "T1"], n // 2)
    shared = rng.normal(0, 1, n)
    rows = []
    for st, k in (("A", 1.0), ("B", 1.0), ("X", 0.0)):
        own = rng.normal(0, 1, n) if k == 0 else 0
        val = k * shared + own + rng.normal(0, 0.7, n)
        rows.append(pd.DataFrame({"context": "GCH", "system": "s",
                                  "cell": cells, "cell_type": types,
                                  "anchor": st, "partner": st,
                                  "band": "all 1kb-10Mb",
                                  "enough_pairs": True, "delta": val}))
    pc = pd.concat(rows)
    # split-half reliability of a value 1 (or 1) signal : 0.49 noise
    summ = pd.DataFrame({"context": "GCH", "system": "s",
                         "anchor": ["A", "B", "X"], "partner": ["A", "B", "X"],
                         "band": "all 1kb-10Mb",
                         "split_half_r": [0.5, 0.5, 0.5]})
    cc = acc.cross_class_correlation(pc, summ, min_cells=30)
    ab = cc[(cc["state_1"] == "A") & (cc["state_2"] == "B")].iloc[0]
    ax = cc[(cc["state_1"] == "A") & (cc["state_2"] == "X")].iloc[0]
    assert ab["r"] > 0.5 and ab["r_disattenuated"] > ab["r"]
    assert abs(ax["r"]) < 0.2
    # a class too noisy across cells is not divided by its reliability
    summ["split_half_r"] = [0.5, 0.05, 0.5]
    cc = acc.cross_class_correlation(pc, summ, min_cells=30)
    ab = cc[(cc["state_1"] == "A") & (cc["state_2"] == "B")].iloc[0]
    assert np.isnan(ab["r_disattenuated"]) and ab["r"] > 0.5


# ------------------------------------------------- the components, checked --


def test_shapley_shares_split_r2_between_groups():
    rng = np.random.default_rng(0)
    n = 2000
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    x3 = 0.8 * x1 + 0.6 * rng.normal(0, 1, n)      # shares x1's variance
    y = 1.0 * x1 + 0.5 * x2 + rng.normal(0, 1, n)
    Z = np.column_stack([x1, x2, x3]) - [x1.mean(), x2.mean(), x3.mean()]
    y = y - y.mean()
    phi, r2 = acc._shapley_r2(y, Z, {"a": [0], "b": [1], "c": [2],
                                     "none": []})
    assert sum(v for v in phi.values() if np.isfinite(v)) == \
        pytest.approx(r2)
    assert np.isnan(phi["none"])
    assert phi["a"] > phi["c"] > 0.01 and phi["b"] == pytest.approx(
        0.25 / 2.25, abs=0.03)


def test_open_site_level_reads_the_ends_at_open_sites(tmp_path):
    pos1 = np.array([1_000, 5_000, 50_000, 90_000])
    pos2 = pos1 + 20_000
    df = pd.DataFrame({"chrom": "chr1", "pos1": pos1, "pos2": pos2,
                       "n1": np.int32(4), "n2": np.int32(4),
                       "k1": np.int32([4, 3, 0, 1]),
                       "k2": np.int32([0, 0, 0, 0]),
                       "posterior1": 0.5, "posterior2": 0.5})
    f = tmp_path / "c.posteriors.parquet"
    df.to_parquet(f)
    sites = acc._codes_of(["chr1"]) * acc.STRIDE + np.array([1_100, 4_950])
    r = acc._read_cell(f, 2, 1000, 10**7, set(), {}, open_g=np.sort(sites),
                       open_w=200, prof_bin=1_000_000)
    assert r["open"] == (7.0, 8.0)                 # the first two left ends
    keys, sums, counts, hs, hn = r["prof"]
    assert counts.sum() == 8 and sums.sum() == pytest.approx(4.0)
    assert (hn <= counts).all() and (hs <= sums + 1e-6).all()
    assert acc._count_pass(f, set()) == (8.0, 32.0)


def test_substructure_pcs_find_a_split_within_a_type():
    rng = np.random.default_rng(1)
    prof, types = {}, {}
    keys = np.arange(200)
    pattern = rng.normal(0, 1, 200)
    for i in range(80):
        sub = i % 2
        v = 0.5 + 0.05 * (1 if sub else -1) * pattern + rng.normal(0, .02, 200)
        prof[f"c{i}"] = (keys, (v * 10).astype(np.float32),
                         np.full(200, 10, np.int32))
        types[f"c{i}"] = "T"
    sp = acc.substructure_pcs({"GCH": prof}, types, n_pcs=3)
    sub = np.array([int(c[1:]) % 2 for c in sp["cell"]])
    assert abs(np.corrcoef(sp["sub_pc1"], sub)[0, 1]) > 0.95
    assert sp["sub_pc1"].std() == pytest.approx(1, abs=0.02)


def test_hotspots_are_called_where_planted_and_annotated():
    rng = np.random.default_rng(2)
    code = acc._chrom_code("chr1")
    grid = acc.BinGrid({code: "chr1"}, 50_000_000, 1_000_000)
    n = 200_000
    lo = rng.integers(0, 45_000_000, n)
    # short contacts, so that a planted bin's contacts end in it too (a
    # contact counts in both its ends' bins)
    hi = lo + rng.integers(1_000, 20_000, n)
    cell = np.sort(rng.integers(0, 100, n))
    hot = ((lo // 1_000_000) % 10 == 3) & ((hi // 1_000_000) % 10 == 3)
    d = rng.normal(0, 1, n) + 0.3 * hot
    c = rng.normal(0, 1, n)
    hs = acc.locus_hotspots(grid, code * acc.STRIDE + lo,
                            code * acc.STRIDE + hi, cell, d, c,
                            np.zeros(n, np.int64), 1)
    hs.insert(0, "context", "GCH")
    hs = acc.call_hotspots(hs, 1, 0.05, 200)
    t = hs[hs["tested"]]
    called = set(t.loc[t["hot"], "start"] // 1_000_000)
    planted = {3, 13, 23, 33, 43}
    assert called >= planted and len(called - planted) <= 1   # FDR 5%
    assert t["cold"].sum() <= 1
    track = np.zeros(grid.n)
    track[[3, 13, 23, 33, 43]] = 1.0
    en = acc.hotspot_enrichment(hs, {"planted": track})
    r = en.iloc[0]
    assert r["rho"] > 0.3 and r["q_rho"] < 0.05 and r["hot_odds_ratio"] > 5
    assert abs(r["rho_control"]) < 0.3


def test_bin_grid_coverage_splits_intervals_across_bins():
    code = acc._chrom_code("chr2")
    grid = acc.BinGrid([code], 10_000_000, 1_000_000)
    cov = grid.coverage(pd.DataFrame({"chrom": ["chr2", "2"],
                                      "start": [500_000, 2_900_000],
                                      "end": [1_500_000, 3_100_000]}))
    np.testing.assert_allclose(cov[:4], [0.5, 0.5, 0.1, 0.1])


def test_pmd_depth_is_hmd_less_pmd_methylation(tmp_path):
    d = make_data(tmp_path, per=10, n_types=2)
    o = tmp_path / "out_hcg"
    acc.main(["--posteriors", f"HCG={d / 'GCH'}",
              "--samples", str(d / "samples.tsv"), "--systems", "pmd",
              "--pmd", str(d / "pmd.bed"), "--min-pairs", "10",
              "--n-boot", "20", "--n-threads", "2", "--n-locus-perm", "0",
              "--covariate-min-cells", "10", "--no-hotspots",
              "--celltype-auc-stats", "none",
              "--plot-format", "png", "--output-dir", str(o)])
    cov = pd.read_csv(o / "cell_covariates.tsv", sep="\t")
    # HMD = A (posterior 0.9 w.p. 0.6), PMD = B (w.p. 0.25)
    assert cov["pmd_depth"].mean() == pytest.approx(0.8 * 0.35, abs=0.04)


def test_the_mca_index_is_gca_less_gcy_methylation(tmp_path):
    d = make_data(tmp_path, per=10, n_types=2)
    for name, frac in (("GCA", 0.30), ("GCY", 0.20), ("HCA", 0.05),
                       ("HCY", 0.01)):
        dd = tmp_path / name
        dd.mkdir()
        for f in (d / "GCH").glob("*.posteriors.parquet"):
            x = pd.read_parquet(f)[["chrom", "pos1", "pos2", "distance"]]
            x["n1"] = np.int32(100)
            x["n2"] = np.int32(100)
            x["k1"] = np.int32(round(100 * frac))
            x["k2"] = np.int32(round(100 * frac))
            x.to_parquet(dd / f.name.replace("posteriors", "read_pairs"))
    o = tmp_path / "out_mca"
    acc.main(["--posteriors", f"GCH={d / 'GCH'}",
              "--samples", str(d / "samples.tsv"), "--systems", "pmd",
              "--pmd", str(d / "pmd.bed"), "--min-pairs", "10",
              "--n-boot", "20", "--n-threads", "2", "--n-locus-perm", "0",
              "--context-counts", f"GCA={tmp_path / 'GCA'}",
              "--context-counts", f"GCY={tmp_path / 'GCY'}",
              "--context-counts", f"HCA={tmp_path / 'HCA'}",
              "--context-counts", f"HCY={tmp_path / 'HCY'}",
              "--covariate-min-cells", "10", "--no-hotspots",
              "--celltype-auc-stats", "none",
              "--plot-format", "png", "--output-dir", str(o)])
    cov = pd.read_csv(o / "cell_covariates.tsv", sep="\t")
    # HCA less HCY: no GpC in either, so no accessibility in the difference
    np.testing.assert_allclose(cov["mca_index"], 0.04, atol=1e-9)
    np.testing.assert_allclose(cov["mca_index_gch"], 0.1, atol=1e-9)
    np.testing.assert_allclose(cov["nonconversion"], 0.01, atol=1e-9)
    assert cov["meth_GCA"].iloc[0] == pytest.approx(0.3)
    # the mock's types look alike, so there is nothing to simulate
    assert cov["doublet_score"].isna().all()


def test_hotspot_outputs_follow_the_planted_compartment(planted):
    _, _, o = planted
    hs = pd.read_csv(o / "locus_hotspots.tsv.gz", sep="\t")
    en = pd.read_csv(o / "locus_hotspot_enrichment.tsv", sep="\t")
    t = hs[hs["tested"].astype(bool)]
    assert len(t) > 20 and t["cells"].min() >= 10
    # coordination is planted in A-A contacts: A bins are the concordant ones
    r = en[en["track"] == "compartment: A"].iloc[0]
    assert r["rho"] > 0.3 and r["rho"] > r["rho_control"] + 0.2
    assert (o / "locus_hotspots_GCH.png").exists()


def test_celltype_auc_finds_the_group_whose_concordance_differs_by_type():
    rng = np.random.default_rng(3)
    n = 180
    cells = [f"c{i}" for i in range(n)]
    types = np.repeat(["T0", "T1", "T2"], n // 3)
    rows = []
    for grp, effect in (("A", 1.0), ("B", 0.0)):
        shift = effect * (types == "T1") - effect * (types == "T2")
        rows.append(pd.DataFrame({
            "context": "GCH", "system": "ab", "cell": cells,
            "cell_type": types, "anchor": grp, "partner": grp,
            "band": "all 1kb-10Mb", "enough_pairs": True,
            "signed_rel": shift + rng.normal(0, 0.5, n),
            "signed_rel_control": rng.normal(0, 0.5, n),
            "mu_anchor": rng.normal(0.3, 0.05, n)}))
    pc = pd.concat(rows, ignore_index=True)
    res, per = acc.celltype_auc(pc, "signed_rel", n_folds=5, seed=0,
                                n_threads=2, n_trees=100)
    r = res.set_index(["group", "features"])["auc"]
    assert r[("A", "concordance")] > 0.8
    assert r[("A", "marginal + concordance")] > 0.75
    for g in ("A", "B"):
        assert r[(g, "control")] < 0.65 and r[(g, "marginal")] < 0.65
    assert r[("B", "concordance")] < 0.65
    assert r[("all classes", "concordance")] > 0.75
    # T0 sits between T1 and T2: still told apart, less well
    pa = per[(per["group"] == "A") & (per["features"] == "concordance")
             ].set_index("cell_type")["auc"]
    assert pa["T1"] > 0.85 and pa["T2"] > 0.85 and pa["T0"] > 0.6


def test_celltype_auc_outputs_without_type_differences(planted):
    _, _, o = planted
    res = pd.read_csv(o / "celltype_auc.tsv", sep="\t")
    assert set(res["features"]) == set(acc.AUC_VARIANTS)
    # the mock's types differ in nothing
    assert res["auc"].between(0.3, 0.75).all()
    assert (o / "celltype_auc_signed_rel.png").exists()
    assert (o / "celltype_auc_by_type.tsv").exists()


def test_the_total_split_finds_cell_type_and_what_is_left():
    rng = np.random.default_rng(4)
    n = 900
    t_ = np.repeat(["T0", "T1", "T2"], n // 3)
    tech = rng.normal(0, 1, n)
    noise_v, perm_v = 1.0, 1.5                       # noise, + loci
    for resid_sd, left in ((0.0, False), (0.8, True)):
        y = (0.7 * (t_ == "T1") - 0.7 * (t_ == "T2") + 0.5 * tech
             + rng.normal(0, resid_sd, n) + rng.normal(0, np.sqrt(perm_v), n))
        X = tech[:, None].copy()
        X[:30, 0] = np.nan                           # some cells unmeasured
        r = acc._total_split(y, t_, X, ["log10_contacts"],
                             np.empty((n, 0)), noise_v, perm_v, 0.08)
        V = np.var(y, ddof=1)
        assert r["tot_cell type"] == pytest.approx(
            (2 * 0.7 ** 2 / 3) / V, rel=0.3)
        assert r["tot_technical"] == pytest.approx(0.25 / V, abs=0.04)
        assert r["tot_noise"] == pytest.approx(noise_v / V)
        if left:
            assert r["tot_residual"] == pytest.approx(0.64 / V, abs=0.05)
            assert r["residual_p"] < 1e-6
        else:
            assert r["tot_residual"] < 0.05 and r["residual_p"] > 0.01
        assert r["type_r2_unique"] <= r["type_r2_alone"] + 0.02


def test_auc_after_removing_technical_factors():
    """Concordance that follows a technical factor which differs by type
    predicts type -- until the factor is regressed out (cell type kept in
    the fit); a real type difference survives the same correction."""
    rng = np.random.default_rng(5)
    n = 240
    cells = [f"c{i}" for i in range(n)]
    types = np.repeat(["T0", "T1", "T2"], n // 3)
    eff = rng.normal(0, 1, n) + 1.5 * (types == "T1")    # differs by type
    rows = []
    for grp, real in (("A", 0.0), ("B", 1.0)):
        rows.append(pd.DataFrame({
            "context": "GCH", "system": "ab", "cell": cells,
            "cell_type": types, "anchor": grp, "partner": grp,
            "band": "all 1kb-10Mb", "enough_pairs": True,
            "signed_rel": 0.8 * eff + real * (types == "T2")
            + rng.normal(0, 0.3, n),
            "signed_rel_control": rng.normal(0, 0.5, n),
            "mu_anchor": rng.normal(0.3, 0.05, n),
            "sd_anchor": 0.4, "sd_partner": 0.4}))
    pc = pd.concat(rows, ignore_index=True)
    cov = pd.DataFrame({"cell": cells, "cell_type": types,
                        "open_site_level": eff})
    res, _ = acc.celltype_auc(pc, "signed_rel", n_folds=5, seed=0,
                              n_threads=2, n_trees=100, cov=cov)
    r = res.set_index(["group", "features"])["auc"]
    assert r[("A", "concordance")] > 0.62
    assert r[("A", "concordance, technical removed")] < 0.58
    assert r[("A", "concordance")] - r[("A", "concordance, technical "
                                         "removed")] > 0.08
    assert r[("B", "concordance, technical removed")] > 0.75


def test_the_doublet_score_picks_out_two_nuclei_in_one_well():
    rng = np.random.default_rng(6)
    n_bins = 300
    keys = np.arange(n_bins)
    a_pat = rng.uniform(0.1, 0.9, n_bins)
    pat = {"A": a_pat, "B": 1 - a_pat}                # two distinct types

    def cell(ty):
        """(sums, ends) of both halves of a cell's ends."""
        p_ = np.clip(pat[ty] + rng.normal(0, 0.05, n_bins), 0, 1)
        h = []
        for _ in range(2):
            n = rng.poisson(15, n_bins).astype(np.int32)
            h.append((rng.binomial(n, p_).astype(np.float32), n))
        return h

    def entry(halves):
        (s1, n1), (s2, n2) = halves
        return (keys, s1 + s2, n1 + n2, s1, n1)
    prof, types = {}, {}
    for i in range(120):
        ty = "A" if i % 2 else "B"
        prof[f"s{i}"], types[f"s{i}"] = entry(cell(ty)), ty
    for i in range(8):                     # one A and one B nucleus, called A
        a_, b_ = cell("A"), cell("B")
        prof[f"d{i}"] = entry([(a_[0][0] + b_[0][0], a_[0][1] + b_[0][1]),
                               (a_[1][0] + b_[1][0], a_[1][1] + b_[1][1])])
        types[f"d{i}"] = "A"
    ds = acc.doublet_scores({"GCH": prof}, types, seed=0).set_index("cell")
    dbl = ds.loc[[c for c in ds.index if c.startswith("d")], "doublet_score"]
    one = ds.loc[[c for c in ds.index if c.startswith("s")], "doublet_score"]
    # (the planted doublets, all A + B, are each other's neighbours too)
    assert dbl.mean() > 0.6 and one.mean() < 0.1
    assert dbl.min() > one.quantile(0.95)
    # two types that look alike: not simulated, not scored
    alike = {c: ("A" if int(c[1:]) % 2 else "A2") for c in prof}
    prof_a = {c: v for c, v in prof.items() if c.startswith("s")
              and types[c] == "A"}
    sc = acc.doublet_scores({"GCH": prof_a}, alike, seed=0)
    assert sc["doublet_score"].isna().all()


def test_hotspot_enrichment_reports_the_part_beyond_gene_density():
    rng = np.random.default_rng(7)
    code = acc._chrom_code("chr1")
    grid = acc.BinGrid([code], 200_000_000, 1_000_000)
    tss = rng.poisson(5, grid.n).astype(float)
    imp = (tss + rng.normal(0, 1, grid.n) > 7).astype(float)   # rides on TSS
    hs = pd.DataFrame({"context": "GCH", "grid_index": np.arange(grid.n),
                       "tested": True,
                       "delta": 0.01 * tss + rng.normal(0, 0.01, grid.n),
                       "control": rng.normal(0, 0.01, grid.n),
                       "hot": False, "cold": False})
    en = acc.hotspot_enrichment(hs, {"TSSs (count)": tss,
                                     "imprinted genes": imp}).set_index(
        "track")
    assert en.loc["imprinted genes", "rho"] > 0.3
    assert abs(en.loc["imprinted genes", "rho_given_tss"]) < 0.15


def _territory(inter_sd, seed=0):
    """Contacts of 3 types x 60 cells on 10 chromosomes: per contact, a
    locus effect and noise, a chromosome effect, a cell effect -- and,
    optionally, an effect of the cell on that chromosome."""
    rng = np.random.default_rng(seed)
    n_cells, per = 180, 600
    cell = np.repeat(np.arange(n_cells), per)
    ctype = np.repeat(np.arange(3), n_cells // 3)
    chrom = rng.integers(1, 11, cell.size)
    g = chrom.astype(np.int64) * acc.STRIDE + rng.integers(0, 10**8, cell.size)
    d = (rng.normal(0, 0.3, 11)[chrom] + rng.normal(0, 0.2, n_cells)[cell]
         + inter_sd * rng.normal(0, 1, (n_cells, 11))[cell, chrom]
         + rng.normal(0, 0.5, 5000)[rng.integers(0, 5000, cell.size)]
         + rng.normal(0, 1, cell.size))
    return acc.cell_chromosome_test(g, cell, ctype, d, 20,
                                    np.random.default_rng(seed + 1))


def test_the_territory_test_keeps_a_cell_effect_on_one_class_pair():
    """A cell more concordant in one class pair only -- whose share of its
    contacts differs by chromosome by chance -- is no territory state."""
    rng = np.random.default_rng(11)
    n_cells, per = 180, 600
    cell = np.repeat(np.arange(n_cells), per)
    ctype = np.repeat(np.arange(3), n_cells // 3)
    chrom = rng.integers(1, 11, cell.size)
    g = chrom.astype(np.int64) * acc.STRIDE + rng.integers(0, 10**8, cell.size)
    pair = rng.integers(0, 3, cell.size)
    d = (np.where(pair == 0, rng.normal(0, 1.0, n_cells)[cell], 0)
         + rng.normal(0, 1, cell.size))
    r = acc.cell_chromosome_test(g, cell, ctype, d, 20,
                                 np.random.default_rng(12), pair=pair)
    assert r["cell_x_chromosome_share"] < 0.05 and r["perm_p"] > 0.05


def test_the_territory_test_finds_cell_specific_chromosome_states():
    r0, r1 = _territory(0.0), _territory(0.25)
    assert r0["cell_x_chromosome_share"] < 0.05 and r0["perm_p"] > 0.05
    assert r1["cell_x_chromosome_share"] > 0.15 and r1["perm_p"] < 0.05
    assert r1["chromosome_share"] > 0.2 and r1["entries"] > 1000
    # particular to cell and chromosome, not one cell effect scaled
    assert r1["idiosyncratic_share"] > 0.6 * r1["cell_x_chromosome_share"]
    assert r1["idiosyncratic_p"] < 0.05


def test_a_cell_effect_scaled_by_chromosome_is_told_apart():
    rng = np.random.default_rng(13)
    n_cells, per = 180, 600
    cell = np.repeat(np.arange(n_cells), per)
    ctype = np.repeat(np.arange(3), n_cells // 3)
    chrom = rng.integers(1, 11, cell.size)
    g = chrom.astype(np.int64) * acc.STRIDE + rng.integers(0, 10**8, cell.size)
    load = rng.normal(0, 1, 11)                     # per chromosome
    eff = rng.normal(0, 0.3, n_cells)                # per cell
    d = eff[cell] * load[chrom] + rng.normal(0, 1, cell.size)
    r = acc.cell_chromosome_test(g, cell, ctype, d, 20,
                                 np.random.default_rng(14))
    assert r["perm_p"] < 0.05
    assert r["scaled_share"] > 2 * r["idiosyncratic_share"]
    assert r["idiosyncratic_p"] > 0.05


def test_cross_class_residual_tells_measured_from_unmeasured_factors():
    rng = np.random.default_rng(8)
    n = 300
    cells = [f"c{i}" for i in range(n)]
    types = np.repeat(["T0", "T1"], n // 2)
    tech = rng.normal(0, 1, n)
    hidden = rng.normal(0, 1, n)
    for hid, want_resid in ((0.0, False), (1.0, True)):
        rows = []
        for st in ("A", "B"):
            rows.append(pd.DataFrame({
                "context": "GCH", "system": "s", "cell": cells,
                "cell_type": types, "anchor": st, "partner": st,
                "band": "all 1kb-10Mb", "enough_pairs": True,
                "delta": tech + hid * hidden + rng.normal(0, 0.4, n)}))
        pc = pd.concat(rows, ignore_index=True)
        summ = pd.DataFrame({"context": "GCH", "system": "s",
                             "anchor": ["A", "B"], "partner": ["A", "B"],
                             "band": "all 1kb-10Mb", "split_half_r": 0.6})
        cov = pd.DataFrame({"cell": cells, "cell_type": types,
                            "open_site_level": tech})
        cc = acc.cross_class_correlation(pc, summ, min_cells=30, cov=cov)
        r = cc.iloc[0]
        assert r["r"] > 0.7
        if want_resid:
            assert r["r_resid"] > 0.6
        else:
            assert abs(r["r_resid"]) < 0.15


def test_a_nonlinear_fit_takes_what_a_straight_line_misses():
    rng = np.random.default_rng(9)
    n = 1200
    t_ = np.repeat(["T0", "T1"], n // 2)
    x = rng.uniform(-2, 2, n)
    y = x ** 2 + rng.normal(0, 0.5, n)                 # not a straight line
    r = acc._total_split(y, t_, x[:, None], ["open_site_level"],
                         np.empty((n, 0)), 0.25, 0.25, 0.02, nonlinear=True)
    assert r["R2_cv_nonlinear"] > r["R2_cv_linear"] + 0.5
    assert r["tot_residual_nonlinear"] < 0.05 < r["tot_residual"]


def test_doublets_scored_when_types_differ_in_few_bins_among_many():
    """Two types apart in 20 of 600 bins: the spread of all 30 components
    would hide that; along the line between their centroids it shows."""
    rng = np.random.default_rng(10)
    n_bins = 600
    keys = np.arange(n_bins)
    base = rng.uniform(0.3, 0.7, n_bins)
    shift = np.zeros(n_bins)
    shift[:20] = 0.35
    pat = {"A": base, "B": np.clip(base + shift, 0, 1)}
    prof, types = {}, {}
    for i in range(160):
        ty = "A" if i % 2 else "B"
        h = []
        for _ in range(2):
            n = rng.poisson(15, n_bins).astype(np.int32)
            h.append((rng.binomial(n, pat[ty]).astype(np.float32), n))
        prof[f"s{i}"] = (keys, h[0][0] + h[1][0], h[0][1] + h[1][1],
                         h[0][0], h[0][1])
        types[f"s{i}"] = ty
    ds = acc.doublet_scores({"GCH": prof}, types, seed=0)
    assert ds["doublet_score"].notna().all()


def test_methylation_at_open_sites_and_away_from_them(tmp_path):
    df = pd.DataFrame({"chrom": "chr1", "pos1": [1_000, 50_000],
                       "pos2": [30_000, 90_000], "n1": np.int32(10),
                       "n2": np.int32(10), "k1": np.int32([6, 1]),
                       "k2": np.int32([2, 1])})
    f = tmp_path / "c.read_pairs.parquet"
    df.to_parquet(f)
    sites = acc._codes_of(["chr1"]) * acc.STRIDE + np.array([1_050])
    k, n, ko, no, kd, nd = acc._count_pass(f, set(), sites, 200, 5000)
    assert (k, n) == (10.0, 40.0)
    assert (ko, no) == (6.0, 10.0)                  # the end at the site
    assert (kd, nd) == (4.0, 30.0)                  # the three far from it


def test_all_regions_pool_every_contact_of_a_system(planted):
    _, pc, o = planted
    allr = pc[(pc["anchor"] == acc.ALL_REGIONS)
              & pc["band"].str.startswith("all")]
    assert set(allr["system"]) >= {"ab", "chromhmm", "pmd", "rt"}
    cls = pc[(pc["anchor"] != acc.ALL_REGIONS) & pc["band"].str.startswith(
        "all") & (pc["system"] == "pmd")]
    one = allr[allr["system"] == "pmd"].set_index("cell")
    # its contacts are the class pairs' together
    np.testing.assert_allclose(
        cls.groupby("cell")["pairs"].sum().reindex(one.index), one["pairs"])
    # signed, it is the contact-weighted mean of the class pairs' values
    cls = cls[np.isfinite(cls["signed_rel"])]
    want = (cls["signed_rel"] * cls["pairs"]).groupby(cls["cell"]).sum() / (
        cls.groupby("cell")["pairs"].sum())
    np.testing.assert_allclose(one["signed_rel"].reindex(want.index), want,
                               rtol=1e-6, atol=1e-12)
    assert (one["signed_rel"].abs() > 0).any()
    a = pd.read_csv(o / "covariate_association.tsv", sep="\t")
    r = a[(a["anchor"] == acc.ALL_REGIONS) & (a["statistic"] == "signed_rel")
          & a["band"].str.startswith("all")]
    assert r["tot_noise"].notna().all() and r["cell_share"].notna().all()
    top = acc._same_state(a, "GCH", "signed_rel", r["band"].iloc[0])
    assert top["anchor"].iloc[0] == acc.ALL_REGIONS
    assert (top["anchor"] == acc.ALL_REGIONS).sum() == 1
    cc = pd.read_csv(o / "cross_class_correlation.tsv", sep="\t")
    assert acc.ALL_REGIONS not in set(cc["state_1"]) | set(cc["state_2"])


# --------------------------------------------------------------- human --


def test_human_chromhmm_group_names_map_to_the_six_classes(tmp_path):
    groups = {"1_Quies1": "quescient", "2_PromF1": "promoters",
              "3_EnhWk1": "weak enhancers", "4_TxWk1": "weak transcription",
              "5_ReprPC1": "polycomb repressed", "6_HET1": "HET",
              "7_znf1": "znf", "8_BivProm1": "bivalent promoters",
              "9_GapArtf1": "others"}
    rows = []
    for i, (st, g) in enumerate(groups.items()):
        rows.append(("chr1", i * 1000, (i + 1) * 1000, st))
    pd.DataFrame(rows).to_csv(tmp_path / "chmm.bed", sep="\t", header=False,
                              index=False)
    pd.DataFrame({"state": [k.split("_")[0] for k in groups],
                  "mneumonics": [k.split("_")[1] for k in groups],
                  "Group": list(groups.values())}).to_csv(
        tmp_path / "groups.csv", index=False)
    iv = acc.read_chromhmm(tmp_path / "chmm.bed", tmp_path / "groups.csv",
                           acc._enrichment())
    got = [iv.classes[k] if k >= 0 else None
           for k in iv.lookup(["chr1"] * 9, np.arange(9) * 1000 + 500)]
    assert got == ["quiescent", "promoter", "enhancer", "transcribed",
                   "polycomb", "heterochromatin", "heterochromatin",
                   "polycomb", None]


def test_replication_timing_from_log2_tracks(tmp_path):
    rows = [("chr1", s_, s_ + 250_000, 1.5 if (s_ // 2_000_000) % 2 == 0
             else -1.2) for s_ in range(0, 10_000_000, 250_000)]
    pd.DataFrame(rows).to_csv(tmp_path / "H1_RT_250kb.bedGraph", sep="\t",
                              header=False, index=False)
    flag = f"RT_H1={tmp_path / 'H1_RT_250kb.bedGraph'}"
    iv = acc.read_rt(None, None, 250_000, 0.0, log2=[acc._rt_path(flag)])
    k = iv.lookup(["chr1", "chr1"], [100_000, 2_100_000])
    assert [iv.classes[x] for x in k] == ["early", "late"]


def test_several_beds_make_one_system_a_class_each(tmp_path):
    for name, (s_, e_) in (("cLAD", (0, 1000)), ("fLAD", (1000, 2000)),
                           ("ciLAD", (2000, 3000))):
        pd.DataFrame([("chr2", s_, e_)]).to_csv(tmp_path / f"{name}.bed",
                                                sep="\t", header=False,
                                                index=False)
    iv = acc.read_annotation("LAD", [tmp_path / f"{n}.bed"
                                     for n in ("cLAD", "fLAD", "ciLAD")])
    assert iv.classes == ("cLAD", "fLAD", "ciLAD")
    k = iv.lookup(["chr2"] * 3, [10, 1500, 2999])
    assert [iv.classes[x] for x in k] == ["cLAD", "fLAD", "ciLAD"]


def test_human_gene_symbols_reach_the_hotspot_tracks():
    code = acc._chrom_code("chr19")
    grid = acc.BinGrid([code], 60_000_000, 1_000_000)
    genes = pd.DataFrame({
        "chrom": "chr19", "start": [1_100_000, 5_100_000, 9_100_000,
                                    13_100_000, 17_100_000],
        "end": [1_200_000, 5_200_000, 9_200_000, 13_200_000, 17_200_000],
        "name": ["PEG3", "ZNF331", "PCDHA1", "OR1A1", "ZNF91"],
        "strand": "+"})
    tr = acc.hotspot_tracks(grid, {}, None, genes, None, {})
    assert tr["imprinted genes"][[1, 5]].min() > 0          # PEG3, ZNF331
    assert tr["protocadherin cluster"][9] > 0
    assert tr["olfactory receptor genes"][13] > 0
    assert tr["zinc-finger (Zfp / ZNF) genes"][[5, 17]].min() > 0


# ------------------------------------------------------------ two groups --


def test_two_groups_of_cells_compared_by_class_and_along_the_genome(tmp_path):
    """Concordance planted in A (= HMD) contacts of type T0 only: T0 against
    T1 + T2 is higher in HMD-HMD, not in PMD-PMD, and the bins that differ
    are HMD bins."""
    d = make_data(tmp_path, per=20, n_types=3,
                  coord=lambda t_, c: 0.9 if t_ == 0 else 0.0)
    o = tmp_path / "out_contrast"
    acc.main(["--posteriors", f"GCH={d / 'GCH'}",
              "--samples", str(d / "samples.tsv"), "--systems", "pmd",
              "--pmd", str(d / "pmd.bed"), "--min-pairs", "10",
              "--n-boot", "200", "--n-threads", "2", "--n-locus-perm", "0",
              "--covariate-min-cells", "10", "--batch-column", "batch",
              "--celltype-auc-stats", "none", "--hotspot-min-contacts", "100",
              "--contrast-group", "T0=T0", "--contrast-group", "rest=T1,T2",
              "--plot-format", "png", "--output-dir", str(o)])
    ct = pd.read_csv(o / "contrast_classes.tsv", sep="\t")
    a = ct[ct["band"].str.startswith("all") & (ct["statistic"] == "delta")
           & (ct["anchor"] == ct["partner"])].set_index("anchor")
    assert a.loc["HMD", "diff"] > 0.05 and a.loc["HMD", "q_diff"] < 0.05
    assert a.loc["HMD", "diff_lo"] > 0
    assert abs(a.loc["PMD", "diff"]) < 0.02
    assert a.loc["HMD", "diff_adjusted"] > 0.05
    assert a.loc["HMD", "batch_adjusted"] == "batch"
    assert abs(a.loc["HMD", "control_diff"]) < 0.02
    assert {acc.ALL_REGIONS} <= set(a.index)
    hc = pd.read_csv(o / "contrast_hotspots.tsv.gz", sep="\t")
    hmd = is_a(hc["start"])
    assert hc.loc[hmd, "diff"].mean() > hc.loc[~hmd, "diff"].mean() + 0.03
    en = pd.read_csv(o / "contrast_hotspot_enrichment.tsv", sep="\t"
                     ).set_index("track")
    assert en.loc["PMD / HMD: HMD", "rho_diff"] > 0.5
    assert (o / "contrast_GCH_delta.png").exists()
    assert (o / "contrast_hotspots_GCH.png").exists()


def _contrast_args(d, o, *groups):
    return ["--posteriors", f"GCH={d / 'GCH'}",
            "--samples", str(d / "samples.tsv"), "--systems", "pmd",
            "--pmd", str(d / "pmd.bed"), "--min-pairs", "10",
            "--n-boot", "200", "--n-threads", "2", "--n-locus-perm", "0",
            "--covariate-min-cells", "10", "--batch-column", "batch",
            "--celltype-auc-stats", "none", "--hotspot-min-contacts", "100",
            *[x for g in groups for x in ("--contrast-group", g)],
            "--plot-format", "png", "--output-dir", str(o)]


def test_contrast_pairs():
    assert acc.contrast_pairs(["a", "b", "c"]) == [("a", "b"), ("a", "c"),
                                                   ("b", "c")]
    assert acc.contrast_pairs(["a", "b", "c"], "c") == [("a", "c"),
                                                        ("b", "c")]
    assert acc.contrast_pairs(["a", "b"]) == [("a", "b")]


def test_three_groups_compared_pair_by_pair(tmp_path):
    """Concordance planted in HMD contacts of T0 only, each type its own
    group: T0 - T1 and T0 - T2 are higher in HMD-HMD, T1 - T2 is not; each
    pair has its own figures and there is one overview of all; each group's
    batches are listed."""
    d = make_data(tmp_path, per=20, n_types=3,
                  coord=lambda t_, c: 0.9 if t_ == 0 else 0.0)
    o = tmp_path / "out3"
    acc.main(_contrast_args(d, o, "T0=T0", "T1=T1", "T2=T2"))
    ct = pd.read_csv(o / "contrast_classes.tsv", sep="\t")
    pairs = {("T0", "T1"), ("T0", "T2"), ("T1", "T2")}
    assert set(zip(ct["group_A"], ct["group_B"])) == pairs
    a = ct[ct["band"].str.startswith("all") & (ct["statistic"] == "delta")
           & (ct["anchor"] == "HMD") & (ct["partner"] == "HMD")
           ].set_index(["group_A", "group_B"])
    assert a.loc[("T0", "T1"), "diff"] > 0.05
    assert a.loc[("T0", "T2"), "diff"] > 0.05
    assert abs(a.loc[("T1", "T2"), "diff"]) < 0.02
    assert a.loc[("T0", "T1"), "q_diff"] < 0.05
    assert (a["batch_adjusted"] == "batch").all()
    for tag in ("T0_vs_T1", "T0_vs_T2", "T1_vs_T2"):
        assert (o / f"contrast_GCH_delta_{tag}.png").exists()
        assert (o / f"contrast_GCH_signed_rel_{tag}.png").exists()
        assert (o / f"contrast_hotspots_GCH_{tag}.png").exists()
    assert (o / "contrast_overview_GCH_delta.png").exists()
    hc = pd.read_csv(o / "contrast_hotspots.tsv.gz", sep="\t")
    assert set(zip(hc["group_A"], hc["group_B"])) == pairs
    en = pd.read_csv(o / "contrast_hotspot_enrichment.tsv", sep="\t")
    hmd = en[en["track"] == "PMD / HMD: HMD"].set_index(["group_A",
                                                          "group_B"])
    assert hmd.loc[("T0", "T1"), "rho_diff"] > 0.5
    assert abs(hmd.loc[("T1", "T2"), "rho_diff"]) < 0.5
    g = pd.read_csv(o / "contrast_groups.tsv", sep="\t")
    assert (g.loc[g["batch_column"] == "all"].set_index("group")["cells"]
            == 20).all()
    by = g[g["batch_column"] == "batch"]
    assert set(by["batch"]) == {"b0", "b1"}
    assert (by.groupby("group")["cells"].sum() == 20).all()


def test_a_group_concordant_everywhere_differs_in_scale_only(tmp_path):
    """Concordance planted in HMD contacts of every type, twice as strong in
    T0: T0 - rest is higher in HMD-HMD and its bins follow HMD, but beyond
    the pair's all-regions ratio the class doesn't differ, and the bins
    don't depart from the rest's map rescaled to T0's."""
    d = make_data(tmp_path, per=20, n_types=3,
                  coord=lambda t_, c: 0.9 if t_ == 0 else 0.45)
    o = tmp_path / "out_scale"
    acc.main(_contrast_args(d, o, "T0=T0", "rest=T1,T2"))
    ct = pd.read_csv(o / "contrast_classes.tsv", sep="\t")
    a = ct[ct["band"].str.startswith("all") & (ct["statistic"] == "delta")
           & (ct["anchor"] == ct["partner"])].set_index("anchor")
    assert a.loc["HMD", "diff"] > 0.03 and a.loc["HMD", "q_diff"] < 0.05
    assert a.loc["HMD", "scale_ratio"] > 1.5
    assert abs(a.loc["HMD", "diff_scaled"]) < a.loc["HMD", "diff"] / 3
    assert a.loc["HMD", "diff_scaled_lo"] < 0 < a.loc["HMD", "diff_scaled_hi"]
    assert np.isnan(a.loc[acc.ALL_REGIONS, "diff_scaled"])
    hc = pd.read_csv(o / "contrast_hotspots.tsv.gz", sep="\t")
    assert hc["scale_k"].iloc[0] > 1.4 and hc["r_maps"].iloc[0] > 0.8
    en = pd.read_csv(o / "contrast_hotspot_enrichment.tsv", sep="\t"
                     ).set_index("track")
    assert en.loc["PMD / HMD: HMD", "rho_diff"] > 0.5
    assert abs(en.loc["PMD / HMD: HMD", "rho_resid"]) < \
        en.loc["PMD / HMD: HMD", "rho_diff"] / 2


def test_map_scale_recovers_a_rescaled_map():
    rng = np.random.default_rng(1)
    s = rng.normal(size=3000)
    se_a, se_b = np.full(s.size, 0.5), np.full(s.size, 0.5)
    b = s + rng.normal(scale=0.5, size=s.size)
    a = 0.1 + 2 * s + rng.normal(scale=0.5, size=s.size)
    sc = acc._map_scale(a, b, se_a, se_b)
    assert abs(sc["k"] - 2) < 0.1 and abs(sc["offset"] - 0.1) < 0.1
    assert abs(sc["r"] - 1) < 0.05


def test_rotation_p_keeps_the_bins_autocorrelation():
    """Two independent autocorrelated series correlate by chance beyond
    what the plain Spearman p allows; against genome rotations they don't,
    and a real association still stands."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(1)

    def ar1(n, phi=0.95):
        e = rng.normal(size=n)
        x = np.empty(n)
        x[0] = e[0]
        for i in range(1, n):
            x[i] = phi * x[i - 1] + e[i]
        return x
    x, y = ar1(3000), ar1(3000)
    assert spearmanr(x, y)[1] < 1e-6
    assert acc._rotation_p(x, y) > 0.05
    assert acc._rotation_p(x, x + rng.normal(scale=x.std(), size=x.size)
                           ) < 0.01


# ------------------------------------------- chrX and imprinted domains --


def add_sex_chroms(d, female_types=("T0",), n_x=1100, n_x_short=700,
                   n_y=150, seed=5, xi=0.1, xa=0.6, haps=None, tag=0.6,
                   par=None, escape=(), hcg=False, m_xi=0.8, m_xa=0.2):
    """chrX read pairs for every cell -- a female's twice a male's -- and
    chrY short pairs for males.  A female's X contact lies on her inactive
    X (accessibility ``xi``) or her active one (``xa``), both ends alike; a
    male's on his one active X.  With ``haps(type, cell)`` -- the haplotype
    of a female's inactive X -- a share ``tag`` of her X contacts and of
    every cell's autosomal ones (a random haplotype) is written to
    haps/{cell}.haplotypes.parquet, as allelic_imbalance.py would; a
    male's one X has no heterozygous SNP.  ``par`` (start, end), and each
    of ``escape``: X that escapes -- an end in it is at ``xa`` on either
    copy.  With ``hcg``, HCG posteriors of the same read pairs too
    (HCG/{cell}.posteriors.parquet): on her inactive X methylation at
    ``m_xi``, on the active one and where X escapes ``m_xa``; elsewhere 0.5,
    apart from accessibility."""
    rng = np.random.default_rng(seed)
    rng_h = np.random.default_rng([seed, 1])
    rng_m = np.random.default_rng([seed, 2])
    if hcg:
        (Path(d) / "HCG").mkdir(exist_ok=True)
    if haps is not None:
        (Path(d) / "haps").mkdir(exist_ok=True)
    s = pd.read_csv(Path(d) / "samples.tsv", sep="\t")
    for cell, typ in zip(s["cell_name"], s["group"]):
        f = Path(d) / "GCH" / f"{cell}.posteriors.parquet"
        df = pd.read_parquet(f)
        fem = typ in female_types
        k = 2 if fem else 1
        n, ns = n_x * k // 2, n_x_short * k // 2
        a_pos = rng.integers(0, 17_000_000, n)
        b_pos = a_pos + np.exp(rng.uniform(np.log(1_000), np.log(3_000_000),
                                           n)).astype(np.int64)
        on_xi = fem & (rng.random(n) < 0.5)
        lvl = np.where(on_xi, xi, xa)
        esc_a, esc_b = np.zeros(n, bool), np.zeros(n, bool)
        for s0, e0 in ([par] if par is not None else []) + list(escape):
            esc_a |= (a_pos >= s0) & (a_pos < e0)
            esc_b |= (b_pos >= s0) & (b_pos < e0)
        pa = np.where(rng.random(n) < np.where(esc_a, xa, lvl), 0.9, 0.1)
        pb = np.where(rng.random(n) < np.where(esc_b, xa, lvl), 0.9, 0.1)
        s_pos = rng.integers(0, 19_000_000, ns)
        s_end = s_pos + rng.integers(200, 800, ns)
        sa = np.where(rng.random(ns) < xa, 0.9, 0.1)
        sb = np.where(rng.random(ns) < xa, 0.9, 0.1)
        add = [pd.DataFrame({"chrom": "chrX",
                             "pos1": np.r_[a_pos, s_pos],
                             "pos2": np.r_[b_pos, s_end],
                             "posterior1": np.r_[pa, sa],
                             "posterior2": np.r_[pb, sb]})]
        if not fem:
            y = rng.integers(0, 10_000_000, n_y)
            add.append(pd.DataFrame({"chrom": "chrY", "pos1": y,
                                     "pos2": y + 300, "posterior1": 0.1,
                                     "posterior2": 0.1}))
        if haps is not None:
            auto = df[~df["chrom"].isin(["chrX", "chrY"])]
            ta = rng_h.random(len(auto)) < tag
            tx = fem & (rng_h.random(n) < tag)
            h_i = haps(typ, cell)
            pd.DataFrame({
                "chrom": np.r_[auto["chrom"].to_numpy()[ta],
                               np.full(tx.sum(), "chrX")],
                "pos1": np.r_[auto["pos1"].to_numpy()[ta], a_pos[tx]],
                "pos2": np.r_[auto["pos2"].to_numpy()[ta], b_pos[tx]],
                "hap": np.r_[rng_h.integers(1, 3, ta.sum()),
                             np.where(on_xi, h_i, 3 - h_i)[tx]].astype(
                    np.int8)}).to_parquet(
                Path(d) / "haps" / f"{cell}.haplotypes.parquet")
        x = pd.concat(add, ignore_index=True)
        x["n1"] = np.int32(3)
        x["n2"] = np.int32(3)
        x["k1"] = np.round(x["posterior1"] * 3).astype(np.int32)
        x["k2"] = np.round(x["posterior2"] * 3).astype(np.int32)
        x["distance"] = (x["pos2"] - x["pos1"]).abs()
        out = pd.concat([df, x[df.columns]], ignore_index=True)
        out["chrom"] = out["chrom"].astype(str)
        out.to_parquet(f)
        if hcg:
            mlv = np.where(on_xi, m_xi, m_xa)
            ma = np.where(rng_m.random(n) < np.where(esc_a, m_xa, mlv), 0.9,
                          0.1)
            mb = np.where(rng_m.random(n) < np.where(esc_b, m_xa, mlv), 0.9,
                          0.1)
            rest = len(out) - len(df) - n

            def coin(k):
                return np.where(rng_m.random(k) < 0.5, 0.9, 0.1)
            h = out.copy()
            h["posterior1"] = np.r_[coin(len(df)), ma, coin(rest)]
            h["posterior2"] = np.r_[coin(len(df)), mb, coin(rest)]
            h["k1"] = np.round(h["posterior1"] * 3).astype(np.int32)
            h["k2"] = np.round(h["posterior2"] * 3).astype(np.int32)
            h.to_parquet(Path(d) / "HCG" / f"{cell}.posteriors.parquet")
    return d


def test_sex_called_from_x_and_y_coverage():
    # 9000 pairs per 20 Mb autosome: two copies of X give as many per bp
    auto = {1: (9000, 20_000_000), 2: (9000, 20_000_000)}

    def cell(x, y):
        return {**auto, 1001: (x, 20_000_000), 1002: (y, 9_000_000)}
    counts = {**{f"f{i}": cell(9000, 2) for i in range(4)},
              "m1": cell(4500, 90), "m2": cell(4400, 100),
              "fy": cell(8800, 60),          # two X and a Y: a doublet
              "xloss": cell(4500, 2),        # one X, no Y: X lost
              "low": {1: (500, 20_000_000), 1001: (500, 20_000_000)}}
    t = acc.call_sex(counts).set_index("cell")
    assert (t.loc[[f"f{i}" for i in range(4)], "sex_call"] == "female").all()
    assert abs(t.loc["f0", "x_dosage"] - 1) < 0.05
    assert (t.loc[["m1", "m2"], "sex_call"] == "male").all()
    assert abs(t.loc["m1", "x_dosage"] - 0.5) < 0.05
    assert t.loc["fy", "sex_call"] == "unknown"
    assert t.loc["xloss", "sex_call"] == "unknown"
    assert t.loc["low", "sex_call"] == "unknown"        # too few pairs
    assert list(acc._given_sex(["F", "male", "xx", "?"])) == [
        "female", "male", "female", "unknown"]


def test_female_cells_with_mismapped_y_stay_female():
    """A female line: Y pairs only from mismapping, as many in the cells
    short of an X as in the rest -- those are not males."""
    auto = {1: (9000, 20_000_000), 2: (9000, 20_000_000)}
    counts = {f"f{i}": {**auto, 1001: (9000 - 50 * i, 20_000_000),
                        1002: (2, 9_000_000)} for i in range(20)}
    counts["xloss"] = {**auto, 1001: (4000, 20_000_000), 1002: (2, 9e6)}
    t = acc.call_sex(counts).set_index("cell")
    assert (t.drop("xloss")["sex_call"] == "female").all()
    assert t.loc["xloss", "sex_call"] == "unknown"


def test_chrx_concordance_in_females_only_and_nothing_else_changes(
        tmp_path):
    """T0 cells female, with an inactive X; T1, T2 male.  X-X contacts are
    concordant beyond the autosomes in females and not in males; sexes are
    called from coverage; and --chrx changes no other output."""
    d = make_data(tmp_path, per=20, n_types=3)
    add_sex_chroms(d)
    o0, o1 = tmp_path / "plain", tmp_path / "chrx"
    acc.main(_contrast_args(d, o0))
    acc.main(_contrast_args(d, o1) + ["--chrx"])
    sx = pd.read_csv(o1 / "cell_sex.tsv", sep="\t")
    want = np.where(sx["cell_type"] == "T0", "female", "male")
    assert (sx["sex_call"].to_numpy() == want).all()
    cx = pd.read_csv(o1 / "chrx_concordance.tsv", sep="\t")
    al = cx[cx["band"].str.startswith("all") & (cx["statistic"] == "delta")]
    a = al[al["comparison"] == "chrX − autosomes"].set_index("group")
    assert a.loc["female", "diff"] > 0.03 and a.loc["female", "diff_lo"] > 0
    assert a.loc["female", "q_diff"] < 0.05
    assert abs(a.loc["male", "diff"]) < 0.02
    assert a.loc["T0", "group_kind"] == "cell type"
    assert a.loc["female", "pairs_test"] > 5000
    # the density control: X against autosomes thinned to its density, and
    # thinning alone, which here changes nothing
    assert set(al["comparison"]) == {
        "chrX − autosomes", "chrX − autosomes at X density",
        "autosomes at X density − autosomes"}
    th = al[al["comparison"] == "autosomes at X density − autosomes"
            ].set_index("group")
    assert abs(th.loc["all cells", "diff"]) < 0.02
    xd = al[al["comparison"] == "chrX − autosomes at X density"
            ].set_index("group")
    assert xd.loc["female", "diff"] > 0.03
    xm = pd.read_csv(o1 / "chrx_map.tsv", sep="\t")
    assert len(xm) > 5 and (xm["autosome_bins"] > 10).all()
    assert (xm["contacts"] >= 100).all()
    pc = pd.read_csv(o1 / "chrx_percell.tsv.gz", sep="\t")
    f = pc[pc["sex"] == "female"]
    assert (f["delta_X"] - f["delta_autosomes"]).median() > 0.03
    assert (o1 / "chrx_GCH.png").exists()
    # signed: a female's X, half of it closed, is more co-closed than her
    # genome; a male's one active X more co-open
    al = cx[cx["band"].str.startswith("all")
            & (cx["comparison"] == "chrX − autosomes")]
    by = {st: g.set_index("group") for st, g in al.groupby("statistic")}
    assert {"signed_rel", "signed_sgn", "w"} <= set(by)
    w = by["w"]
    assert w.loc["female", "test"] < 0 < w.loc["male", "test"]
    assert w.loc["female", "diff_hi"] < 0 < w.loc["male", "diff_lo"]
    assert by["signed_rel"].loc["female", "diff"] < -0.005
    assert by["signed_sgn"].loc["female", "diff"] < -0.02
    for st in ("signed_rel", "signed_sgn"):
        for col in ("reference", "test"):
            s_, d_ = by[st][col], by["delta"][col]
            assert (s_.abs() <= d_.abs() + 1e-12).all()
    np.testing.assert_allclose(by["signed_sgn"]["test"].abs(),
                               by["delta"]["test"].abs())
    m = pc[pc["sex"] == "male"]
    assert (f["w_X"] - f["w_autosomes"]).median() < 0
    assert (m["w_X"] - m["w_autosomes"]).median() > 0
    np.testing.assert_allclose(f["signed_rel_X"], f["delta_X"] * f["w_X"])
    assert (o1 / "chrx_signed_GCH.png").exists()
    for name in ("summary.tsv", "percell.tsv.gz", "locus_hotspots.tsv.gz"):
        pd.testing.assert_frame_equal(pd.read_csv(o0 / name, sep="\t"),
                                      pd.read_csv(o1 / name, sep="\t"))


def test_chrx_by_haplotype_moves_the_copies_difference_into_the_null(
        tmp_path):
    """T0 female and clonal -- hap2 the inactive X in every cell -- T1
    female and polyclonal (either haplotype), T2 male.  With partners from
    the contact's own haplotype, T0's X excess moves into the null (the
    copy part) and nothing is left beyond the autosomes; in T1 a partner
    of the same haplotype is no closer, so the copy part is ~0 though the
    excess is there; T0's hap2 is the less accessible, co-closed copy."""
    d = make_data(tmp_path, per=20, n_types=3)
    add_sex_chroms(d, female_types=("T0", "T1"), par=(0, 2_000_000),
                   haps=lambda typ, c: (2 if typ == "T0" else
                                        1 + int(c.split("_c")[1]) % 2))
    o = tmp_path / "out"
    acc.main(_contrast_args(d, o) + ["--chrx", "--chrx-par", "0-2000000",
                                     "--haplotypes", str(d / "haps"),
                                     "--chrx-track-bin", "GCH=2500000"])
    h = pd.read_csv(o / "chrx_haplotype_concordance.tsv", sep="\t")
    al = h[h["band"].str.startswith("all")]

    def row(cmp_, st, g):
        r = al[(al["comparison"] == cmp_) & (al["statistic"] == st)
               & (al["group"] == g)]
        assert len(r) == 1, (cmp_, st, g)
        return r.iloc[0]
    cp = row(acc.HAP_COPY_PART, "delta", "T0")
    assert cp["diff"] > 0.05 and cp["diff_lo"] > 0 and cp["q_diff"] < 0.05
    assert row("chrX: either copy − same copy", "delta", "T0")["diff"] > 0.05
    assert abs(row("autosomes: either copy − same copy", "delta",
                   "T0")["diff"]) < 0.01
    assert row("either copy: chrX − autosomes", "delta", "T0")["diff"] > 0.05
    assert abs(row("same copy: chrX − autosomes", "delta",
                   "T0")["diff"]) < 0.015
    # polyclonal: the excess is there, a same-haplotype partner no closer
    assert row("either copy: chrX − autosomes", "delta", "T1")["diff"] > 0.05
    assert abs(row(acc.HAP_COPY_PART, "delta", "T1")["diff"]) < 0.015
    # T0's hap2, the inactive copy: less accessible, co-closed
    lv = row("chrX: hap2 − hap1", "level", "T0")
    assert lv["diff"] < -0.3 and lv["diff_hi"] < 0
    w = row("chrX: hap2 − hap1", "w", "T0")
    assert w["test"] < 0 < w["reference"]
    assert abs(row("autosomes: hap2 − hap1", "level", "T0")["diff"]) < 0.03
    pc = pd.read_csv(o / "chrx_haplotype_percell.tsv.gz", sep="\t")
    t0 = pc[pc["cell_type"] == "T0"]
    assert ((t0["level_X_hap2"] - t0["level_X_hap1"]) < -0.2).all()
    t1 = pc[pc["cell_type"] == "T1"]
    dd = t1["level_X_hap2"] - t1["level_X_hap1"]
    assert (dd < -0.2).sum() == 10 and (dd > 0.2).sum() == 10
    assert (pc.loc[pc["cell_type"] == "T2", "pairs_X_hap1"] == 0).all()
    # PAR1's tagged contacts are left out of X, not counted on a copy
    assert (t0["pairs_X_controls_left_out"] > 0).all()
    assert (o / "chrx_haplotypes_GCH.png").exists()
    # along X, window by window -- 2.5 Mb, slid 0.5 Mb: T0's hap2 the less
    # accessible everywhere; a male's X has no SNP, so no track
    tr = pd.read_csv(o / "chrx_haplotype_tracks.tsv", sep="\t")
    g0 = tr[(tr["context"] == "GCH") & (tr["group"] == "T0")]
    cp = g0[g0["haplotype"].isin(["hap1", "hap2"])]
    lv = cp.pivot(index="center", columns="haplotype",
                  values="level").dropna()
    assert len(lv) >= 25 and ((lv["hap2"] - lv["hap1"]) < -0.2).all()
    assert (g0["bin_end"] - g0["bin_start"]).max() == 2_500_000
    assert np.median(np.diff(np.sort(cp["center"].unique()))) == 500_000
    assert (cp["delta_lo"] <= cp["delta"]).mean() > 0.9
    assert (cp["delta"] <= cp["delta_hi"]).mean() > 0.9
    # the haplotypes' difference, on the same resampled cells
    df = g0[g0["haplotype"] == "hap1-hap2"]
    assert len(df) == len(lv) and (df["level"] > 0.2).all()
    assert (df["delta_lo"] <= df["delta_hi"]).all()
    assert not tr["group"].isin(["T2", "male"]).any()


def test_par1_escapes_so_its_contacts_show_no_excess(tmp_path):
    """T0 female: her inactive X closed, but not in PAR1 (0-2 Mb here),
    which escapes -- both copies alike.  The rest of X is concordant beyond
    the autosomes and beyond PAR1; PAR1 is not beyond the autosomes."""
    d = make_data(tmp_path, per=20, n_types=3)
    add_sex_chroms(d, par=(0, 2_000_000))
    o = tmp_path / "o"
    acc.main(_contrast_args(d, o) + ["--chrx", "--chrx-par", "0-2000000"])
    cx = pd.read_csv(o / "chrx_concordance.tsv", sep="\t")
    al = cx[cx["band"].str.startswith("all") & (cx["statistic"] == "delta")]
    by = {c: g.set_index("group") for c, g in al.groupby("comparison")}
    assert {"chrX PAR1 − autosomes", "chrX − chrX PAR1"} <= set(by)
    x = by["chrX − autosomes"].loc["female", "diff"]
    assert x > 0.03
    assert abs(by["chrX PAR1 − autosomes"].loc["female", "diff"]) < 0.4 * x
    xp = by["chrX − chrX PAR1"].loc["female"]
    assert xp["diff"] > 0.03 and xp["q_diff"] < 0.05
    assert by["chrX PAR1 − autosomes"].loc["female", "pairs_test"] > 1000
    pc = pd.read_csv(o / "chrx_percell.tsv.gz", sep="\t")
    assert {"pairs_PAR1", "delta_PAR1", "w_PAR1"} <= set(pc.columns)
    assert (o / "chrx_GCH.png").exists()


def test_escape_genes_outside_par1():
    par = xci.PAR1["hg38"]
    iv = xci.xci_escape_regions("hg38", par)
    assert len(iv) > 20 and (iv[1:, 0] > iv[:-1, 1]).all()  # disjoint
    assert iv[0, 0] == par[1]            # XG straddles PAR1's end: clipped

    def inside(p):
        return bool(((iv[:, 0] <= p) & (p < iv[:, 1])).any())
    assert inside(53_200_000)            # KDM5C
    assert inside(41_350_000)            # DDX3X
    assert not inside(73_830_000)        # XIST: left out
    assert 4e6 < (iv[:, 1] - iv[:, 0]).sum() < 7e6
    assert xci.xci_escape_regions("mm10") is None
    lk = acc.ChromosomeClasses(par, iv)
    x = acc._chrom_code("X") * acc.STRIDE
    assert list(lk.lookup_g([x + 1_000_000, x + 53_200_000,
                             x + 73_830_000, 5 * acc.STRIDE])) == [
        lk.i_par, lk.i_esc, 1, 0]


def test_escape_genes_join_par1_as_controls(tmp_path):
    """T0 female: her inactive X closed except in PAR1 (0-2 Mb) and an
    escape region (8-10 Mb) -- both copies alike there.  Each control, and
    the two together, show no excess; the rest of X shows it beyond each."""
    d = make_data(tmp_path, per=20, n_types=3)
    add_sex_chroms(d, par=(0, 2_000_000), escape=[(8_000_000, 10_000_000)])
    (d / "escape.bed").write_text("chrX\t8000000\t10000000\n"
                                  "chr1\t0\t1000000\n")
    o = tmp_path / "o"
    acc.main(_contrast_args(d, o) + ["--chrx", "--chrx-par", "0-2000000",
                                     "--chrx-escape", str(d / "escape.bed")])
    cx = pd.read_csv(o / "chrx_concordance.tsv", sep="\t")
    al = cx[cx["band"].str.startswith("all") & (cx["statistic"] == "delta")
            & (cx["group"] == "female")].set_index("comparison")
    x = al.loc["chrX − autosomes", "diff"]
    assert x > 0.03
    for ctl in ("chrX PAR1", "chrX escape", "chrX PAR1 + escape"):
        assert abs(al.loc[f"{ctl} − autosomes", "diff"]) < 0.4 * x, ctl
        assert al.loc[f"chrX − {ctl}", "diff"] > 0.03, ctl
    assert al.loc["chrX PAR1 + escape − autosomes", "pairs_test"] > \
        al.loc["chrX PAR1 − autosomes", "pairs_test"]
    pc = pd.read_csv(o / "chrx_percell.tsv.gz", sep="\t")
    assert {"pairs_PAR1", "pairs_escape", "delta_escape"} <= set(pc.columns)


def test_chrx_across_contexts_and_controls(tmp_path):
    """T0 female, clonal (hap2 her inactive X): on it her X is closed (GCH)
    and methylated (HCG), except in PAR1 (0-2 Mb) and an escape region
    (8-10 Mb), alike on both copies.  HCG at one end, GCH at the other: the
    rest of X anti-concordant beyond the autosomes; the controls, and the
    contacts from a control to the rest of X, either way round, not.  Same
    context, the control-to-X contacts not concordant either.  By
    haplotype, partners from the contact's own copy take X's cross-context
    anti-concordance away."""
    d = make_data(tmp_path, per=20, n_types=3)
    add_sex_chroms(d, par=(0, 2_000_000), escape=[(8_000_000, 10_000_000)],
                   hcg=True, haps=lambda typ, c: 2)
    (d / "escape.bed").write_text("chrX\t8000000\t10000000\n")
    o = tmp_path / "o"
    acc.main(_contrast_args(d, o)
             + ["--posteriors", f"HCG={d / 'HCG'}", "--chrx",
                "--chrx-par", "0-2000000", "--chrx-escape",
                str(d / "escape.bed"), "--haplotypes", str(d / "haps")])
    cc = pd.read_csv(o / "chrx_cross_context.tsv", sep="\t")
    al = cc[cc["band"].str.startswith("all") & (cc["statistic"] == "delta")
            & (cc["group"] == "female")]
    reg = al[al["analysis"] == "regions"].set_index("comparison")
    x = reg.loc["HCG×GCH: chrX − autosomes"]
    assert x["diff"] < -0.04 and x["q_diff"] < 0.05
    cn = "chrX PAR1 + escape"
    for t in (cn, f"HCG in {cn} × GCH in chrX", f"GCH in {cn} × HCG in chrX"):
        assert abs(reg.loc[f"HCG×GCH: {t} − autosomes", "diff"]) < \
            0.4 * abs(x["diff"]), t
        assert reg.loc[f"HCG×GCH: chrX − {t}", "diff"] < -0.03, t
    hp = al[al["analysis"] == "haplotypes"].set_index("comparison")
    assert hp.loc["HCG×GCH: chrX: either copy − same copy", "diff"] < -0.04
    assert hp.loc[f"HCG×GCH: {acc.HAP_COPY_PART}", "diff"] < -0.04
    cx = pd.read_csv(o / "chrx_concordance.tsv", sep="\t")
    g = cx[cx["band"].str.startswith("all") & (cx["statistic"] == "delta")
           & (cx["group"] == "female")]
    for ctx in ("GCH", "HCG"):
        c_ = g[g["context"] == ctx].set_index("comparison")
        xs = c_.loc["chrX − autosomes", "diff"]
        assert xs > 0.03, ctx
        assert abs(c_.loc[f"{cn} × chrX − autosomes", "diff"]) < 0.4 * xs
        assert c_.loc[f"chrX − {cn} × chrX", "diff"] > 0.03, ctx
    pc = pd.read_csv(o / "chrx_cross_context_percell.tsv.gz", sep="\t")
    assert {"pairs_chrX", "delta_chrX", "pairs_chrX_hap2"} <= set(pc.columns)
    assert (o / "chrx_contexts.png").exists()
    # by the contacts' distance, per copy: hap2, her inactive X, the less
    # accessible and the more methylated at every distance (log2 hap1 over
    # hap2); HCG at one end and GCH at the other has no one level
    dt = pd.read_csv(o / "chrx_haplotype_distance.tsv", sep="\t")
    assert {"GCH", "HCG", "HCG×GCH"} <= set(dt["context"])
    f = dt[(dt["group"] == "female") & (dt["set"] == "chrX")
           & (dt["haplotype"] == "hap1-hap2")]
    assert (f.loc[f["context"] == "GCH", "level"] > 1).all()
    assert (f.loc[f["context"] == "HCG", "level"] < -1).all()
    assert f.loc[f["context"] == "HCG×GCH", "level"].isna().all()
    assert f.loc[f["context"] == "HCG×GCH", "delta"].notna().all()
    a = dt[(dt["group"] == "female") & (dt["set"] == "autosomes")
           & (dt["haplotype"] == "hap1-hap2") & (dt["context"] == "GCH")]
    assert len(a) and (a["level"].abs() < 0.3).all()
    assert ((dt["dist_hi"] / dt["dist_lo"]).round(3).isin(
        [round(10 ** s_, 3) for s_ in acc.DIST_STEPS])).all()
    tr = pd.read_csv(o / "chrx_haplotype_tracks.tsv", sep="\t")
    assert "HCG×GCH" in set(tr["context"])


def test_chrx_without_x_reads_stops_and_says_why(tmp_path):
    """Posteriors made without chrX: --chrx stops before the long run,
    saying how to get chrX back; the sexes it found (none) are written."""
    d = make_data(tmp_path, per=10, n_types=2)
    o = tmp_path / "no_x"
    with pytest.raises(SystemExit) as e:
        acc.main(_contrast_args(d, o) + ["--chrx"])
    assert "no cell has a read pair on chrX" in str(e.value)
    assert "--exclude-chrom none" in str(e.value)
    sx = pd.read_csv(o / "cell_sex.tsv", sep="\t")
    assert (sx["sex_call"] == "unknown").all()
    assert not (o / "summary.tsv").exists()


def test_haplotypes_without_files_for_the_cells_stop_early(tmp_path):
    d = make_data(tmp_path, per=10, n_types=2)
    add_sex_chroms(d)
    for hd in (d / "missing", d / "elsewhere"):
        if hd.name == "elsewhere":
            hd.mkdir()
            (hd / "another_cell.haplotypes.parquet").write_text("")
        with pytest.raises(SystemExit, match="no <cell>.haplotypes.parquet"):
            acc.main(_contrast_args(d, tmp_path / "o")
                     + ["--chrx", "--haplotypes", str(hd)])
    assert not (tmp_path / "o" / "summary.tsv").exists()


def test_imprinted_domains_from_genes():
    genes = pd.DataFrame({
        "chrom": ["chr7", "chr7", "chr7", "chr7", "chr7", "chr11"],
        "start": [59_200_000, 59_500_000, 60_000_000, 142_500_000,
                  142_650_000, 12_000_000],
        "end": [59_300_000, 59_600_000, 60_100_000, 142_510_000,
                142_670_000, 12_100_000],
        "name": ["Ube3a", "Atp10a", "Snrpn", "H19", "Igf2", "Grb10"],
        "strand": "+"})
    dm = acc.imprinted_domains(genes, merge=500_000, flank=50_000)
    assert set(dm["name"]) == {"Grb10", "Ube3a", "Snrpn", "H19–Igf2"}
    h = dm.set_index("name").loc["H19–Igf2"]
    assert h["start"] == 142_450_000 and h["end"] == 142_720_000
    dm2 = acc.imprinted_domains(genes, merge=1_000_000)
    assert "Ube3a–Snrpn" in set(dm2["name"])
    assert "Atp10a" not in ",".join(dm2["genes"])


def _domain_data(seed=0, n_cells=40, per=3000):
    """Contacts on chr1 and chr2 (30 Mb each); parts of ΔC with noise,
    plus 0.1 inside D1 (chr1 10-10.5 Mb) in every cell and inside D2 (chr2
    20-20.5 Mb) in cells 0-19 only."""
    rng = np.random.default_rng(seed)
    n = n_cells * per
    cell = np.repeat(np.arange(n_cells), per)
    code = rng.integers(1, 3, n)
    lo = rng.integers(0, 29_000_000, n)
    hi = np.minimum(lo + np.exp(rng.uniform(np.log(1e3), np.log(1e6), n)
                                ).astype(np.int64), 30_000_000 - 1)
    g_lo, g_hi = code * acc.STRIDE + lo, code * acc.STRIDE + hi
    d = rng.normal(0, 0.05, n)
    c = rng.normal(0, 0.05, n)
    in1 = (code == 1) & (lo >= 10_000_000) & (hi < 10_500_000)
    in2 = (code == 2) & (lo >= 20_000_000) & (hi < 20_500_000)
    d = d + 0.1 * in1 + 0.1 * (in2 & (cell < 20))
    dom = pd.DataFrame({"chrom": ["chr1", "chr2"],
                        "start": [10_000_000, 20_000_000],
                        "end": [10_500_000, 20_500_000],
                        "name": ["D1", "D2"], "genes": ["G1", "Ube3a"]})
    tss = np.r_[1 * acc.STRIDE + rng.integers(0, 30_000_000, 300),
                2 * acc.STRIDE + rng.integers(0, 30_000_000, 300)]
    return dom, g_lo, g_hi, cell, d, c, tss


def test_domain_concordance_against_matched_windows():
    dom, g_lo, g_hi, cell, d, c, tss = _domain_data()
    groups = {"all cells": np.arange(40), "A": np.arange(20),
              "B": np.arange(20, 40)}
    rows, crows, wins = acc.domain_concordance(
        dom, g_lo, g_hi, cell, d, c, 40, groups, pairs=[("A", "B")],
        tss_g=tss, n_controls=60, min_contacts=30, n_boot=100, seed=1)
    t = pd.DataFrame(rows).set_index(["group", "domain"])
    assert t.loc[("all cells", "D1"), "excess"] > 0.07
    assert t.loc[("all cells", "D1"), "z"] > 3
    assert t.loc[("all cells", "D1"), "p"] < 0.01
    assert t.loc[("all cells", "D1"), "windows"] == 60
    assert abs(t.loc[("B", "D2"), "excess"]) < 0.03
    assert t.loc[("all cells", "all domains"), "excess_lo"] > 0
    ct = pd.DataFrame(crows).set_index("domain")
    assert ct.loc["D2", "diff"] > 0.07 and ct.loc["D2", "z"] > 3
    assert abs(ct.loc["D1", "z"]) < 3
    assert ct.loc["D2", "lo"] > 0
    assert (wins.groupby("domain").size() == 60).all()
    assert (wins["contacts"] >= 30).all()


def test_a_sparse_domain_is_not_tested_and_its_noise_counts():
    """The windows' spread alone would make a domain of a few contacts look
    extreme: its own standard error enters z, and below the windows'
    minimum of contacts it is not tested at all."""
    rng = np.random.default_rng(3)
    cv = rng.normal(0, 0.1, 100)
    assert acc._vs_windows(1.0, cv)["z"] > 5
    assert abs(acc._vs_windows(1.0, cv, se=1.0)["z"]) < 1.5
    v = acc._vs_windows(1.0, cv, se=1.0, tested=False)
    assert np.isnan(v["z"]) and np.isnan(v["p"]) and v["windows"] == 100
    dom, g_lo, g_hi, cell, d, c, tss = _domain_data()
    tiny = pd.DataFrame({"chrom": ["chr1"], "start": [25_000_000],
                         "end": [25_020_000], "name": ["tiny"],
                         "genes": ["T1"]})
    rows, _, _ = acc.domain_concordance(
        pd.concat([dom, tiny], ignore_index=True), g_lo, g_hi, cell, d, c,
        40, {"all cells": np.arange(40)}, tss_g=tss, n_controls=30,
        min_contacts=30, n_boot=50, seed=4)
    t = pd.DataFrame(rows).set_index("domain")
    assert t.loc["tiny", "contacts"] < 30 and np.isnan(t.loc["tiny", "z"])
    assert t.loc["D1", "z"] > 3


def test_matched_windows_share_the_domains_chromatin():
    """Contacts on chr1 below 15 Mb are class pair 1, all others 0: D1's
    windows come from there, D2's from elsewhere."""
    dom, g_lo, g_hi, cell, d, c, tss = _domain_data()
    code, pos = g_lo // acc.STRIDE, g_lo % acc.STRIDE
    pair = ((code == 1) & (pos < 15_000_000)).astype(np.int64)
    _, _, wins = acc.domain_concordance(
        dom, g_lo, g_hi, cell, d, c, 40, {"all cells": np.arange(40)},
        pair=pair, n_controls=30, min_contacts=30, n_boot=50, seed=2)
    w1 = wins[wins["domain"] == "D1"]
    w2 = wins[wins["domain"] == "D2"]
    assert ((w1["code"] == 1) & (w1["end"] <= 15_000_000)).all()
    assert ((w2["code"] == 2) | (w2["start"] >= 15_000_000)).all()


def test_pcdh_domains_from_genes():
    """The whole cluster and its alpha, beta, gamma parts; Pcdh9, a
    non-clustered protocadherin, is not one of them."""
    genes = pd.DataFrame({
        "chrom": ["chr18"] * 7 + ["chr14"],
        "start": [36_950_000, 37_000_000, 37_100_000, 37_300_000,
                  37_350_000, 37_700_000, 37_900_000, 118_000_000],
        "end": [36_960_000, 37_010_000, 37_110_000, 37_310_000,
                37_360_000, 37_710_000, 37_910_000, 119_000_000],
        "name": ["Pcdha1", "Pcdha2", "Pcdhac2", "Pcdhb1", "Pcdhb2",
                 "Pcdhga1", "Pcdhgc5", "Pcdh9"], "strand": "+"})
    dm = acc.pcdh_domains(genes, flank=50_000).set_index("name")
    assert list(dm.index) == ["Pcdh cluster", "Pcdha", "Pcdhb", "Pcdhg"]
    assert (dm.loc["Pcdh cluster", "start"],
            dm.loc["Pcdh cluster", "end"]) == (36_900_000, 37_960_000)
    assert (dm.loc["Pcdha", "start"], dm.loc["Pcdha", "end"]) == (
        36_900_000, 37_160_000)
    assert "Pcdh9" not in ",".join(dm["genes"])
    assert (dm["chrom"] == "chr18").all()
    assert acc.pcdh_domains(genes.iloc[[7]]).empty


def test_domain_levels_per_group_and_pair():
    # 4 cells x 2 domains x (ends, posterior sum)
    lv = np.array([[[10, 8], [10, 2]], [[10, 8], [10, 2]],
                   [[10, 2], [10, 2]], [[10, 2], [10, 2]]], float)
    rows, crows = acc.domain_levels(lv, ["D1", "D2"],
                                    {"A": [0, 1], "B": [2, 3]},
                                    [("A", "B")], n_boot=50, seed=1)
    t = pd.DataFrame(rows).set_index(["group", "domain"])
    assert t.loc[("A", "D1"), "level"] == 0.8
    assert t.loc[("B", "D1"), "level"] == 0.2
    c = pd.DataFrame(crows).set_index("domain")
    assert abs(c.loc["D1", "level_diff"] - 0.6) < 1e-9
    assert c.loc["D2", "level_diff"] == 0


def plant_region_concordance(d, types=("T0",), chrom="chr1",
                             lo=200_000, hi=1_160_000, p=0.9, seed=7):
    """In cells of ``types``, a contact with both ends in [lo, hi) copies
    its first end's state into its second with probability ``p``."""
    rng = np.random.default_rng(seed)
    s = pd.read_csv(Path(d) / "samples.tsv", sep="\t")
    for cell, typ in zip(s["cell_name"], s["group"]):
        if typ not in types:
            continue
        f = Path(d) / "GCH" / f"{cell}.posteriors.parquet"
        df = pd.read_parquet(f)
        a = np.minimum(df["pos1"], df["pos2"])
        b = np.maximum(df["pos1"], df["pos2"])
        m = ((df["chrom"].astype(str) == chrom) & (a >= lo) & (b < hi)
             & (b - a >= 1000) & (rng.random(len(df)) < p))
        df.loc[m, "posterior2"] = df.loc[m, "posterior1"]
        df.loc[m, "k2"] = df.loc[m, "k1"]
        df["chrom"] = df["chrom"].astype(str)
        df.to_parquet(f)


def test_pcdh_neurons_against_glia(tmp_path):
    """Concordance planted inside the protocadherin cluster in T0
    ('neuron') cells only: neuron - glia is positive there, beyond the
    matched windows, and each group's level in the cluster is reported."""
    d = make_data(tmp_path, per=15, n_types=2)
    plant_region_concordance(d)
    pd.DataFrame({"c": "chr1",
                  "s": [250_000, 300_000, 400_000, 600_000, 900_000,
                        1_100_000],
                  "e": [260_000, 310_000, 410_000, 610_000, 910_000,
                        1_110_000],
                  "n": ["Pcdha1", "Pcdha2", "Pcdhac2", "Pcdhb1", "Pcdhga1",
                        "Pcdhgc5"], "sc": 0, "st": "+"}
                 ).to_csv(d / "genes.bed", sep="\t", header=False,
                          index=False)
    o = tmp_path / "out_pcdh"
    acc.main(_contrast_args(d, o, "neuron=T0", "glia=T1") + [
        "--pcdh", "--gene-annotation", str(d / "genes.bed"),
        "--imprinting-windows", "20", "--imprinting-min-contacts", "5"])
    dm = pd.read_csv(o / "pcdh_domains.tsv", sep="\t")
    assert list(dm["name"]) == ["Pcdh cluster", "Pcdha", "Pcdhb", "Pcdhg"]
    ic = pd.read_csv(o / "pcdh_contrast.tsv", sep="\t").set_index("domain")
    r = ic.loc["Pcdh cluster"]
    assert (r["group_A"], r["group_B"]) == ("neuron", "glia")
    assert r["diff"] > 0.05 and r["lo"] > 0 and r["excess"] > 0.05
    assert r["z"] > 2 and np.isfinite(r["level_diff"])
    it = pd.read_csv(o / "pcdh_concordance.tsv", sep="\t")
    assert not (it["domain"] == "all domains").any()
    c = it[(it["domain"] == "Pcdh cluster")].set_index("group")
    assert c.loc["neuron", "excess"] > 0.05
    assert 0 < c.loc["glia", "level"] < 1
    assert (o / "pcdh_GCH.png").exists()


def test_imprinting_run_writes_its_tables(tmp_path):
    d = make_data(tmp_path, per=15, n_types=2,
                  coord=lambda t_, c: 0.8)
    pd.DataFrame({"c": ["chr1", "chr1", "chr2"],
                  "s": [100_000, 400_000, 4_300_000],
                  "e": [150_000, 450_000, 4_350_000],
                  "n": ["H19", "Igf2", "Grb10"], "sc": 0, "st": "+"}
                 ).to_csv(d / "genes.bed", sep="\t", header=False,
                          index=False)
    o = tmp_path / "out_imp"
    acc.main(_contrast_args(d, o, "T0=T0", "T1=T1") + [
        "--imprinting", "--gene-annotation", str(d / "genes.bed"),
        "--imprinting-windows", "20", "--imprinting-min-contacts", "5"])
    dm = pd.read_csv(o / "imprinting_domains.tsv", sep="\t")
    assert set(dm["name"]) == {"H19–Igf2", "Grb10"}
    it = pd.read_csv(o / "imprinting_concordance.tsv", sep="\t")
    assert {"all cells", "T0", "T1"} <= set(it["group"])
    assert (it["domain"] == "all domains").any()
    assert (it.loc[it["domain"] != "all domains", "contacts"] > 0).all()
    ic = pd.read_csv(o / "imprinting_contrast.tsv", sep="\t")
    assert set(zip(ic["group_A"], ic["group_B"])) == {("T0", "T1")}
    w = pd.read_csv(o / "imprinting_windows.tsv", sep="\t")
    assert set(w["domain"]) <= {"H19–Igf2", "Grb10"} and len(w) > 0
    assert set(w["chrom"]) <= {"chr1", "chr2"}
    assert (o / "imprinting_GCH.png").exists()


# ------------------------------------------------ variable and stable loci --


def test_variable_and_stable_loci_across_cell_types():
    """Bins 5, 15, 25, 35: concordant in type 0 only; bins 8, 18, 28, 38:
    concordant alike in all three types."""
    rng = np.random.default_rng(15)
    code = acc._chrom_code("chr1")
    grid = acc.BinGrid([code], 50_000_000, 1_000_000)
    n = 300_000
    lo = rng.integers(0, 40_000_000, n)
    hi = lo + rng.integers(1_000, 20_000, n)
    ty = rng.integers(0, 3, n)
    b = lo // 1_000_000
    var_b, st_b = np.isin(b, [5, 15, 25, 35]), np.isin(b, [8, 18, 28, 38])
    d = rng.normal(0, 1, n) + 0.6 * (var_b & (ty == 0)) + 0.6 * st_b
    c = rng.normal(0, 1, n)
    g0 = code * acc.STRIDE
    lv = acc.locus_variability(grid, g0 + lo, g0 + hi, ty, {"delta": (d, c)},
                               ["T0", "T1", "T2"], min_contacts=100,
                               fraction=0.1)
    got_v = set(lv.loc[lv["set"] == "variable", "start"] // 1_000_000)
    got_s = set(lv.loc[lv["set"] == "stable", "start"] // 1_000_000)
    assert got_v == {5, 15, 25, 35}
    assert got_s == {8, 18, 28, 38}
    assert (lv["type:T0"] > lv["type:T1"]).loc[lv["set"] == "variable"].all()
    lv.insert(0, "context", "GCH")
    track_v, track_s = np.zeros(grid.n), np.zeros(grid.n)
    track_v[[5, 15, 25, 35]] = 1.0
    track_s[[8, 18, 28, 38]] = 1.0
    en = acc.variability_enrichment(lv, {"tv": track_v, "ts": track_s}
                                    ).set_index(["set", "track"])
    assert en.loc[("variable", "tv"), "log2_fold"] > 3
    assert en.loc[("stable", "ts"), "log2_fold"] > 3
    assert en.loc[("variable", "tv"), "q"] < 0.05


def test_the_variability_outputs_are_written(planted):
    _, _, o = planted
    lv = pd.read_csv(o / "locus_variability.tsv.gz", sep="\t")
    assert {"delta", "signed_rel"} <= set(lv["statistic"])
    assert {"between_type_sd", "het_q", "set", "type:T0"} <= set(lv.columns)
    for nm in ("variable", "variable_beyond_type_level", "stable",
               "tested"):
        assert (o / f"locus_{nm}_GCH_delta.bed").exists()
    assert (o / "locus_variability_GCH_signed_rel.png").exists()
    assert (o / "locus_variability_enrichment.tsv").exists()


def test_variability_beyond_each_types_overall_level():
    """Type 0 is twice as concordant everywhere; in bins 7, 17, 27, 37 type
    1 alone is high.  The plain call follows type 0's scale in the most
    concordant bins; beyond each type's level, only the four remain."""
    rng = np.random.default_rng(16)
    code = acc._chrom_code("chr1")
    grid = acc.BinGrid([code], 50_000_000, 1_000_000)
    n = 400_000
    lo = rng.integers(0, 40_000_000, n)
    hi = lo + rng.integers(1_000, 20_000, n)
    ty = rng.integers(0, 3, n)
    b = lo // 1_000_000
    base = np.where(b % 2 == 0, 0.8, 0.2)        # even bins more concordant
    scale = np.array([2.0, 1.0, 1.0])[ty]
    part = np.isin(b, [7, 17, 27, 37]) & (ty == 1)
    d = base * scale + 0.5 * part + rng.normal(0, 1, n)
    c = rng.normal(0, 1, n)
    g0 = code * acc.STRIDE
    lv = acc.locus_variability(grid, g0 + lo, g0 + hi, ty, {"delta": (d, c)},
                               ["T0", "T1", "T2"], min_contacts=100,
                               fraction=0.1)
    plain = set(lv.loc[lv["set"] == "variable", "start"] // 1_000_000)
    scaled = set(lv.loc[lv["set_scaled"] == "variable", "start"]
                 // 1_000_000)
    assert not plain & {7, 17, 27, 37}               # the scale's bins
    assert all(x % 2 == 0 for x in plain)
    assert scaled == {7, 17, 27, 37}


def test_posteriors_that_are_not_there_say_why(tmp_path):
    """A --posteriors directory missing, empty, or of other cells: each
    stops at once, saying which."""
    import shutil
    d = make_data(tmp_path, per=5, n_types=2)
    (tmp_path / "empty").mkdir()
    (tmp_path / "other").mkdir()
    shutil.copy(next((d / "GCH").glob("*.posteriors.parquet")),
                tmp_path / "other" / "cellZ.posteriors.parquet")
    args = _contrast_args(d, tmp_path / "o")
    i = args.index("--posteriors") + 1
    for sub, msg in (("nope", "no such directory"),
                     ("empty", r"no \*\.posteriors\.parquet"),
                     ("other", "none of its 1 cells")):
        a = list(args)
        a[i] = f"GCH={tmp_path / sub}"
        with pytest.raises(SystemExit, match=msg):
            acc.main(a)


def test_track_bins_per_context():
    """--chrx-track-bin: a size for all, per context, 'none'; 'auto' takes
    the finest bin where a copy's median bin holds 100 contacts."""
    assert acc.parse_track_bins("auto") == {None: -1}
    assert acc.parse_track_bins("2500000") == {None: 2_500_000}
    assert acc.parse_track_bins("hcg=10000000,GCH=auto") == {
        None: -1, "HCG": 10_000_000, "GCH": -1}
    assert acc.parse_track_bins("none") == {None: 0}
    with pytest.raises(ValueError):
        acc.parse_track_bins("HCG=-5")
    rng = np.random.default_rng(0)
    # 80,000 ends over 150 Mb (a contact is two): a copy's 1 Mb bin holds
    # ~133 contacts -> 1 Mb
    pos = rng.integers(0, 150_000_000, 80_000)
    hap = rng.integers(0, 2, pos.size)
    assert acc.track_bin_auto(pos, hap) == 1_000_000
    # 16,000: ~67 at 2.5 Mb, ~133 at 5 Mb
    assert acc.track_bin_auto(pos[:16_000], hap[:16_000]) == 5_000_000
    # 1,200: ~40 even at 20 Mb -> the coarsest
    assert acc.track_bin_auto(pos[:1_200], hap[:1_200]) == 20_000_000


def test_sliding_windows_show_where_concordance_turns():
    """A wide window slid in small steps: a copy whose ΔC drops from 0.2 to
    0 at 50 Mb turns there smoothly -- 0.2 a window's half-width before,
    0 after, halfway at 50 Mb -- the other copy flat; their difference with
    its interval."""
    rng = np.random.default_rng(1)
    n = 40_000
    base = acc._chrom_code("X") * acc.STRIDE
    lo = rng.integers(0, 99_000_000, n)
    hi = lo + 1_000
    cell = rng.integers(0, 50, n)
    hap = rng.integers(0, 2, n)
    d = np.where((hap == 0) & (lo < 50_000_000), 0.2, 0.0) \
        + rng.normal(0, 0.05, n)
    t = acc._haplotype_tracks(cell, np.ones(n, bool), hap, base + lo,
                              base + hi, d, np.full(n, 0.5),
                              np.full(n, 0.5), {"G": np.arange(50)}, 50,
                              10_000_000, 100, 0, track_step=1_000_000)
    h1 = t[t["haplotype"] == "hap1"].set_index("center")
    assert (t["bin_end"] - t["bin_start"]).max() == 10_000_000
    assert np.allclose(np.diff(h1.index[5:-5]), 1_000_000)
    assert abs(h1.loc[40_000_000, "delta"] - 0.2) < 0.01
    assert abs(h1.loc[60_000_000, "delta"]) < 0.01
    assert abs(h1.loc[50_000_000, "delta"] - 0.1) < 0.02
    h2 = t[t["haplotype"] == "hap2"]
    assert h2["delta"].abs().max() < 0.02
    df = t[t["haplotype"] == "hap1-hap2"].set_index("center")
    assert df.loc[40_000_000, "delta_lo"] > 0.15
    assert df.loc[60_000_000, "delta_lo"] < 0 < df.loc[60_000_000,
                                                       "delta_hi"]
