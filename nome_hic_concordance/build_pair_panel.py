"""
build_pair_panel.py
===================

Generate a multi-category TF-anchored pair panel BEDPE for the pair-concordance
contact-aware analysis pipeline.

Pipeline:
  1. For each TF specified in the config, download or load its JASPAR PWM(s).
  2. Scan PWMs against the reference genome (mm10 / hg38) using MOODS.
  3. Optionally intersect motif occurrences with ChIP-seq peak BEDs (ENCODE).
  4. For each TF category, enumerate motif anchor pairs within the specified
     distance bands, sample down to the target panel size, and assign a
     category label.
  5. Build co_folded / non_folded / random_distance_matched control categories
     from a pseudobulk cooler.
  6. Concatenate all categories into a single BEDPE with a `category` column,
     deduplicate by (chrom_i, start_i, chrom_j, start_j), and write parquet + TSV.

Output schema:
  pair_id  chrom_i  start_i  end_i  chrom_j  start_j  end_j  distance
  tier  category  distance_band

Example:
  python build_pair_panel.py \\
      --config configs/gm12878_imr90_panel.yaml \\
      --genome-fa /ref/hg38.fa \\
      --pseudobulk-cool /data/pseudobulk_mix.mcool::resolutions/25000 \\
      --chrom-sizes /ref/hg38.chrom.sizes \\
      --out-dir output/panels/gm12878_imr90/ \\
      --threads 8

The config YAML specifies the TF categories, motif IDs, optional ChIP BEDs,
distance bands, and per-category target panel sizes. See
configs/gm12878_imr90_panel.example.yaml.

Requirements:
  pip install MOODS-python pyranges cooler pyyaml joblib pyfaidx pyarrow
  External (optional): bedtools (only if --use-bedtools fallback)

Author: meth_concord pair_concordance subpackage utilities
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
import sys
import urllib.request
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyranges as pr
import yaml
from joblib import Parallel, delayed

try:
    import MOODS.parsers
    import MOODS.scan
    import MOODS.tools
except ImportError:
    sys.exit(
        "MOODS-python is required. Install via: pip install MOODS-python"
    )

try:
    from pyfaidx import Fasta
except ImportError:
    sys.exit("pyfaidx is required. Install via: pip install pyfaidx")


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("build_pair_panel")


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------
@dataclass
class TFCategoryConfig:
    """Configuration for a single TF category in the pair panel."""

    category: str                 # e.g. "tf_anchored_NEUROD2"
    jaspar_ids: list[str]         # e.g. ["MA0668.1"] (one or more PWMs to scan together)
    chip_bed: Optional[str] = None  # path to ChIP-seq peak BED for intersection
    target_pairs_per_band: int = 5_000
    pseudo_pvalue_threshold: float = 1e-4   # FIMO-equivalent threshold


@dataclass
class PanelConfig:
    """Full pair-panel configuration."""

    tf_categories: list[TFCategoryConfig] = field(default_factory=list)
    distance_bands: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "short": (100_000, 500_000),
            "mid": (500_000, 2_000_000),
            "long": (2_000_000, 5_000_000),
        }
    )
    anchor_size: int = 5_000              # 5 kb anchors
    contacts_top_decile: bool = True       # filter to top decile pseudobulk contact pairs for co_folded
    random_seed: int = 42
    include_co_folded: bool = True
    include_random_matched: bool = True
    target_co_folded_per_band: int = 3_000
    target_random_per_band: int = 3_000
    # Reference genome chromosomes to scan (default: autosomes; user can override)
    chromosomes: Optional[list[str]] = None
    jaspar_release: str = "JASPAR2024_CORE_vertebrates_non-redundant"  # JASPAR release identifier


# ----------------------------------------------------------------------
# JASPAR motif fetching
# ----------------------------------------------------------------------
JASPAR_BASE = "https://jaspar.genereg.net/api/v1/matrix/{matrix_id}.pfm"


def fetch_jaspar_pfm(matrix_id: str, cache_dir: Path) -> Path:
    """
    Download a JASPAR PFM file (counts format) and cache it.

    Returns the local path. PFM format is what MOODS expects.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    pfm_path = cache_dir / f"{matrix_id}.pfm"
    if pfm_path.exists():
        return pfm_path

    url = JASPAR_BASE.format(matrix_id=matrix_id)
    log.info(f"  fetching JASPAR PFM: {matrix_id} -> {pfm_path}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read().decode("utf-8")
        pfm_path.write_text(content)
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch JASPAR matrix {matrix_id} from {url}: {e}\n"
            f"You can manually download the PFM and place it at {pfm_path}"
        )
    return pfm_path


# ----------------------------------------------------------------------
# Motif scanning with MOODS
# ----------------------------------------------------------------------
def scan_motifs_on_chromosome(
    chrom: str,
    sequence: str,
    pfm_paths: list[Path],
    pvalue_threshold: float,
    bg_freqs: list[float] = (0.25, 0.25, 0.25, 0.25),
) -> pd.DataFrame:
    """
    Scan one or more PWMs against a single chromosome sequence.

    Returns a DataFrame with columns: chrom, start, end, motif_id, score, strand.
    Positions are 0-based, half-open.
    """
    if len(sequence) == 0 or not pfm_paths:
        return pd.DataFrame(columns=["chrom", "start", "end", "motif_id", "score", "strand"])

    # Parse PFMs into MOODS log-odds matrices, both strands
    matrices = []
    motif_ids = []
    for pfm_path in pfm_paths:
        matrix = MOODS.parsers.pfm_to_log_odds(str(pfm_path), bg_freqs, 1e-4)
        rev_matrix = MOODS.tools.reverse_complement(matrix, 4)
        matrices.append(matrix)
        matrices.append(rev_matrix)
        motif_ids.append(pfm_path.stem)
        motif_ids.append(pfm_path.stem)  # same id for rev complement; we record strand below

    # Per-matrix score thresholds from the requested p-value
    thresholds = [
        MOODS.tools.threshold_from_p(m, bg_freqs, pvalue_threshold)
        for m in matrices
    ]

    seq_bytes = sequence.upper()
    results = MOODS.scan.scan_dna(seq_bytes, matrices, bg_freqs, thresholds, 7)

    rows = []
    for matrix_idx, hits in enumerate(results):
        motif_id = motif_ids[matrix_idx]
        is_reverse = (matrix_idx % 2 == 1)
        strand = "-" if is_reverse else "+"
        motif_len = len(matrices[matrix_idx][0])
        for hit in hits:
            start = hit.pos
            end = start + motif_len
            rows.append((chrom, start, end, motif_id, hit.score, strand))

    if not rows:
        return pd.DataFrame(columns=["chrom", "start", "end", "motif_id", "score", "strand"])

    df = pd.DataFrame(rows, columns=["chrom", "start", "end", "motif_id", "score", "strand"])
    return df


def _scan_one_chrom_worker(
    genome_fa_str: str,
    chrom: str,
    pfm_path_strs: list[str],
    pvalue_threshold: float,
) -> pd.DataFrame:
    """Worker function: opens its own Fasta handle (cannot pickle file handles).

    Must be a module-level function so joblib can pickle the call (not
    the captured closure with an open Fasta object).
    """
    fasta = Fasta(genome_fa_str, as_raw=True)
    try:
        if chrom not in fasta:
            log.warning(f"  chromosome {chrom} not in FASTA, skipping")
            return pd.DataFrame()
        seq = fasta[chrom][:].upper()
        log.info(f"  scanning {chrom} ({len(seq):,} bp)")
        pfm_paths = [Path(p) for p in pfm_path_strs]
        return scan_motifs_on_chromosome(chrom, seq, pfm_paths, pvalue_threshold)
    finally:
        # pyfaidx's Fasta has a close() method; best-effort cleanup.
        try:
            fasta.close()
        except Exception:
            pass


def scan_genome_for_motifs(
    genome_fa: Path,
    pfm_paths: list[Path],
    chromosomes: list[str],
    pvalue_threshold: float,
    threads: int,
) -> pd.DataFrame:
    """Scan PWMs across multiple chromosomes in parallel.

    Each worker opens its own Fasta handle — file handles cannot be
    pickled, so we cannot share a single Fasta across processes.
    """
    pfm_path_strs = [str(p) for p in pfm_paths]
    genome_fa_str = str(genome_fa)

    dfs = Parallel(n_jobs=threads)(
        delayed(_scan_one_chrom_worker)(
            genome_fa_str, chrom, pfm_path_strs, pvalue_threshold,
        )
        for chrom in chromosomes
    )
    out = pd.concat([d for d in dfs if len(d) > 0], ignore_index=True)
    log.info(f"  total motif occurrences: {len(out):,}")
    return out


# ----------------------------------------------------------------------
# ChIP-seq intersection
# ----------------------------------------------------------------------
def intersect_motifs_with_chip(
    motifs: pd.DataFrame,
    chip_bed_path: str,
) -> pd.DataFrame:
    """
    Restrict motif occurrences to those overlapping a ChIP-seq peak.

    chip_bed_path: BED file (3+ columns: chrom, start, end). May be gzipped.
    """
    if not chip_bed_path or not Path(chip_bed_path).exists():
        log.warning(
            f"  ChIP-seq BED {chip_bed_path} not found; skipping intersection. "
            f"All {len(motifs):,} motif occurrences retained."
        )
        return motifs

    log.info(f"  intersecting {len(motifs):,} motifs with ChIP-seq peaks from {chip_bed_path}")
    chip_pr = pr.read_bed(chip_bed_path)
    motif_pr = pr.PyRanges(motifs.rename(columns={
        "chrom": "Chromosome",
        "start": "Start",
        "end": "End",
    }))
    overlap = motif_pr.overlap(chip_pr)
    out = overlap.df.rename(columns={
        "Chromosome": "chrom",
        "Start": "start",
        "End": "end",
    })
    log.info(f"  retained {len(out):,} motif occurrences after ChIP intersection")
    return out


# ----------------------------------------------------------------------
# Anchor construction
# ----------------------------------------------------------------------
def motifs_to_anchors(
    motifs: pd.DataFrame,
    anchor_size: int,
    chrom_sizes: dict[str, int],
) -> pd.DataFrame:
    """
    Convert motif occurrences to fixed-size anchor windows centered on each motif.

    Drops anchors that extend past chromosome boundaries. Deduplicates within-anchor
    overlaps (keeps highest-scoring motif per anchor center).
    """
    if len(motifs) == 0:
        return pd.DataFrame(columns=["chrom", "anchor_start", "anchor_end", "motif_score"])

    motifs = motifs.copy()
    motifs["center"] = (motifs["start"] + motifs["end"]) // 2
    motifs["anchor_start"] = (motifs["center"] - anchor_size // 2).clip(lower=0)
    motifs["anchor_end"] = motifs["anchor_start"] + anchor_size

    # Drop anchors past chrom end
    motifs["chrom_size"] = motifs["chrom"].map(chrom_sizes).fillna(0).astype(int)
    valid = motifs["anchor_end"] <= motifs["chrom_size"]
    motifs = motifs[valid].copy()
    motifs = motifs.drop(columns=["chrom_size", "center"])

    # Collapse motifs into anchors at 1 kb granularity to avoid huge nearby-motif clusters
    motifs["anchor_bin"] = motifs["anchor_start"] // 1_000
    best = motifs.sort_values("score", ascending=False).drop_duplicates(
        subset=["chrom", "anchor_bin"], keep="first"
    )
    out = best[["chrom", "anchor_start", "anchor_end", "score"]].rename(
        columns={"score": "motif_score"}
    ).reset_index(drop=True)
    return out


# ----------------------------------------------------------------------
# Pair enumeration
# ----------------------------------------------------------------------
def _enumerate_pairs_one_chrom(
    chrom: str,
    starts: np.ndarray,
    ends: np.ndarray,
    scores: np.ndarray,
    d_min: int,
    d_max: int,
) -> pd.DataFrame:
    """Vectorised pair enumeration on one chromosome for one distance band.

    For each anchor i, finds all j > i where d_min <= start_j - start_i <= d_max
    using vectorised np.searchsorted + np.repeat / cumulative-arange.  Replaces
    the previous O(N_pairs) Python double-loop.
    """
    n = len(starts)
    if n < 2:
        return pd.DataFrame()

    # For each i, the range of valid j indices is [low_i, high_i)
    lows = np.searchsorted(starts, starts + d_min, side="left")
    highs = np.searchsorted(starts, starts + d_max, side="right")

    # Restrict to j > i: j_start = max(low, i+1)
    arange_n = np.arange(n)
    j_starts = np.maximum(lows, arange_n + 1)
    j_ends = highs
    counts = np.maximum(j_ends - j_starts, 0)

    total = int(counts.sum())
    if total == 0:
        return pd.DataFrame()

    # i indices: repeat each i by counts[i]
    i_idx = np.repeat(arange_n, counts)
    # j indices: for each group of (j_start, j_end), build the arange
    # using a cumulative-offset trick (no Python loop)
    starts_per_group = np.repeat(j_starts, counts)
    cum_offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    j_idx = starts_per_group + cum_offsets

    s_i = starts[i_idx]
    s_j = starts[j_idx]
    e_i = ends[i_idx]
    e_j = ends[j_idx]
    sc_sum = scores[i_idx] + scores[j_idx]
    dist = s_j - s_i

    return pd.DataFrame({
        "chrom_i": chrom,
        "start_i": s_i,
        "end_i": e_i,
        "chrom_j": chrom,
        "start_j": s_j,
        "end_j": e_j,
        "distance": dist.astype(np.int64),
        "score_sum": sc_sum,
    })


def _enumerate_pairs_for_band(
    anchors_by_chrom: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    d_min: int,
    d_max: int,
) -> pd.DataFrame:
    """Worker: enumerate pairs for one distance band across all chromosomes."""
    parts = []
    for chrom, starts, ends, scores in anchors_by_chrom:
        df = _enumerate_pairs_one_chrom(chrom, starts, ends, scores, d_min, d_max)
        if len(df) > 0:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def enumerate_pairs_for_anchors(
    anchors: pd.DataFrame,
    distance_bands: dict[str, tuple[int, int]],
    target_per_band: int,
    rng: np.random.Generator,
    category: str,
    threads: int = 1,
) -> pd.DataFrame:
    """For a given anchor set, enumerate all anchor x anchor pairs within each
    distance band on the same chromosome, then sample ``target_per_band`` per band.

    Vectorised inner loop + parallelism across distance bands (3 bands × parallel
    chromosomes -> good speedup).
    """
    if len(anchors) < 2:
        return pd.DataFrame()

    # Pre-extract per-chromosome arrays once.
    anchors_by_chrom: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for chrom, group in anchors.groupby("chrom", sort=False):
        g = group.sort_values("anchor_start").reset_index(drop=True)
        if len(g) < 2:
            continue
        anchors_by_chrom.append((
            chrom,
            g["anchor_start"].to_numpy(),
            g["anchor_end"].to_numpy(),
            g["motif_score"].to_numpy(),
        ))

    # Parallel across distance bands (small N, but each does a chunk of work).
    band_items = list(distance_bands.items())
    if threads > 1 and len(band_items) > 1:
        band_dfs = Parallel(n_jobs=min(threads, len(band_items)))(
            delayed(_enumerate_pairs_for_band)(anchors_by_chrom, d_min, d_max)
            for _, (d_min, d_max) in band_items
        )
    else:
        band_dfs = [
            _enumerate_pairs_for_band(anchors_by_chrom, d_min, d_max)
            for _, (d_min, d_max) in band_items
        ]

    out_rows = []
    for (band_name, _), df in zip(band_items, band_dfs):
        if len(df) == 0:
            continue
        if len(df) > target_per_band:
            weights = df["score_sum"].clip(lower=0.01).to_numpy()
            weights = weights / weights.sum()
            idx = rng.choice(len(df), size=target_per_band, replace=False, p=weights)
            df = df.iloc[idx].reset_index(drop=True)
        df["distance_band"] = band_name
        df["category"] = category
        df = df.drop(columns=["score_sum"])
        out_rows.append(df)

    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True)


# ----------------------------------------------------------------------
# Pseudobulk-contact-derived categories: co_folded / random_matched
# ----------------------------------------------------------------------
def build_co_folded_pairs(
    pseudobulk_cool: str,
    distance_bands: dict[str, tuple[int, int]],
    anchor_size: int,
    target_per_band: int,
    rng: np.random.Generator,
    chrom_sizes: dict[str, int],
) -> pd.DataFrame:
    """
    Build co_folded pairs from top-decile pseudobulk contact frequencies, per distance band.

    Uses cooler to fetch contact matrices at the cooler's native resolution.
    """
    try:
        import cooler
    except ImportError:
        log.warning("cooler not installed; skipping co_folded category. pip install cooler")
        return pd.DataFrame()

    c = cooler.Cooler(pseudobulk_cool)
    resolution = c.binsize
    log.info(f"  pseudobulk cooler resolution: {resolution:,} bp")

    out_rows = []
    for band_name, (d_min, d_max) in distance_bands.items():
        log.info(f"  co_folded selection for band={band_name} ({d_min:,}–{d_max:,} bp)")
        per_chrom = []
        for chrom in c.chromnames:
            if chrom not in chrom_sizes:
                continue
            try:
                mat = c.matrix(balance=False, sparse=True).fetch(chrom).tocoo()
            except Exception:
                continue
            if mat.nnz == 0:
                continue
            # mat.row, mat.col are bin indices on this chromosome
            offsets = c.offset(chrom)
            bin_starts = (mat.row + 0) * resolution  # within-chrom bin start coords
            bin_starts_j = (mat.col + 0) * resolution
            distances = np.abs(bin_starts_j - bin_starts)
            counts = mat.data
            mask = (distances >= d_min) & (distances <= d_max) & (mat.col > mat.row)
            if not mask.any():
                continue
            bs_i = bin_starts[mask]
            bs_j = bin_starts_j[mask]
            ct = counts[mask]
            # Top decile by raw count
            threshold = np.quantile(ct, 0.9)
            keep = ct >= threshold
            for s_i, s_j in zip(bs_i[keep], bs_j[keep]):
                center_i = s_i + resolution // 2
                center_j = s_j + resolution // 2
                anchor_i_start = max(0, center_i - anchor_size // 2)
                anchor_j_start = max(0, center_j - anchor_size // 2)
                anchor_i_end = anchor_i_start + anchor_size
                anchor_j_end = anchor_j_start + anchor_size
                if anchor_j_end > chrom_sizes[chrom]:
                    continue
                per_chrom.append((
                    chrom, anchor_i_start, anchor_i_end,
                    chrom, anchor_j_start, anchor_j_end,
                    int(anchor_j_start - anchor_i_start),
                ))

        if not per_chrom:
            continue
        df = pd.DataFrame(per_chrom, columns=[
            "chrom_i", "start_i", "end_i", "chrom_j", "start_j", "end_j", "distance",
        ])
        if len(df) > target_per_band:
            idx = rng.choice(len(df), size=target_per_band, replace=False)
            df = df.iloc[idx].reset_index(drop=True)
        df["distance_band"] = band_name
        df["category"] = "co_folded"
        out_rows.append(df)

    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True)


def build_random_matched_pairs(
    distance_bands: dict[str, tuple[int, int]],
    chrom_sizes: dict[str, int],
    anchor_size: int,
    target_per_band: int,
    rng: np.random.Generator,
    blacklist_bed: Optional[str] = None,
    chromosomes: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Generate random distance-matched control pairs.
    Optionally avoids blacklist regions (ENCODE).
    """
    if chromosomes is None:
        chromosomes = list(chrom_sizes.keys())
    chrom_sizes_sub = {c: chrom_sizes[c] for c in chromosomes if c in chrom_sizes}

    blacklist = None
    if blacklist_bed and Path(blacklist_bed).exists():
        blacklist = pr.read_bed(blacklist_bed)

    out_rows = []
    for band_name, (d_min, d_max) in distance_bands.items():
        log.info(f"  random_matched selection for band={band_name}")
        sampled = 0
        attempts = 0
        max_attempts = target_per_band * 20
        rows = []
        chroms = list(chrom_sizes_sub.keys())
        chrom_weights = np.array([chrom_sizes_sub[c] for c in chroms], dtype=float)
        chrom_weights /= chrom_weights.sum()
        while sampled < target_per_band and attempts < max_attempts:
            chrom = rng.choice(chroms, p=chrom_weights)
            size = chrom_sizes_sub[chrom]
            if size < d_max + anchor_size:
                attempts += 1
                continue
            d = int(rng.integers(d_min, d_max + 1))
            anchor_i_start = int(rng.integers(0, size - d - anchor_size))
            anchor_j_start = anchor_i_start + d
            anchor_i_end = anchor_i_start + anchor_size
            anchor_j_end = anchor_j_start + anchor_size
            if anchor_j_end > size:
                attempts += 1
                continue
            if blacklist is not None:
                probe = pr.PyRanges(pd.DataFrame({
                    "Chromosome": [chrom, chrom],
                    "Start": [anchor_i_start, anchor_j_start],
                    "End": [anchor_i_end, anchor_j_end],
                }))
                if len(probe.overlap(blacklist).df) > 0:
                    attempts += 1
                    continue
            rows.append((chrom, anchor_i_start, anchor_i_end,
                         chrom, anchor_j_start, anchor_j_end, d))
            sampled += 1
            attempts += 1

        if not rows:
            continue
        df = pd.DataFrame(rows, columns=[
            "chrom_i", "start_i", "end_i", "chrom_j", "start_j", "end_j", "distance",
        ])
        df["distance_band"] = band_name
        df["category"] = "random_distance_matched"
        out_rows.append(df)

    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True)


# ----------------------------------------------------------------------
# Final assembly
# ----------------------------------------------------------------------
def assemble_panel(
    category_dfs: list[pd.DataFrame],
    tier_name: str = "tier1",
) -> pd.DataFrame:
    """Concatenate, deduplicate, assign pair_ids, write parquet+TSV."""
    if not category_dfs:
        return pd.DataFrame()
    panel = pd.concat([d for d in category_dfs if len(d) > 0], ignore_index=True)

    # Deduplicate by (chrom_i, start_i, chrom_j, start_j); keep first category encountered
    before = len(panel)
    panel = panel.drop_duplicates(subset=["chrom_i", "start_i", "chrom_j", "start_j"], keep="first")
    if len(panel) < before:
        log.info(f"  deduplicated {before - len(panel)} pairs across categories")

    # Stable pair IDs derived from category + chrom + positions
    def _mkid(row):
        h = sha256(f"{row['category']}|{row['chrom_i']}|{row['start_i']}|"
                   f"{row['chrom_j']}|{row['start_j']}".encode()).hexdigest()[:10]
        return f"{tier_name}_{row['category']}_{h}"

    panel["pair_id"] = panel.apply(_mkid, axis=1)
    panel["tier"] = tier_name

    # Reorder columns
    cols = ["pair_id", "chrom_i", "start_i", "end_i",
            "chrom_j", "start_j", "end_j", "distance",
            "tier", "category", "distance_band"]
    return panel[cols]


def read_chrom_sizes(path: Path) -> dict[str, int]:
    """Read a tab-separated chrom.sizes file."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            out[parts[0]] = int(parts[1])
    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build a multi-category TF-anchored pair panel BEDPE",
    )
    parser.add_argument("--config", required=True, help="YAML config with TF categories")
    parser.add_argument("--genome-fa", required=True, help="Indexed reference genome FASTA")
    parser.add_argument("--chrom-sizes", required=True, help="chrom.sizes file")
    parser.add_argument("--pseudobulk-cool", default=None,
                        help="cooler URI (file.mcool::resolutions/N) for co_folded category")
    parser.add_argument("--blacklist-bed", default=None,
                        help="ENCODE blacklist BED to avoid in random_matched")
    parser.add_argument("--jaspar-cache", default="./jaspar_cache",
                        help="Directory to cache JASPAR PFMs")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jaspar_cache = Path(args.jaspar_cache)
    jaspar_cache.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(args.config) as f:
        raw = yaml.safe_load(f)
    tf_categories = [TFCategoryConfig(**c) for c in raw.get("tf_categories", [])]
    cfg = PanelConfig(
        tf_categories=tf_categories,
        distance_bands={k: tuple(v) for k, v in raw.get("distance_bands", {
            "short": [100_000, 500_000],
            "mid": [500_000, 2_000_000],
            "long": [2_000_000, 5_000_000],
        }).items()},
        anchor_size=raw.get("anchor_size", 5_000),
        random_seed=raw.get("random_seed", 42),
        include_co_folded=raw.get("include_co_folded", True),
        include_random_matched=raw.get("include_random_matched", True),
        target_co_folded_per_band=raw.get("target_co_folded_per_band", 3_000),
        target_random_per_band=raw.get("target_random_per_band", 3_000),
        chromosomes=raw.get("chromosomes", None),
    )

    chrom_sizes = read_chrom_sizes(Path(args.chrom_sizes))
    chromosomes = cfg.chromosomes or [c for c in chrom_sizes if "_" not in c and c not in ("chrM", "chrY")]
    log.info(f"Scanning {len(chromosomes)} chromosomes: {chromosomes[:5]}...")

    rng = np.random.default_rng(cfg.random_seed)

    # Per-TF category build
    category_dfs = []
    for tf_cfg in cfg.tf_categories:
        log.info(f"=== {tf_cfg.category} (motifs: {tf_cfg.jaspar_ids}) ===")
        pfm_paths = [fetch_jaspar_pfm(mid, jaspar_cache) for mid in tf_cfg.jaspar_ids]
        motifs = scan_genome_for_motifs(
            Path(args.genome_fa), pfm_paths, chromosomes,
            tf_cfg.pseudo_pvalue_threshold, args.threads,
        )
        if tf_cfg.chip_bed:
            motifs = intersect_motifs_with_chip(motifs, tf_cfg.chip_bed)
        anchors = motifs_to_anchors(motifs, cfg.anchor_size, chrom_sizes)
        log.info(f"  {len(anchors):,} distinct anchors after collapsing")
        pairs = enumerate_pairs_for_anchors(
            anchors, cfg.distance_bands, tf_cfg.target_pairs_per_band,
            rng, tf_cfg.category,
            threads=args.threads,
        )
        log.info(f"  {len(pairs):,} pairs across {len(cfg.distance_bands)} bands")
        category_dfs.append(pairs)

    # Co-folded category
    if cfg.include_co_folded and args.pseudobulk_cool:
        log.info("=== co_folded (pseudobulk top decile) ===")
        co_pairs = build_co_folded_pairs(
            args.pseudobulk_cool, cfg.distance_bands, cfg.anchor_size,
            cfg.target_co_folded_per_band, rng, chrom_sizes,
        )
        log.info(f"  {len(co_pairs):,} co_folded pairs")
        category_dfs.append(co_pairs)

    # Random distance-matched
    if cfg.include_random_matched:
        log.info("=== random_distance_matched ===")
        rand_pairs = build_random_matched_pairs(
            cfg.distance_bands, chrom_sizes, cfg.anchor_size,
            cfg.target_random_per_band, rng, args.blacklist_bed, chromosomes,
        )
        log.info(f"  {len(rand_pairs):,} random_matched pairs")
        category_dfs.append(rand_pairs)

    # Assemble and write
    panel = assemble_panel(category_dfs, tier_name="tier1")
    log.info(f"\nFinal panel: {len(panel):,} pairs across {panel['category'].nunique()} categories")
    log.info(panel.groupby(["category", "distance_band"]).size().to_string())

    parquet_path = out_dir / "pair_panel.parquet"
    bedpe_path = out_dir / "pair_panel.bedpe"
    panel.to_parquet(parquet_path, index=False)
    panel.to_csv(bedpe_path, sep="\t", index=False)
    log.info(f"\nWrote {parquet_path}")
    log.info(f"Wrote {bedpe_path}")


if __name__ == "__main__":
    main()
