"""Pair panel construction: BEDPE-style tables of region pairs to analyse.

A "pair panel" defines pairs of genomic regions (anchors) for concordance
analysis.  Each pair has two 5 kb anchors, a category label, and a tier.

**Tier 1** (TF-driven) categories:

* **tf_anchored** — both anchors centred on TF motif instances loaded
  from BED files.
* **random_distance_matched** — random pairs of anchors drawn at matched
  genomic distances.

**Tier 2** (unbiased multi-scale) categories:

* **unbiased_compartment** — random bin-pairs at compartment scale (~1 Mb).
* **unbiased_domain** — random bin-pairs at TAD/domain scale (~250 kb).
* **unbiased_loop** — random bin-pairs at loop scale (~25 kb).

Pairs are stratified into distance bands (short / mid / long) so that
downstream analyses can compare TF-anchored pairs to distance-matched
controls within the same distance range.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..utils import chrom_is_autosomal_or_sex, ensure_dir

logger = logging.getLogger(__name__)

PANEL_COLUMNS: list[str] = [
    "pair_id",
    "chrom_i", "start_i", "end_i",
    "chrom_j", "start_j", "end_j",
    "distance",
    "category",
    "distance_band",
    "tier",
]

DEFAULT_DISTANCE_BANDS: dict[str, tuple[int, int]] = {
    "short": (100_000, 500_000),
    "mid": (500_000, 2_000_000),
    "long": (2_000_000, 5_000_000),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_panel(path: str) -> pd.DataFrame:
    """Load a pair panel from a TSV/BEDPE file.

    Expected columns: pair_id, chrom_i, start_i, end_i, chrom_j, start_j,
    end_j, distance, category, distance_band.  The ``tier`` column is
    optional — if absent, all rows default to ``"tier1"``.

    Parameters
    ----------
    path : str
        Path to a tab-separated panel file (with header row).

    Returns
    -------
    pd.DataFrame
        Panel with the standard ``PANEL_COLUMNS``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns are missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Panel file not found: {path}")

    df = pd.read_csv(path, sep="\t")

    # ``tier`` is optional; default to "tier1" for legacy panels.
    required = set(PANEL_COLUMNS) - {"tier"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Panel file is missing required columns: {sorted(missing)}"
        )

    if "tier" not in df.columns:
        df["tier"] = "tier1"

    # Enforce integer types for coordinate columns.
    for col in ("start_i", "end_i", "start_j", "end_j", "distance"):
        df[col] = df[col].astype(np.int64)

    return df[PANEL_COLUMNS]


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def _load_motif_bed(bed_path: str) -> pd.DataFrame:
    """Load a motif BED file (BED6: chrom, start, end, name, score, strand).

    Returns a DataFrame with columns ``chrom, start, end``.  Only autosomal
    and sex chromosomes are kept.
    """
    cols = ["chrom", "start", "end", "name", "score", "strand"]
    try:
        df = pd.read_csv(bed_path, sep="\t", header=None, names=cols, comment="#")
    except Exception:
        # Fall back to fewer columns if BED has <6 fields.
        df = pd.read_csv(
            bed_path, sep="\t", header=None,
            names=["chrom", "start", "end"], usecols=[0, 1, 2], comment="#",
        )

    df = df[df["chrom"].apply(chrom_is_autosomal_or_sex)].copy()
    df["start"] = df["start"].astype(np.int64)
    df["end"] = df["end"].astype(np.int64)
    return df[["chrom", "start", "end"]]


def _sample_motif_pairs(
    motifs: pd.DataFrame,
    anchor_size: int,
    band_name: str,
    band_min: int,
    band_max: int,
    n_pairs: int,
    rng: np.random.Generator,
    chrom_sizes: dict[str, int],
    category: str,
) -> list[dict]:
    """Sample pairs of motif sites on the same chromosome within a distance band.

    Each anchor is an ``anchor_size``-wide window centred on the motif midpoint.
    Pairs whose anchors extend beyond chromosome bounds are discarded.
    """
    rows: list[dict] = []

    per_chrom: dict[str, np.ndarray] = {}
    for chrom, grp in motifs.groupby("chrom"):
        midpoints = ((grp["start"].values + grp["end"].values) // 2).astype(np.int64)
        midpoints.sort()
        per_chrom[chrom] = midpoints

    # Collect all valid candidate pairs across chromosomes, then sample.
    candidate_chroms: list[str] = []
    candidate_idx_i: list[int] = []
    candidate_idx_j: list[int] = []

    half = anchor_size // 2
    for chrom, mids in per_chrom.items():
        chrom_len = chrom_sizes.get(chrom)
        if chrom_len is None or len(mids) < 2:
            continue
        for a in range(len(mids)):
            # Binary-search for the range of valid partners.
            lo = np.searchsorted(mids, mids[a] + band_min, side="left")
            hi = np.searchsorted(mids, mids[a] + band_max, side="left")
            for b in range(lo, hi):
                # Bounds check for both anchors.
                if mids[a] - half < 0 or mids[a] + half > chrom_len:
                    continue
                if mids[b] - half < 0 or mids[b] + half > chrom_len:
                    continue
                candidate_chroms.append(chrom)
                candidate_idx_i.append(a)
                candidate_idx_j.append(b)

    n_candidates = len(candidate_chroms)
    if n_candidates == 0:
        logger.warning(
            f"No valid {category} motif pairs in band {band_name} "
            f"({band_min:,}-{band_max:,} bp)"
        )
        return rows

    n_sample = min(n_pairs, n_candidates)
    chosen = rng.choice(n_candidates, size=n_sample, replace=False)

    for idx in chosen:
        chrom = candidate_chroms[idx]
        mids = per_chrom[chrom]
        mid_i = int(mids[candidate_idx_i[idx]])
        mid_j = int(mids[candidate_idx_j[idx]])
        start_i = mid_i - half
        start_j = mid_j - half
        dist = abs(mid_j - mid_i)
        rows.append({
            "chrom_i": chrom,
            "start_i": start_i,
            "end_i": start_i + anchor_size,
            "chrom_j": chrom,
            "start_j": start_j,
            "end_j": start_j + anchor_size,
            "distance": dist,
            "category": category,
            "distance_band": band_name,
        })

    return rows


def _sample_random_pairs(
    chrom_sizes: dict[str, int],
    anchor_size: int,
    band_name: str,
    band_min: int,
    band_max: int,
    n_pairs: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Sample random pairs of anchors at distances within a band.

    Anchors are placed uniformly across autosomal/sex chromosomes, weighted
    by effective chromosome length (chrom_len - anchor_size - band_max).
    """
    rows: list[dict] = []

    # Build a weighted chromosome list for sampling.
    valid_chroms: list[str] = []
    weights: list[float] = []
    for chrom, length in chrom_sizes.items():
        if not chrom_is_autosomal_or_sex(chrom):
            continue
        effective = length - anchor_size - band_max
        if effective > 0:
            valid_chroms.append(chrom)
            weights.append(float(effective))

    if not valid_chroms:
        logger.warning(
            f"No chromosomes large enough for random pairs in band "
            f"{band_name} ({band_min:,}-{band_max:,} bp)"
        )
        return rows

    weight_arr = np.array(weights)
    weight_arr /= weight_arr.sum()

    # Over-sample to account for boundary rejections, then trim.
    n_attempts = n_pairs * 2
    sampled_chroms = rng.choice(valid_chroms, size=n_attempts, p=weight_arr)

    half = anchor_size // 2
    for chrom in sampled_chroms:
        if len(rows) >= n_pairs:
            break
        chrom_len = chrom_sizes[chrom]
        max_start = chrom_len - anchor_size
        if max_start <= 0:
            continue

        # Place anchor_i randomly.
        mid_i = int(rng.integers(half, chrom_len - half))

        # Sample distance within the band.
        dist = int(rng.integers(band_min, band_max))

        # Randomly pick direction.
        if rng.random() < 0.5:
            mid_j = mid_i + dist
        else:
            mid_j = mid_i - dist

        # Ensure anchor_j stays in bounds.
        if mid_j - half < 0 or mid_j + half > chrom_len:
            continue

        # Canonical order: i before j.
        if mid_i > mid_j:
            mid_i, mid_j = mid_j, mid_i

        start_i = mid_i - half
        start_j = mid_j - half
        rows.append({
            "chrom_i": chrom,
            "start_i": start_i,
            "end_i": start_i + anchor_size,
            "chrom_j": chrom,
            "start_j": start_j,
            "end_j": start_j + anchor_size,
            "distance": abs(mid_j - mid_i),
            "category": "random_distance_matched",
            "distance_band": band_name,
        })

    if len(rows) < n_pairs:
        logger.warning(
            f"Only sampled {len(rows)}/{n_pairs} random pairs in band "
            f"{band_name} (chromosome bounds limited available positions)"
        )

    return rows


def build_panel(
    chrom_sizes: dict[str, int],
    tf_motif_beds: dict[str, str] | None = None,
    anchor_size: int = 5_000,
    distance_bands: dict[str, tuple[int, int]] | None = None,
    n_per_category_per_band: int = 5_000,
    output_path: str | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build a pair panel of genomic region pairs for concordance analysis.

    If *tf_motif_beds* is provided, creates ``tf_anchored`` pairs where both
    anchors are centred on TF motif instances.  Also creates
    ``random_distance_matched`` control pairs at matched distances.

    Parameters
    ----------
    chrom_sizes : dict
        Mapping ``chrom -> length`` (bp).
    tf_motif_beds : dict, optional
        Mapping ``motif_name -> BED_path``.  Each BED should be BED6 format
        (chrom, start, end, name, score, strand).  If ``None``, only random
        pairs are generated.
    anchor_size : int
        Width of each anchor region in bp (default 5000).
    distance_bands : dict, optional
        Mapping ``band_name -> (min_distance, max_distance)``.  Defaults to
        short (100 kb -- 500 kb), mid (500 kb -- 2 Mb), long (2 Mb -- 5 Mb).
    n_per_category_per_band : int
        Target number of pairs per category per distance band.
    output_path : str, optional
        If given, write the panel as a TSV file at this path.
    random_state : int
        Seed for the random number generator (default 42).

    Returns
    -------
    pd.DataFrame
        Panel with columns: pair_id, chrom_i, start_i, end_i, chrom_j,
        start_j, end_j, distance, category, distance_band.
    """
    if distance_bands is None:
        distance_bands = DEFAULT_DISTANCE_BANDS

    rng = np.random.default_rng(random_state)
    all_rows: list[dict] = []

    # --- TF-anchored pairs ---
    if tf_motif_beds is not None:
        for motif_name, bed_path in tf_motif_beds.items():
            logger.info(f"Loading TF motif BED: {motif_name} from {bed_path}")
            motifs = _load_motif_bed(bed_path)
            logger.info(f"  {len(motifs):,} motif sites loaded")

            category = f"tf_anchored_{motif_name}"
            for band_name, (band_min, band_max) in distance_bands.items():
                pairs = _sample_motif_pairs(
                    motifs, anchor_size, band_name, band_min, band_max,
                    n_per_category_per_band, rng, chrom_sizes, category,
                )
                logger.info(
                    f"  {motif_name} / {band_name}: {len(pairs):,} pairs"
                )
                all_rows.extend(pairs)

    # --- Random distance-matched pairs ---
    for band_name, (band_min, band_max) in distance_bands.items():
        pairs = _sample_random_pairs(
            chrom_sizes, anchor_size, band_name, band_min, band_max,
            n_per_category_per_band, rng,
        )
        logger.info(
            f"  random_distance_matched / {band_name}: {len(pairs):,} pairs"
        )
        all_rows.extend(pairs)

    if not all_rows:
        logger.warning("Panel is empty — no pairs were generated")
        return pd.DataFrame(columns=PANEL_COLUMNS)

    df = pd.DataFrame(all_rows)

    # All pairs from build_panel are Tier 1.
    df["tier"] = "tier1"

    # Assign pair IDs with T1_ prefix.
    df.insert(0, "pair_id", [f"T1_pair_{i:06d}" for i in range(len(df))])

    # Enforce column order and types.
    for col in ("start_i", "end_i", "start_j", "end_j", "distance"):
        df[col] = df[col].astype(np.int64)
    df = df[PANEL_COLUMNS]

    logger.info(
        f"Panel built: {len(df):,} pairs across "
        f"{df['category'].nunique()} categories and "
        f"{df['distance_band'].nunique()} distance bands"
    )

    if output_path is not None:
        out_dir = Path(output_path).parent
        ensure_dir(str(out_dir))
        df.to_csv(output_path, sep="\t", index=False)
        logger.info(f"Panel written to {output_path}")

    return df


# ---------------------------------------------------------------------------
# Tier 2: unbiased multi-scale panel
# ---------------------------------------------------------------------------

DEFAULT_TIER2_SCALES: dict[str, dict] = {
    "compartment": {
        "bin_size": 1_000_000,
        "distance_range": (5_000_000, 50_000_000),
    },
    "domain": {
        "bin_size": 250_000,
        "distance_range": (500_000, 5_000_000),
    },
    "loop": {
        "bin_size": 25_000,
        "distance_range": (100_000, 2_000_000),
    },
}


def _distance_to_band(distance: int) -> str:
    """Assign a distance band label based on absolute distance."""
    if distance < 500_000:
        return "short"
    elif distance < 2_000_000:
        return "mid"
    else:
        return "long"


def build_tier2_multiscale_panel(
    chrom_sizes: dict[str, int],
    scales: dict[str, dict] | None = None,
    n_per_scale: int = 10_000,
    output_path: str | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build Tier 2 unbiased multi-scale panel.

    For each scale, tiles the genome into non-overlapping bins of the
    scale's bin_size, then randomly samples pairs of bins on the same
    chromosome within the scale's distance_range.

    Categories: unbiased_compartment, unbiased_domain, unbiased_loop.
    All rows have tier="tier2".

    Parameters
    ----------
    chrom_sizes : dict
        Mapping ``chrom -> length`` (bp).
    scales : dict, optional
        Mapping ``scale_name -> {"bin_size": int, "distance_range": (lo, hi)}``.
        Defaults to compartment (1 Mb), domain (250 kb), loop (25 kb).
    n_per_scale : int
        Target number of pairs per scale (default 10 000).
    output_path : str, optional
        If given, write the panel as a TSV file at this path.
    random_state : int
        Seed for the random number generator (default 42).

    Returns
    -------
    pd.DataFrame
        Panel with columns matching ``PANEL_COLUMNS``.
    """
    if scales is None:
        scales = DEFAULT_TIER2_SCALES

    rng = np.random.default_rng(random_state)
    all_rows: list[dict] = []

    for scale_name, cfg in scales.items():
        bin_size: int = cfg["bin_size"]
        dist_lo, dist_hi = cfg["distance_range"]
        category = f"unbiased_{scale_name}"

        # Tile each chromosome into non-overlapping bins.
        # Key: chrom -> array of bin start positions.
        chrom_bins: dict[str, np.ndarray] = {}
        for chrom, length in chrom_sizes.items():
            if not chrom_is_autosomal_or_sex(chrom):
                continue
            n_bins = length // bin_size
            if n_bins < 2:
                continue
            chrom_bins[chrom] = np.arange(n_bins) * bin_size

        # Build candidate pool: for every chromosome enumerate valid
        # bin-pair indices (i, j) where j > i and distance is in range.
        # Distance is measured centre-to-centre.
        cand_chroms: list[str] = []
        cand_i: list[int] = []
        cand_j: list[int] = []

        for chrom, bins in chrom_bins.items():
            centres = bins + bin_size // 2
            n = len(centres)
            # Minimum / maximum index offset that satisfies distance range.
            min_offset = max(1, int(np.ceil(dist_lo / bin_size)))
            max_offset = int(np.floor(dist_hi / bin_size))
            for a in range(n):
                lo_j = a + min_offset
                hi_j = min(a + max_offset + 1, n)
                for b in range(lo_j, hi_j):
                    d = centres[b] - centres[a]
                    if dist_lo <= d <= dist_hi:
                        cand_chroms.append(chrom)
                        cand_i.append(a)
                        cand_j.append(b)

        n_candidates = len(cand_chroms)
        if n_candidates == 0:
            logger.warning(
                f"No valid bin-pairs for scale {scale_name} "
                f"(bin_size={bin_size:,}, distance_range="
                f"{dist_lo:,}-{dist_hi:,})"
            )
            continue

        n_sample = min(n_per_scale, n_candidates)
        chosen = rng.choice(n_candidates, size=n_sample, replace=False)

        for idx in chosen:
            chrom = cand_chroms[idx]
            bins = chrom_bins[chrom]
            start_i = int(bins[cand_i[idx]])
            start_j = int(bins[cand_j[idx]])
            centre_i = start_i + bin_size // 2
            centre_j = start_j + bin_size // 2
            dist = centre_j - centre_i

            all_rows.append({
                "chrom_i": chrom,
                "start_i": start_i,
                "end_i": start_i + bin_size,
                "chrom_j": chrom,
                "start_j": start_j,
                "end_j": start_j + bin_size,
                "distance": dist,
                "category": category,
                "distance_band": _distance_to_band(dist),
            })

        logger.info(
            f"  {category}: {min(n_sample, len(chosen)):,} pairs sampled "
            f"from {n_candidates:,} candidates"
        )

    if not all_rows:
        logger.warning("Tier 2 panel is empty — no pairs were generated")
        return pd.DataFrame(columns=PANEL_COLUMNS)

    df = pd.DataFrame(all_rows)
    df["tier"] = "tier2"
    df.insert(0, "pair_id", [f"T2_pair_{i:06d}" for i in range(len(df))])

    for col in ("start_i", "end_i", "start_j", "end_j", "distance"):
        df[col] = df[col].astype(np.int64)
    df = df[PANEL_COLUMNS]

    logger.info(
        f"Tier 2 panel built: {len(df):,} pairs across "
        f"{df['category'].nunique()} categories and "
        f"{df['distance_band'].nunique()} distance bands"
    )

    if output_path is not None:
        out_dir = Path(output_path).parent
        ensure_dir(str(out_dir))
        df.to_csv(output_path, sep="\t", index=False)
        logger.info(f"Tier 2 panel written to {output_path}")

    return df


# ---------------------------------------------------------------------------
# Combined panel
# ---------------------------------------------------------------------------


def build_combined_panel(
    tier1_panel: pd.DataFrame | None,
    tier2_panel: pd.DataFrame | None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Concatenate Tier 1 and Tier 2 panels with disjoint pair_id namespaces.

    Parameters
    ----------
    tier1_panel : pd.DataFrame or None
        Tier 1 panel (from :func:`build_panel`).
    tier2_panel : pd.DataFrame or None
        Tier 2 panel (from :func:`build_tier2_multiscale_panel`).
    output_path : str, optional
        If given, write the combined panel as a TSV file at this path.

    Returns
    -------
    pd.DataFrame
        Combined panel with columns matching ``PANEL_COLUMNS``.
    """
    parts: list[pd.DataFrame] = []
    if tier1_panel is not None and len(tier1_panel) > 0:
        parts.append(tier1_panel)
    if tier2_panel is not None and len(tier2_panel) > 0:
        parts.append(tier2_panel)

    if not parts:
        logger.warning("Combined panel is empty — no input panels provided")
        return pd.DataFrame(columns=PANEL_COLUMNS)

    df = pd.concat(parts, ignore_index=True)

    # Ensure no duplicate pair_ids.
    if df["pair_id"].duplicated().any():
        logger.warning(
            "Duplicate pair_ids detected across tiers; re-numbering within "
            "each tier to restore disjoint namespaces"
        )
        new_ids: list[str] = []
        for tier_label, grp in df.groupby("tier", sort=False):
            prefix = "T1_" if tier_label == "tier1" else "T2_"
            new_ids.extend(
                [f"{prefix}pair_{i:06d}" for i in range(len(grp))]
            )
        df["pair_id"] = new_ids

    # Ensure standard column set and order.
    for col in PANEL_COLUMNS:
        if col not in df.columns:
            df[col] = "tier1" if col == "tier" else ""
    df = df[PANEL_COLUMNS]

    logger.info(
        f"Combined panel: {len(df):,} pairs "
        f"(tier1={len(df[df['tier'] == 'tier1']):,}, "
        f"tier2={len(df[df['tier'] == 'tier2']):,})"
    )

    if output_path is not None:
        out_dir = Path(output_path).parent
        ensure_dir(str(out_dir))
        df.to_csv(output_path, sep="\t", index=False)
        logger.info(f"Combined panel written to {output_path}")

    return df
