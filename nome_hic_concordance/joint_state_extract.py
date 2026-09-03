"""Part A record assembly: BAM -> the design A.1 per-read-end joint table.

Thin shaping layer over ``_joint_state_core.parse_cell_joint_ends``.  The
counting itself is in C++ because it is per-read; everything here is per-cell
bookkeeping, so it stays in Python where it can be read.

The one judgement encoded here is **permissive emission** (design A.1): an end
is written whatever its site counts, including zero of both marks.  Parts B and
C decide what to require.  Filtering at extraction time would make the
denominator -- how many ends were seen at all -- unrecoverable without another
pass over the BAM.
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from . import _joint_state_core  # type: ignore[attr-defined]
    _CPP_AVAILABLE = True
except ImportError:                                        # pragma: no cover
    _CPP_AVAILABLE = False

#: Marks extracted by default.  HCH is not passed as a context -- it is derived
#: from each read's own reference span (see joint_state_core.cpp) -- so it is
#: absent here and present in the output.
DEFAULT_MARKS = ("GCH", "HCG")

STRAND_NAMES = {0: "unknown", 1: "CT", 2: "GA"}


def extract_cell(
    bam_path: str,
    context_positions: Mapping[str, Mapping[str, np.ndarray]],
    cell_id: str,
    marks: Sequence[str] = DEFAULT_MARKS,
    snp_positions: Mapping[str, np.ndarray] | None = None,
    trim_ends: int = 5,
    mapq_min: int = 30,
    baseq_min: int = 0,
    strand_min_evidence: int = 1,
    keep_supplementary: bool = False,
    min_distance: int = 1000,
    cis_only: bool = True,
    count_hch: bool = True,
) -> pd.DataFrame:
    """One cell's BAM to the A.1 record, one row per read end.

    Parameters
    ----------
    context_positions : ``{mark: {chrom: structured array}}``
        As produced by :func:`~.reference.extract_context_positions` or
        :func:`~.reference.load_context_from_bed`; the structured arrays carry
        ``pos`` (int64) and ``ref_base`` (uint8).
    marks : sequence of str
        Which keys of *context_positions* to score, in output column order.
    snp_positions : ``{chrom: int64 array}``, optional
        Counted per end as ``n_masked_snp``.  These are positions ALREADY
        removed from the context arrays upstream; the count is what lets a
        downstream step tell a low site count from a masked one.

    Returns
    -------
    DataFrame with the design A.1 columns.  Empty (with the right columns and
    dtypes) when the BAM yields no pair.
    """
    if not _CPP_AVAILABLE:                                 # pragma: no cover
        raise RuntimeError(
            "_joint_state_core is not built.  Run 'pip install -e .'; if the "
            "build reports that the extension links but does not load, the "
            "installed htslib does not match this interpreter's architecture."
        )

    missing = [m for m in marks if m not in context_positions]
    if missing:
        raise KeyError(f"context_positions has no entry for {missing}; "
                       f"available: {sorted(context_positions)}")

    contexts = [
        {c: (np.ascontiguousarray(a["pos"], dtype=np.int64),
             np.ascontiguousarray(a["ref_base"], dtype=np.uint8))
         for c, a in context_positions[m].items() if len(a) > 0}
        for m in marks
    ]
    snp = {c: np.ascontiguousarray(v, dtype=np.int64)
           for c, v in (snp_positions or {}).items() if len(v) > 0}

    raw = _joint_state_core.parse_cell_joint_ends(
        bam_path, contexts, snp,
        trim_ends=trim_ends, mapq_min=mapq_min, baseq_min=baseq_min,
        strand_min_evidence=strand_min_evidence,
        keep_supplementary=keep_supplementary,
        count_hch=count_hch, emit_read_id=True,
        min_distance=min_distance, cis_only=cis_only,
    )

    stats = dict(raw["stats"])
    logger.info(
        "  %s: %d pairs (%d reads kept, %d unpaired, %d below min-distance, "
        "%d strand-unknown, %d without MD); peak buffer %d reads",
        cell_id, stats["n_pairs"], stats["n_reads_kept"], stats["n_unpaired"],
        stats["n_below_min_distance"], stats["n_strand_unknown"],
        stats["n_no_md"], stats["peak_buffered_reads"],
    )
    if stats["n_no_md"]:
        logger.warning(
            "  %s: %d reads carry no MD tag and were scored as zero sites. "
            "Run 'samtools calmd' -- without MD neither the conversion strand "
            "nor the HCH control can be derived.", cell_id, stats["n_no_md"])

    names = list(raw["chrom_names"])
    chrom_idx = np.asarray(raw["chrom"])
    cols = {
        "read_id": list(raw["read_id"]),
        "cell_id": cell_id,
        "end_idx": np.asarray(raw["end_idx"], dtype=np.int8),
        "chrom": pd.Categorical.from_codes(
            chrom_idx, categories=names) if len(chrom_idx) else
            pd.Categorical([], categories=names),
        "pos_5p": np.asarray(raw["pos_5p"], dtype=np.int64),
        "pos_3p": np.asarray(raw["pos_3p"], dtype=np.int64),
        "strand": np.where(np.asarray(raw["is_reverse"]).astype(bool), "-", "+"),
        "conv_strand": pd.Categorical(
            [STRAND_NAMES[int(v)] for v in np.asarray(raw["conv_strand"])],
            categories=list(STRAND_NAMES.values())),
        "mapq": np.asarray(raw["mapq"], dtype=np.int16),
    }
    for i, mark in enumerate(marks):
        key = mark.lower()
        cols[f"n_{key}"] = np.asarray(raw["n"][i], dtype=np.int32)
        cols[f"k_{key}"] = np.asarray(raw["k"][i], dtype=np.int32)
    cols["n_hch"] = np.asarray(raw["n_hch"], dtype=np.int32)
    cols["k_hch"] = np.asarray(raw["k_hch"], dtype=np.int32)
    cols["n_masked_snp"] = np.asarray(raw["n_masked_snp"], dtype=np.int32)
    cols["frag_len"] = np.asarray(raw["frag_len"], dtype=np.int32)
    cols["contact_dist"] = np.asarray(raw["contact_dist"], dtype=np.int64)
    cols["is_cis"] = np.asarray(raw["is_cis"]).astype(bool)
    cols["pair_idx"] = np.asarray(raw["pair_idx"], dtype=np.int64)

    df = pd.DataFrame(cols)
    df.attrs["stats"] = stats
    return df


def conversion_rate(df: pd.DataFrame, level: str = "molecule") -> pd.Series:
    """Per-end or per-molecule conversion rate from the HCH control (A.3.5).

    ``k_hch`` counts UNCONVERTED cytosines, so the rate is ``1 - k/n``.

    Incomplete conversion inflates apparent methylation in both contexts at
    both ends of the same molecule.  That is a molecule-level artifact, so it
    survives within-cell mate shuffling and the Part B null cannot see it --
    which is why ``level='molecule'`` (both ends pooled) is the default and the
    one to filter on.  Ends with no HCH site return NaN rather than 1.0; a
    missing control is not a passing control.
    """
    if level not in ("end", "molecule"):
        raise ValueError("level must be 'end' or 'molecule'")
    if level == "end":
        k, n = df["k_hch"].to_numpy(), df["n_hch"].to_numpy()
        idx = df.index
    else:
        g = df.groupby("pair_idx", sort=True)[["k_hch", "n_hch"]].sum()
        k, n = g["k_hch"].to_numpy(), g["n_hch"].to_numpy()
        idx = g.index
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(n > 0, 1.0 - k / np.maximum(n, 1), np.nan)
    return pd.Series(rate, index=idx, name="conversion_rate")
