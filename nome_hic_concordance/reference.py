"""Cytosine context extraction from a .2bit reference genome.

Extracts positions of user-specified cytosine trinucleotide contexts on both
strands of the reference. GCG is always excluded (ambiguous between GCH and
HCG). Optionally removes positions overlapping dbSNP variants.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np

from .utils import chrom_is_autosomal_or_sex, ensure_dir

logger = logging.getLogger(__name__)


# IUPAC ambiguity codes. Keys are IUPAC letters; values are the concrete bases
# each code matches.
IUPAC: dict[str, set[str]] = {
    "A": {"A"}, "C": {"C"}, "G": {"G"}, "T": {"T"},
    "H": {"A", "C", "T"}, "W": {"A", "T"}, "S": {"C", "G"},
    "R": {"A", "G"}, "Y": {"C", "T"}, "K": {"G", "T"},
    "M": {"A", "C"}, "B": {"C", "G", "T"}, "D": {"A", "G", "T"},
    "V": {"A", "C", "G"}, "N": {"A", "C", "G", "T"},
}

# Complement of each IUPAC letter (so that reversing and complementing a
# context pattern gives its minus-strand equivalent).
IUPAC_COMP: dict[str, str] = {
    "A": "T", "T": "A", "C": "G", "G": "C",
    "H": "D", "D": "H", "W": "W", "S": "S",
    "R": "Y", "Y": "R", "K": "M", "M": "K",
    "B": "V", "V": "B", "N": "N",
}


# Structured dtype used for per-context, per-chromosome arrays.
CONTEXT_DTYPE = np.dtype([("pos", np.int64), ("ref_base", np.uint8)])


def _matches_context(tri: str, context: str) -> bool:
    """Return True iff a 3-base trinucleotide matches an IUPAC context pattern."""
    if len(tri) != 3 or len(context) != 3:
        return False
    for base, code in zip(tri, context):
        if code not in IUPAC:
            return False
        if base not in IUPAC[code]:
            return False
    return True


def _reverse_complement_context(context: str) -> str:
    """Reverse-complement an IUPAC context pattern (e.g. 'GCH' -> 'DGC')."""
    return "".join(IUPAC_COMP[c] for c in reversed(context))


def _load_snp_positions(vcf_path: str, chrom: str) -> set[int]:
    """Return 0-based positions of SNPs on ``chrom`` in a tabix-indexed VCF."""
    import pysam  # local import keeps reference.py importable without pysam at module load

    snp_positions: set[int] = set()
    try:
        vcf = pysam.VariantFile(vcf_path)
    except (OSError, ValueError) as exc:
        logger.warning(f"Could not open dbSNP VCF {vcf_path}: {exc}")
        return snp_positions

    try:
        try:
            fetcher = vcf.fetch(chrom)
        except ValueError:
            # Chromosome not present in the VCF index; silently return empty.
            return snp_positions
        for record in fetcher:
            # VCF is 1-based, we store 0-based.
            snp_positions.add(record.pos - 1)
    finally:
        vcf.close()
    return snp_positions


def _extract_chrom_contexts(
    twobit_path: str,
    chrom: str,
    contexts: list[str],
    snp_positions: set[int] | None = None,
) -> tuple[str, dict[str, np.ndarray]]:
    """Extract context positions for a single chromosome, both strands.

    Returns
    -------
    (chrom, {context_name: structured array})

    Each structured array has fields:
        pos : int64       0-based position of the cytosine on the + strand
        ref_base : uint8  ord('C') for + strand sites, ord('G') for - strand sites
    """
    import py2bit  # local import for speed

    tb = py2bit.open(twobit_path)
    try:
        seq = tb.sequence(chrom).upper()
    finally:
        tb.close()

    results: dict[str, np.ndarray] = {}

    # Pre-compute minus-strand patterns and the corresponding GCG-complement
    # ('CGC') exclusion so we don't have to redo it per scan position.
    minus_contexts = {ctx: _reverse_complement_context(ctx) for ctx in contexts}

    for context in contexts:
        positions: list[int] = []
        ref_bases: list[int] = []

        minus_ctx = minus_contexts[context]
        for i in range(len(seq) - 2):
            tri = seq[i:i + 3]

            plus_match = _matches_context(tri, context)
            minus_match = _matches_context(tri, minus_ctx)
            if not plus_match and not minus_match:
                continue

            # Exclude GCG on the + strand: ambiguous between GCH and HCG, which
            # is the central NOMe-HiC convention. The minus-strand complement
            # of GCG is CGC, which we also must exclude when matching.
            if tri == "GCG" or tri == "CGC":
                continue

            # Skip the trinucleotide if any of its positions overlaps a SNP.
            if snp_positions is not None:
                if any((p in snp_positions) for p in (i, i + 1, i + 2)):
                    continue

            # The context cytosine is the middle base of the trinucleotide in
            # both strand conventions (GC*H*: + strand C is middle when we
            # interpret the pattern as starting at i; for minus strand the G
            # at position i+1 is the complement of the C we care about).
            center_pos = i + 1

            if plus_match:
                positions.append(center_pos)
                ref_bases.append(ord("C"))
            if minus_match:
                positions.append(center_pos)
                ref_bases.append(ord("G"))

        if positions:
            arr = np.empty(len(positions), dtype=CONTEXT_DTYPE)
            arr["pos"] = positions
            arr["ref_base"] = ref_bases
            # Sort primarily by position; two sites (one per strand) can share
            # a position only if the trinucleotide itself is palindromic, which
            # never happens for GCH / HCG after the GCG exclusion above.
            arr.sort(order="pos")
            results[context] = arr
        else:
            results[context] = np.empty(0, dtype=CONTEXT_DTYPE)

    return chrom, results


def extract_context_positions(
    twobit_path: str,
    output_dir: str,
    contexts: list[str] | None = None,
    dbsnp_vcf: str | None = None,
    n_threads: int = 4,
    chroms: Iterable[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Extract cytosine context positions from a .2bit reference genome.

    Parameters
    ----------
    twobit_path : str
        Path to a .2bit reference file.
    output_dir : str
        Directory for per-chromosome .npz outputs.
    contexts : list[str], optional
        IUPAC context patterns to extract. Default ``["GCH"]``.
    dbsnp_vcf : str, optional
        Path to a tabix-indexed dbSNP VCF. If given, context sites whose
        trinucleotide overlaps any SNP are excluded.
    n_threads : int
        Parallel workers for per-chromosome processing.
    chroms : iterable of str, optional
        Restrict processing to these chromosomes. Defaults to chr1..chr22,
        chrX, chrY (autosomes + sex chromosomes, excluding chrM and alts).

    Returns
    -------
    dict : context -> {chrom: structured array}
        Structured array dtype: ('pos' int64, 'ref_base' uint8).
    """
    import py2bit
    from joblib import Parallel, delayed

    if contexts is None:
        contexts = ["GCH"]

    tb = py2bit.open(twobit_path)
    try:
        all_chroms = list(tb.chroms())
    finally:
        tb.close()

    if chroms is None:
        chrom_list = [c for c in all_chroms if chrom_is_autosomal_or_sex(c)]
    else:
        chrom_list = [c for c in chroms if c in all_chroms]

    snp_per_chrom: dict[str, set[int]] = {}
    if dbsnp_vcf:
        logger.info(f"Loading dbSNP positions from {dbsnp_vcf} for {len(chrom_list)} chromosomes...")
        for c in chrom_list:
            snp_per_chrom[c] = _load_snp_positions(dbsnp_vcf, c)

    logger.info(
        f"Extracting contexts {contexts} from {len(chrom_list)} chromosomes "
        f"(both strands) using {n_threads} threads..."
    )

    if n_threads > 1 and len(chrom_list) > 1:
        results = Parallel(n_jobs=n_threads)(
            delayed(_extract_chrom_contexts)(
                twobit_path, c, contexts, snp_per_chrom.get(c)
            )
            for c in chrom_list
        )
    else:
        results = [
            _extract_chrom_contexts(twobit_path, c, contexts, snp_per_chrom.get(c))
            for c in chrom_list
        ]

    out_path = ensure_dir(output_dir)
    all_positions: dict[str, dict[str, np.ndarray]] = {ctx: {} for ctx in contexts}

    for chrom, ctx_results in results:
        for ctx, arr in ctx_results.items():
            all_positions[ctx][chrom] = arr
            np.savez_compressed(
                out_path / f"context_{ctx}.{chrom}.npz",
                pos=arr["pos"],
                ref_base=arr["ref_base"],
            )
            logger.info(f"  {chrom} {ctx}: {len(arr):,} sites (both strands)")

    return all_positions


def load_context_positions(
    input_dir: str,
    contexts: list[str],
    chroms: Iterable[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Load previously extracted context positions from .npz files.

    Parameters
    ----------
    input_dir : str
        Directory containing ``context_{ctx}.{chrom}.npz`` files.
    contexts : list[str]
        Contexts to load.
    chroms : iterable of str, optional
        Chromosomes to load. If None, auto-discover from filenames.
    """
    in_path = Path(input_dir)
    if not in_path.exists():
        raise FileNotFoundError(f"Context directory not found: {input_dir}")

    all_positions: dict[str, dict[str, np.ndarray]] = {}
    for ctx in contexts:
        ctx_files = sorted(in_path.glob(f"context_{ctx}.*.npz"))
        if chroms is not None:
            chrom_set = set(chroms)
            ctx_files = [f for f in ctx_files if _chrom_from_npz(f.name) in chrom_set]

        per_chrom: dict[str, np.ndarray] = {}
        for f in ctx_files:
            chrom = _chrom_from_npz(f.name)
            with np.load(f) as z:
                arr = np.empty(len(z["pos"]), dtype=CONTEXT_DTYPE)
                arr["pos"] = z["pos"]
                arr["ref_base"] = z["ref_base"]
            per_chrom[chrom] = arr
        all_positions[ctx] = per_chrom

    return all_positions


def _load_bed_chrom(bed_path: str, chrom: str) -> tuple[str, np.ndarray]:
    """Load context sites for one chromosome from a tabix-indexed BED.

    Each worker opens its own TabixFile handle (file handles cannot be
    shared across processes).
    """
    import pysam

    tbx = pysam.TabixFile(str(bed_path))
    positions: list[int] = []
    ref_bases: list[int] = []

    try:
        for row in tbx.fetch(chrom):
            fields = row.split("\t")
            start = int(fields[1])
            ref_base_char = _parse_ref_base(fields)
            positions.append(start)
            ref_bases.append(ord(ref_base_char))
    finally:
        tbx.close()

    if positions:
        arr = np.empty(len(positions), dtype=CONTEXT_DTYPE)
        arr["pos"] = positions
        arr["ref_base"] = ref_bases
        arr.sort(order="pos")
    else:
        arr = np.empty(0, dtype=CONTEXT_DTYPE)

    return chrom, arr


def load_context_from_bed(
    bed_paths: dict[str, str],
    chroms: Iterable[str] | None = None,
    n_threads: int = 4,
) -> dict[str, dict[str, np.ndarray]]:
    """Load context positions from tabix-indexed BED files.

    This is the fast alternative to :func:`extract_context_positions` when
    the user already has pre-built GCH / HCG position files.  Chromosomes
    are loaded in parallel via ``joblib``.

    Parameters
    ----------
    bed_paths : dict
        Mapping ``context_name -> path`` to a **bgzipped, tabix-indexed**
        BED file (``*.bed.gz`` with a companion ``.tbi``).

        Accepted BED formats (auto-detected per file):

        * **4-column** ``chrom  start  end  strand_or_base``
          where column 4 is ``+`` / ``-`` (strand) or ``C`` / ``G``
          (reference base at the site).  ``+`` / ``C`` = plus-strand
          cytosine;  ``-`` / ``G`` = minus-strand cytosine.
        * **6-column (BED6)** ``chrom  start  end  name  score  strand``
          where strand is in the standard column 6.
        * **3-column** ``chrom  start  end`` — assumes all sites are
          plus-strand cytosines (``ref_base='C'``).

        Positions must be 0-based, half-open (standard BED).

    chroms : iterable of str, optional
        Restrict to these chromosomes. ``None`` loads all chromosomes
        present in the tabix index.
    n_threads : int
        Number of parallel workers for per-chromosome loading.

    Returns
    -------
    dict : context_name -> {chrom: structured array}
    """
    import pysam
    from joblib import Parallel, delayed

    all_positions: dict[str, dict[str, np.ndarray]] = {}

    for ctx, bed_path in bed_paths.items():
        logger.info(f"Loading {ctx} positions from {bed_path} ...")

        # Discover chromosomes from the tabix index.
        tbx = pysam.TabixFile(str(bed_path))
        if chroms is not None:
            chrom_list = [c for c in chroms if c in tbx.contigs]
        else:
            chrom_list = list(tbx.contigs)
        tbx.close()

        # Parallel per-chromosome loading.
        if n_threads > 1 and len(chrom_list) > 1:
            results = Parallel(n_jobs=n_threads)(
                delayed(_load_bed_chrom)(bed_path, c) for c in chrom_list
            )
        else:
            results = [_load_bed_chrom(bed_path, c) for c in chrom_list]

        per_chrom: dict[str, np.ndarray] = {}
        total_sites = 0
        for chrom, arr in results:
            per_chrom[chrom] = arr
            total_sites += len(arr)

        all_positions[ctx] = per_chrom
        logger.info(
            f"  {ctx}: {total_sites:,} sites across {len(chrom_list)} chromosomes"
        )

    return all_positions


def _parse_ref_base(fields: list[str]) -> str:
    """Determine the reference base (C or G) from a BED record's columns.

    Supports 3-, 4-, and 6-column BED formats (see :func:`load_context_from_bed`).
    """
    n = len(fields)

    # 6-column BED: strand is in column 6.
    if n >= 6:
        strand = fields[5].strip()
        if strand == "+":
            return "C"
        if strand == "-":
            return "G"

    # 4+ columns: column 4 may be strand (+/-) or ref_base (C/G).
    if n >= 4:
        col4 = fields[3].strip()
        if col4 in ("+", "C"):
            return "C"
        if col4 in ("-", "G"):
            return "G"

    # 3-column BED or unrecognised column 4: default to + strand.
    return "C"


def _chrom_from_npz(fname: str) -> str:
    """Parse the chromosome out of ``context_{ctx}.{chrom}.npz``."""
    # Strip leading 'context_' and trailing '.npz'
    stem = fname[len("context_"):-len(".npz")]
    # stem is now '{ctx}.{chrom}' -- split only on the first '.'
    ctx, chrom = stem.split(".", 1)
    return chrom
