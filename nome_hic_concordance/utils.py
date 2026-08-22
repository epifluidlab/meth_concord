"""Small I/O and logging helpers shared across the pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> None:
    """Install a consistent root logging handler for the pipeline.

    Parameters
    ----------
    level : int
        Root logging level (default: INFO).
    log_file : str, optional
        If provided, also write log records to this file.
    """
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    # Remove any pre-existing handlers so repeated CLI calls stay clean.
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def chrom_is_autosomal_or_sex(chrom: str) -> bool:
    """Return True for chr1-chr22, chrX, chrY; False for chrM, alt contigs, etc."""
    if chrom in ("chrX", "chrY"):
        return True
    if chrom.startswith("chr") and chrom[3:].isdigit():
        return True
    # Also accept bare '1', '2', ... style chromosome names.
    if chrom.isdigit():
        return True
    if chrom in ("X", "Y"):
        return True
    return False


# ---------------------------------------------------------------------------
# Chromosome exclusion, shared by every CLI so one spelling works everywhere
# ---------------------------------------------------------------------------

def chrom_key(chrom: str) -> str:
    """Canonical form, so 'chr1'/'1' and 'chrM'/'M'/'MT' compare equal.

    Reference sources disagree on naming -- a 2bit may say 'chrM' where a BAM
    header says 'MT' -- and an exclusion that silently matches nothing is worse
    than no exclusion at all, because the run still looks like it worked.
    """
    c = str(chrom).strip()
    if c.lower().startswith("chr"):
        c = c[3:]
    c = c.upper()
    return "M" if c in ("M", "MT") else c


def parse_exclude_chroms(values) -> set[str]:
    """``--exclude-chrom`` -> set of canonical keys.

    Accepts the flag repeated, comma-separated, or both:
    ``--exclude-chrom chrM,chrX --exclude-chrom chrY``.
    """
    if not values:
        return set()
    if isinstance(values, str):
        values = [values]
    out: set[str] = set()
    for v in values:
        for part in str(v).split(","):
            if part.strip():
                out.add(chrom_key(part))
    return out


def parse_include_chroms(values) -> list[str]:
    """``--include-chrom`` -> contig names, in the order given.

    The mirror of :func:`parse_exclude_chroms`, but a LIST rather than a set
    of canonical keys: these names are handed to the reference reader, which
    needs them spelled the way the reference spells them.  Accepts the flag
    repeated, comma-separated, or both.
    """
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    out: list[str] = []
    for v in values:
        for part in str(v).split(","):
            part = part.strip()
            if part and part not in out:
                out.append(part)
    return out


def add_include_chrom_arg(parser) -> None:
    """Register --include-chrom identically across the CLIs."""
    parser.add_argument(
        "--include-chrom", action="append", default=None, metavar="CHROM",
        help="Contigs to ADD to the default set (chr1-22, chrX, chrY), "
             "repeatable and/or comma-separated.  Use '--include-chrom chrM' "
             "to keep the mitochondrial genome: it is excluded by default "
             "because it is not somewhere you want bins, but it carries no "
             "nucleosomes, which makes it the one reference for per-cell "
             "assay reliability that does not assume the tumour left its "
             "chromatin alone (scripts/reference_concordance.py).  Spell it "
             "as the reference does; a name matching no contig is an error, "
             "not a silent skip.")


def chrom_excluded(chrom: str, excluded: set[str]) -> bool:
    return bool(excluded) and chrom_key(chrom) in excluded


def bin_label_chrom(label: str) -> str:
    """Chromosome out of a bin column label such as ``'chr1:250000'``."""
    s = str(label)
    return s.rsplit(":", 1)[0] if ":" in s else s


def add_exclude_chrom_arg(parser) -> None:
    """Register --exclude-chrom identically across the CLIs."""
    parser.add_argument(
        "--exclude-chrom", action="append", default=None, metavar="CHROM",
        help="Contigs to drop from the analysis. Repeatable and/or "
             "comma-separated: --exclude-chrom chrM,chrX --exclude-chrom chrY. "
             "chrM/M/MT and chr1/1 are matched interchangeably, and a value "
             "matching no contig warns rather than silently doing nothing. "
             "Use 'chrM,chrX,chrY' to restrict the analysis to autosomes.")


def report_excluded(excluded: set[str], present, kind: str = "contig"):
    """Log what was dropped, and warn about anything that matched nothing.

    Returns the subset of ``present`` that survives.
    """
    log = logging.getLogger("nome_hic_concordance")
    if not excluded:
        return list(present)
    kept, dropped = [], []
    for c in present:
        (dropped if chrom_excluded(c, excluded) else kept).append(c)
    if dropped:
        log.info(f"Excluding {len(dropped)} {kind}(s): "
                    f"{', '.join(map(str, sorted(dropped)))}")
    unmatched = excluded - {chrom_key(c) for c in dropped}
    if unmatched:
        log.warning(f"--exclude-chrom matched no {kind} for: "
                       f"{', '.join(sorted(unmatched))}")
    return kept
