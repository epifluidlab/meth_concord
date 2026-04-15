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
