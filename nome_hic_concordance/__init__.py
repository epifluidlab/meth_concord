"""NOMe-HiC Long-Range Chromatin Accessibility Concordance Pipeline.

This package quantifies long-range GCH concordance per genomic bin per cell
from NOMe-HiC bisulfite Hi-C data, producing a cells x bins matrix suitable
for UMAP-based cell clustering.

See ``design/NOME_HIC_CONCORDANCE_PIPELINE_DESIGN.md`` for the full design.
"""

__version__ = "1.0.0"

from . import (
    bam_parser,
    concordance,
    distance_analysis,
    matrix_ops,
    mixture_model,
    posterior,
    reference,
    shrinkage,
    utils,
    visualization,
)

__all__ = [
    "bam_parser",
    "concordance",
    "distance_analysis",
    "matrix_ops",
    "mixture_model",
    "posterior",
    "reference",
    "shrinkage",
    "utils",
    "visualization",
]
