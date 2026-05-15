"""Pair-concordance extension for contact-aware chromatin coordination analysis.

Implements the hierarchical beta-binomial model with linked-vs-unlinked
decomposition, three parallel per-cell embeddings (E_contact, E_coord,
E_full), and component-contribution analysis.

See ``design/PAIR_CONCORDANCE_DESIGN_v2.md`` for the full design.
"""

from . import (
    classify,
    component_analysis,
    embed,
    empirical_bayes,
    hierarchical,
    panel,
    visualization,
)

__all__ = [
    "classify",
    "component_analysis",
    "embed",
    "empirical_bayes",
    "hierarchical",
    "panel",
    "visualization",
]
