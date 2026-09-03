"""The ``--samples`` manifest, validated once and shared by all three parts.

Design section 2.  Everything here runs BEFORE any compute, because each of
these mistakes is one that otherwise surfaces hours later as a wrong number
rather than as an error:

* a ``--group-order`` that does not match the manifest silently drops cells
  from every figure while the pipeline reports success;
* a group name containing ``/`` -- and the real order has three -- turns a
  per-group output path into a directory that was never created, or worse into
  a path that collides with another group's;
* a group sitting in one batch makes cell type and batch inseparable for that
  group, which no downstream correction can undo and no downstream diagnostic
  will attribute correctly.

So the loader collects every problem it can find and raises once, rather than
failing on the first and making the user re-run to discover the second.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .utils import manifest_text

logger = logging.getLogger(__name__)

#: Canonical internal names, and the flag that overrides each.
COLUMN_FLAGS = {
    "bam": "--bam-column",
    "name": "--name-column",
    "group": "--group-column",
    "color": "--color-column",
    "batch": "--batch-column",
}

#: Anything outside this becomes ``_`` in a path.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class ManifestError(ValueError):
    """Every problem found in one manifest, reported together."""

    def __init__(self, problems):
        self.problems = list(problems)
        body = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(
            f"{len(self.problems)} problem(s) in --samples:\n{body}")


def safe_name(group: str) -> str:
    """Filename-safe form of a group label (``IT-L2/3`` -> ``IT-L2_3``)."""
    return _UNSAFE.sub("_", group).strip("_") or "unnamed"


@dataclass
class Manifest:
    """A validated manifest.

    ``table`` carries the canonical column names regardless of what the file
    called them; ``source_columns`` records the mapping so error messages and
    metadata can speak in the user's terms.
    """

    table: pd.DataFrame
    group_order: list[str]
    colors: dict[str, str]
    safe_names: dict[str, str]
    source_columns: dict[str, str]
    path: str
    optional_missing: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.table)

    @property
    def cells(self) -> list[str]:
        return self.table["name"].tolist()

    def contingency(self) -> pd.DataFrame:
        """group x batch cell counts, rows in ``group_order``."""
        if "batch" not in self.table:
            return pd.DataFrame(index=self.group_order)
        tab = pd.crosstab(self.table["group"], self.table["batch"])
        return tab.reindex(self.group_order).fillna(0).astype(int)

    def confounded_groups(self) -> list[str]:
        """Groups present in exactly one batch."""
        tab = self.contingency()
        if tab.empty or tab.shape[1] < 2:
            return []
        return [g for g in tab.index if int((tab.loc[g] > 0).sum()) == 1
                and int(tab.loc[g].sum()) > 0]

    def report(self) -> None:
        """Log the startup summary the design asks for."""
        logger.info("Manifest %s: %d cells, %d groups",
                    self.path, len(self.table), len(self.group_order))
        for missing in self.optional_missing:
            logger.info("  no %s column; %s features disabled",
                        COLUMN_FLAGS[missing], missing)
        renamed = {g: s for g, s in self.safe_names.items() if g != s}
        if renamed:
            logger.info("  group names sanitized for paths: %s",
                        ", ".join(f"{g!r}->{s!r}" for g, s in renamed.items()))
        tab = self.contingency()
        if tab.empty:
            return
        logger.info("  group x batch:\n%s", tab.to_string())
        confounded = self.confounded_groups()
        if confounded:
            logger.warning(
                "  %d group(s) sit in a SINGLE batch: %s.  Cell-type and batch "
                "effects are not separable for these; no batch correction can "
                "recover the difference and any group-level result for them is "
                "confounded by construction.",
                len(confounded), ", ".join(repr(g) for g in confounded))

    def metadata(self) -> dict:
        return {
            "samples": self.path,
            "n_cells": len(self.table),
            "group_order": list(self.group_order),
            "colors": dict(self.colors),
            "group_safe_names": dict(self.safe_names),
            "source_columns": dict(self.source_columns),
            "confounded_groups": self.confounded_groups(),
        }


def _has_index(bam: Path) -> bool:
    return any(p.exists() for p in (
        bam.with_suffix(bam.suffix + ".bai"),
        bam.with_suffix(bam.suffix + ".csi"),
        bam.with_suffix(".bai"),
        bam.with_suffix(".csi"),
    ))


def load_manifest(
    path: str,
    *,
    bam_column: str = "bam",
    name_column: str = "name",
    group_column: str = "group",
    color_column: str = "color",
    batch_column: str = "batch",
    group_order: str | list[str] | None = None,
    require_bam: bool = True,
    require_index: bool = True,
) -> Manifest:
    """Load and validate a ``--samples`` TSV.

    ``group_order`` is the ``--group-order`` string (comma-delimited) or an
    already-split list.  Group names legitimately contain ``/`` (``IT-L2/3``),
    so only ``,`` splits.

    Raises
    ------
    ManifestError
        With every problem found, not just the first.
    """
    src = {"bam": bam_column, "name": name_column, "group": group_column,
           "color": color_column, "batch": batch_column}
    df = pd.read_csv(manifest_text(path), sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    problems: list[str] = []
    optional_missing: list[str] = []

    for canon in ("name", "group"):
        if src[canon] not in df.columns:
            problems.append(
                f"required column {src[canon]!r} ({COLUMN_FLAGS[canon]}) not in "
                f"manifest; found {list(df.columns)}")
    if require_bam and src["bam"] not in df.columns:
        problems.append(
            f"required column {src['bam']!r} ({COLUMN_FLAGS['bam']}) not in "
            f"manifest; found {list(df.columns)}")
    if problems:
        raise ManifestError(problems)

    rename = {v: k for k, v in src.items() if v in df.columns}
    table = df.rename(columns=rename)
    for canon in ("bam", "color", "batch"):
        if canon not in table.columns:
            optional_missing.append(canon)
    keep = [c for c in ("name", "bam", "group", "color", "batch")
            if c in table.columns]
    table = table[keep].copy()
    for c in keep:
        table[c] = table[c].astype(str).str.strip()

    # ---- name uniqueness -------------------------------------------------
    dup = table["name"][table["name"].duplicated()].unique().tolist()
    if dup:
        problems.append(
            f"{len(dup)} duplicated {src['name']!r} value(s): "
            f"{', '.join(dup[:5])}{' ...' if len(dup) > 5 else ''}.  Names join "
            f"all three parts, so a duplicate silently merges two cells.")

    # ---- BAMs exist and are indexed --------------------------------------
    if "bam" in table.columns and require_bam:
        missing = [b for b in table["bam"] if not Path(b).exists()]
        if missing:
            problems.append(
                f"{len(missing)} {src['bam']!r} path(s) do not exist, first: "
                f"{missing[0]}")
        if require_index:
            unindexed = [b for b in table["bam"]
                         if Path(b).exists() and not _has_index(Path(b))]
            if unindexed:
                problems.append(
                    f"{len(unindexed)} BAM(s) have no .bai/.csi index, first: "
                    f"{unindexed[0]}")

    # ---- group order matches the manifest, both ways ---------------------
    present = list(dict.fromkeys(table["group"].tolist()))
    if group_order is None:
        order = sorted(present)
    else:
        order = ([g.strip() for g in group_order.split(",") if g.strip()]
                 if isinstance(group_order, str) else list(group_order))
        unknown = [g for g in order if g not in present]
        unlisted = [g for g in present if g not in order]
        if unknown:
            problems.append(
                f"--group-order names {len(unknown)} group(s) absent from the "
                f"manifest: {', '.join(map(repr, unknown))}")
        if unlisted:
            problems.append(
                f"{len(unlisted)} manifest group(s) missing from "
                f"--group-order: {', '.join(map(repr, unlisted))}.  They would "
                f"be dropped from every ordered output without an error.")
        dups = [g for g in set(order) if order.count(g) > 1]
        if dups:
            problems.append(f"--group-order repeats {', '.join(map(repr, dups))}")

    # ---- one colour per group -------------------------------------------
    colors: dict[str, str] = {}
    if "color" in table.columns:
        for g, sub in table.groupby("group")["color"]:
            distinct = sorted(set(sub) - {"", "nan", "None"})
            if len(distinct) > 1:
                problems.append(
                    f"group {g!r} has {len(distinct)} different "
                    f"{src['color']!r} values: {', '.join(distinct)}")
            elif distinct:
                colors[g] = distinct[0]
        uncolored = [g for g in present if g not in colors]
        if uncolored and colors:
            problems.append(
                f"{len(uncolored)} group(s) have no {src['color']!r} value "
                f"while others do: {', '.join(map(repr, uncolored[:5]))}")

    # ---- filename sanitization, and its collisions -----------------------
    safe = {g: safe_name(g) for g in present}
    collided: dict[str, list[str]] = {}
    for g, s in safe.items():
        collided.setdefault(s, []).append(g)
    for s, gs in collided.items():
        if len(gs) > 1:
            problems.append(
                f"groups {', '.join(map(repr, gs))} all sanitize to {s!r}; "
                f"their per-group outputs would overwrite each other")

    if problems:
        raise ManifestError(problems)

    table = table.reset_index(drop=True)
    return Manifest(table=table, group_order=order, colors=colors,
                    safe_names=safe, source_columns=src, path=str(path),
                    optional_missing=optional_missing)


def add_manifest_args(parser, *, require_samples: bool = True) -> None:
    """The shared ``--samples`` flag group (design section 2)."""
    g = parser.add_argument_group("samples manifest")
    g.add_argument("--samples", required=require_samples,
                   help="TSV manifest; see design section 2.")
    g.add_argument("--bam-column", default="bam")
    g.add_argument("--name-column", default="name")
    g.add_argument("--group-column", default="group")
    g.add_argument("--color-column", default="color")
    g.add_argument("--batch-column", default="batch")
    g.add_argument("--group-order", default=None,
                   help="Comma-delimited group order for heatmap columns, "
                        "legends and per-group tables.  Group names may "
                        "contain '/', so only ',' splits.")


def load_from_args(args, **kw) -> Manifest:
    """:func:`load_manifest` driven by :func:`add_manifest_args` flags."""
    return load_manifest(
        args.samples,
        bam_column=args.bam_column, name_column=args.name_column,
        group_column=args.group_column, color_column=args.color_column,
        batch_column=args.batch_column, group_order=args.group_order, **kw)
