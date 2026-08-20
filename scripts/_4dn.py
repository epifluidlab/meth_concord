"""Shared plumbing for the 4DN portal fetchers.

Two things here rather than in each fetcher, because getting either subtly
wrong is silent:

  * the portal's own ``@@download`` URL answers 403 without an account, while
    the identical file is served unauthenticated from the ``open_data_url``
    S3 mirror recorded on the record;
  * every download is md5-checked against the portal's own checksum, so a
    truncated transfer fails here instead of becoming a quiet hole in a track.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger("4dn")

API = "https://data.4dnucleome.org/search/"

#: The record fields every fetcher needs.  ``open_data_url`` is the important
#: one -- without it a file cannot be retrieved unauthenticated at all.
FIELDS = ("accession", "track_and_facet_info", "genome_assembly",
          "file_type_detailed", "description", "open_data_url",
          "file_size", "md5sum")


def search(params, fields=FIELDS, timeout=180):
    """Run a portal search and return its ``@graph``.

    The portal answers a zero-result search with 404 rather than an empty
    list, which reads as a broken URL; it is translated back to [] here so
    callers can report "nothing matched" against the filters they used.
    """
    query = urllib.parse.urlencode(params, doseq=True)
    query += "".join(f"&field={f}" for f in fields) + "&limit=all&format=json"
    url = f"{API}?{query}"
    logger.info("querying 4DN ...")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as fh:
            return json.load(fh).get("@graph", [])
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []
        raise


_TIER = re.compile(r"\s*\(Tier\s*\d+\)\s*")


def cell_name(raw):
    """'HFFc6 (Tier 1)' -> 'HFFc6'.  The tier is portal curation metadata and
    only gets in the way once the name is a track label.

    Only the tier is removed, never the tail after it.  4DN writes derived
    states as a suffix -- 'H1-hESC (Tier 1) differentiated to definitive
    endoderm' -- so truncating at the tier collapses that into plain H1-hESC
    and averages two different cell states into one track, with nothing in the
    output to show it happened.
    """
    name = _TIER.sub(" ", str(raw or "")).strip()
    return re.sub(r"\s+", "_", name)


def download(rec, dest: Path, timeout=600):
    """Fetch one record to ``dest``, verifying md5.  Returns ``dest``."""
    dest = Path(dest)
    url = rec.get("open_data_url")
    if not url:
        raise SystemExit(
            f"{rec.get('accession')} has no open_data_url, so it can only be "
            f"fetched from the portal's authenticated endpoint (which answers "
            f"403 without an account).  Download it by hand from "
            f"https://data.4dnucleome.org/files-processed/{rec.get('accession')}/")
    want = rec.get("md5sum")
    if dest.exists() and want:
        if hashlib.md5(dest.read_bytes()).hexdigest() == want:
            logger.info("  cached %s", dest.name)
            return dest
    logger.info("  downloading %s (%.1f MB) ...", dest.name,
                (rec.get("file_size") or 0) / 1e6)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        tmp.write_bytes(fh.read())
    if want:
        got = hashlib.md5(tmp.read_bytes()).hexdigest()
        if got != want:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"{dest.name}: md5 {got} != portal's {want}; the "
                             f"download was truncated or corrupted")
    tmp.replace(dest)
    return dest


def write_flags(paths, path, flag="--signal-track", header=None):
    """Write a ready-to-splice argument list: ``$(cat flags.txt)``."""
    parts = [f"{flag} {label}={Path(p).resolve()}"
             for label, p in sorted(paths.items())]
    text = (f"# {header}\n" if header else "") + " ".join(parts) + "\n"
    Path(path).write_text(text)
    return path
