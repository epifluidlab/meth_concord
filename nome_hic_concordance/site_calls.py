"""Per-cytosine (position, state) calls from a cell's BAM.

The main pipeline's :func:`bam_parser._count_context_in_read` walks the
individual context sites of a read but only accumulates ``(k, n)`` — the
positions and per-site calls are discarded.  Site-level analyses (co-methylation
decay, "HCG and its next GCH") need those calls, so this module re-runs the same
walk and *keeps* them.

The QC and per-site logic are deliberately identical to ``bam_parser`` — same
skip-mask (unmapped / mate-unmapped / secondary / duplicate, and supplementary
unless kept), same ``mapq_min`` applied to BOTH mates, same cis-only pairing,
same ``trim_ends``, and the same bisulfite-strand rule (C→T reads score only
ref=C sites, G→A reads only ref=G) — so a read that contributes ``k/n`` upstream
contributes exactly those same sites here.
"""

from __future__ import annotations

import logging

import numpy as np

from .bam_parser import (_FLAG_PROPER_PAIR, _SKIP_FLAGS, _SKIP_FLAGS_BASE,
                         _infer_conversion_strand)

logger = logging.getLogger(__name__)

#: context codes used in the returned arrays (compact, join-friendly)
CTX_CODE = {"GCH": 0, "HCG": 1}


def site_calls_in_read(read, ctx_pos, ctx_ref_base, trim_ends: int = 5):
    """Per-site ``(positions, states)`` for one read.

    The per-site analogue of ``bam_parser._count_context_in_read``: same strand
    inference, same trimming, same fast/indel paths, but emitting each
    informative site instead of only its tally.  ``states`` is 1 = methylated
    (base kept as C/G), 0 = unmethylated (converted to T/A).

    Returns ``(int64 positions, uint8 states)``; both empty when the read is
    unusable (no strand call, fully trimmed, no overlapping sites).
    """
    empty = (np.empty(0, np.int64), np.empty(0, np.uint8))
    ref_start = read.reference_start + trim_ends
    ref_end = read.reference_end - trim_ends
    if ref_start >= ref_end:
        return empty

    i0 = int(np.searchsorted(ctx_pos, ref_start, side="left"))
    i1 = int(np.searchsorted(ctx_pos, ref_end, side="left"))
    if i0 >= i1:
        return empty

    seq = read.query_sequence
    if seq is None:
        return empty

    strand = _infer_conversion_strand(read)
    if strand == "CT":
        informative_ref = 67          # ord('C') — + strand sites only
    elif strand == "GA":
        informative_ref = 71          # ord('G') — − strand sites only
    else:
        return empty                  # strand undeterminable → skip the read

    site_positions = ctx_pos[i0:i1].tolist()
    site_ref_bases = ctx_ref_base[i0:i1].tolist()
    pos_out: list[int] = []
    state_out: list[int] = []

    if read.reference_length == read.query_alignment_length:
        # ---- Fast path: no indels ----
        offset = read.query_alignment_start - read.reference_start
        seq_len = len(seq)
        for j in range(len(site_positions)):
            rb = site_ref_bases[j]
            if rb != informative_ref:
                continue
            qp = site_positions[j] + offset
            if qp < 0 or qp >= seq_len:
                continue
            base = seq[qp]
            if rb == 67:
                if base == "C":
                    pos_out.append(site_positions[j]); state_out.append(1)
                elif base == "T":
                    pos_out.append(site_positions[j]); state_out.append(0)
            else:
                if base == "G":
                    pos_out.append(site_positions[j]); state_out.append(1)
                elif base == "A":
                    pos_out.append(site_positions[j]); state_out.append(0)
    else:
        # ---- Slow path: has indels ----
        pairs_dict = {
            ref_p: query_p
            for query_p, ref_p in read.get_aligned_pairs(matches_only=True)
        }
        for j in range(len(site_positions)):
            rb = site_ref_bases[j]
            if rb != informative_ref:
                continue
            qp = pairs_dict.get(site_positions[j])
            if qp is None:
                continue
            base = seq[qp]
            if rb == 67:
                if base == "C":
                    pos_out.append(site_positions[j]); state_out.append(1)
                elif base == "T":
                    pos_out.append(site_positions[j]); state_out.append(0)
            else:
                if base == "G":
                    pos_out.append(site_positions[j]); state_out.append(1)
                elif base == "A":
                    pos_out.append(site_positions[j]); state_out.append(0)

    if not pos_out:
        return empty
    return (np.asarray(pos_out, np.int64), np.asarray(state_out, np.uint8))


def iter_cell_read_pairs(
    bam_path: str,
    context_positions: dict[str, dict[str, np.ndarray]],
    trim_ends: int = 5,
    mapq_min: int = 30,
    keep_supplementary: bool = False,
    require_proper_pair: bool = False,
    max_pairs_per_cell: int = 0,
    rng: np.random.Generator | None = None,
):
    """Yield one high-quality cis read PAIR at a time, with per-site calls.

    ``context_positions`` maps context name ("GCH"/"HCG") → ``{chrom: structured
    array with 'pos' and 'ref_base'}`` (the pipeline's Step-1 output).

    Each yielded item is::

        (chrom, mid1, mid2, {ctx: (pos1, state1)}, {ctx: (pos2, state2)})

    where ``mid*`` are the read-end midpoints (matching bam_parser's pair
    positions).  ``max_pairs_per_cell`` (0 = unlimited) caps the pairs kept per
    cell via reservoir-free head sampling — cheap and adequate because BAM order
    is coordinate-sorted per chromosome, so the cap is applied per chromosome to
    avoid biasing toward chr1.
    """
    import pysam

    skip_flags = _SKIP_FLAGS_BASE if keep_supplementary else _SKIP_FLAGS
    contexts = list(context_positions)
    chroms = sorted({c for ctx in contexts for c in context_positions[ctx]})
    per_chrom_cap = 0
    if max_pairs_per_cell and chroms:
        per_chrom_cap = max(1, int(max_pairs_per_cell // len(chroms)))

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for chrom in chroms:
            arrs = {}
            for ctx in contexts:
                a = context_positions[ctx].get(chrom)
                if a is not None and len(a):
                    arrs[ctx] = (a["pos"], a["ref_base"])
            if not arrs:
                continue
            try:
                read_iter = bam.fetch(chrom)
            except ValueError:
                continue

            buf: dict[str, object] = {}
            kept = 0
            for read in read_iter:
                if per_chrom_cap and kept >= per_chrom_cap:
                    break
                flag = read.flag
                if flag & skip_flags:
                    continue
                if require_proper_pair and not (flag & _FLAG_PROPER_PAIR):
                    continue
                if read.mapping_quality < mapq_min:
                    continue
                if read.next_reference_name != chrom:     # cis only
                    continue
                qname = read.query_name
                mate = buf.pop(qname, None)
                if mate is None:
                    buf[qname] = read
                    continue
                if mate.mapping_quality < mapq_min:       # mapQ>30 BOTH ends
                    continue

                r1, r2 = ((read, mate)
                          if read.reference_start <= mate.reference_start
                          else (mate, read))
                calls1, calls2 = {}, {}
                for ctx, (cpos, cref) in arrs.items():
                    calls1[ctx] = site_calls_in_read(r1, cpos, cref, trim_ends)
                    calls2[ctx] = site_calls_in_read(r2, cpos, cref, trim_ends)
                if not any(len(c[0]) for c in calls1.values()) and \
                   not any(len(c[0]) for c in calls2.values()):
                    continue                              # no informative site
                mid1 = (r1.reference_start + r1.reference_end) // 2
                mid2 = (r2.reference_start + r2.reference_end) // 2
                kept += 1
                yield chrom, mid1, mid2, calls1, calls2
            del buf
