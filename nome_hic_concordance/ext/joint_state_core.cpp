/* joint_state_core -- BAM -> per-read-END counts for N cytosine contexts at once.
 *
 * Part A of design/joint_state_rank1_design.md.  The existing extractors could
 * not be called for this:
 *
 *   _bam_parser_core.parse_cell_bam  takes ONE context per call and returns no
 *       read identity, so GCH and HCG counts from two calls cannot be rejoined
 *       within a read end -- which is the entire object of the analysis.
 *   _site_dist_core.parse_cell_contacts  does take both contexts in one pass,
 *       but hard-codes exactly two, discards the qname after mate pairing, and
 *       reservoir-samples to --max-pairs.
 *
 * What is new here, and only this:
 *   - N contexts in a single pass, aggregated to (k, n) per end per context.
 *   - HCH derived from the read's own reference span, not from a site table
 *     (see reconstruct_reference).
 *   - read identity and the per-end geometry of design A.1 carried through.
 *   - no subsampling.
 *
 * Everything else -- conversion-strand inference, the flag mask, the trim and
 * MAPQ and base-quality gates -- is the established behaviour, reached through
 * bam_common.hpp.
 */

#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <htslib/hts.h>
#include <htslib/sam.h>

#include "bam_common.hpp"

namespace py = pybind11;
using namespace nhc;

// Fixed ceiling so a read's counters live in the struct rather than in a
// heap allocation per read.  GCH + HCG + any future mark; raise if needed.
static constexpr int MAX_CTX = 8;

/* ------------------------------------------------------------------------ *
 * Reference reconstruction
 *
 * HCH is the conversion control (design A.3.5) and it is dense: roughly 800 M
 * sites in mouse, which as a (int64 pos, uint8 base) site table is ~7 GB and
 * has to be built and shipped to every worker.  The read already carries what
 * is needed -- CIGAR gives the ref/query correspondence and MD gives the
 * reference base wherever it differs from the read -- so the reference span
 * under each alignment is reconstructible for free, and the trinucleotide
 * context can be read off it directly.
 *
 * This also removes the site table from the OTHER contexts' lookup: once the
 * span is in hand, every context resolves through the same ref->query map,
 * with no second CIGAR walk and no separate indel path.
 * ------------------------------------------------------------------------ */
static bool reconstruct_reference(const bam1_t* b,
                                  std::string& ref_span,
                                  std::vector<int32_t>& ref2query)
{
    const uint8_t* md_aux = bam_aux_get(b, "MD");
    if (!md_aux) return false;
    const char* md = bam_aux2Z(md_aux);
    if (!md) return false;

    const uint32_t* cigar = bam_get_cigar(b);
    const int n_cigar = b->core.n_cigar;
    const uint8_t* seq = bam_get_seq(b);
    const int seq_len = b->core.l_qseq;

    const int64_t ref_len = bam_endpos(b) - b->core.pos;
    if (ref_len <= 0) return false;
    ref_span.assign((size_t)ref_len, 'N');
    ref2query.assign((size_t)ref_len, -1);

    // Pass 1 -- CIGAR.  Matched positions get the READ base; pass 2 corrects
    // the ones where MD says the reference differs.
    int64_t r = 0;
    int q = 0;
    for (int i = 0; i < n_cigar; ++i) {
        const int op = bam_cigar_op(cigar[i]);
        const int len = (int)bam_cigar_oplen(cigar[i]);
        switch (op) {
        case BAM_CMATCH: case BAM_CEQUAL: case BAM_CDIFF:
            for (int j = 0; j < len; ++j, ++r, ++q) {
                if (r < ref_len && q < seq_len) {
                    ref_span[(size_t)r] = bam_base_at(seq, q);
                    ref2query[(size_t)r] = q;
                }
            }
            break;
        case BAM_CINS: case BAM_CSOFT_CLIP:
            q += len;
            break;
        case BAM_CDEL: case BAM_CREF_SKIP:
            r += len;               // reference with no query; MD fills it
            break;
        default:                    // hard clip, pad
            break;
        }
    }

    // Pass 2 -- MD.  Every reference position in [pos, endpos) is consumed by
    // exactly one M or D operation, and both consume reference in order, so
    // MD's own coordinate walks ref_span from 0 with no further bookkeeping.
    int64_t rr = 0;
    const char* p = md;
    while (*p && rr < ref_len) {
        if (*p >= '0' && *p <= '9') {
            rr += strtol(p, (char**)&p, 10);
        } else if (*p == '^') {
            ++p;
            while (*p && isalpha((unsigned char)*p)) {
                if (rr < ref_len)
                    ref_span[(size_t)rr] = (char)toupper((unsigned char)*p);
                ++rr; ++p;
            }
        } else {
            if (rr < ref_len)
                ref_span[(size_t)rr] = (char)toupper((unsigned char)*p);
            ++rr; ++p;
        }
    }
    return true;
}

/* ------------------------------------------------------------------------ *
 * Per-read counting
 * ------------------------------------------------------------------------ */

struct ReadCounts {
    int32_t k[MAX_CTX] = {0};
    int32_t n[MAX_CTX] = {0};
    int32_t k_hch = 0, n_hch = 0;
    int32_t n_snp  = 0;
};

/* One context's (k, n) over the post-trim span, resolved through ref2query. */
static inline void count_one_context(
    const int64_t* ctx_pos, const uint8_t* ctx_ref, int64_t n_ctx,
    int64_t ref_start, const std::vector<int32_t>& r2q,
    const uint8_t* seq, const uint8_t* qual, bool have_qual, int baseq_min,
    uint8_t informative_ref, int64_t lo, int64_t hi,
    int32_t& k_out, int32_t& n_out)
{
    if (n_ctx == 0) return;
    const int64_t* first = std::lower_bound(ctx_pos, ctx_pos + n_ctx, lo);
    const int64_t* last  = std::lower_bound(ctx_pos, ctx_pos + n_ctx, hi);
    const int64_t ref_len = (int64_t)r2q.size();

    for (const int64_t* it = first; it < last; ++it) {
        if (ctx_ref[it - ctx_pos] != informative_ref) continue;   // other strand
        const int64_t r = *it - ref_start;
        if (r < 0 || r >= ref_len) continue;
        const int32_t q = r2q[(size_t)r];
        if (q < 0) continue;                                      // deleted
        if (have_qual && qual[q] < baseq_min) continue;
        const char base = bam_base_at(seq, q);
        if (informative_ref == 'C') {
            if (base == 'C') { ++k_out; ++n_out; }
            else if (base == 'T') { ++n_out; }
        } else {
            if (base == 'G') { ++k_out; ++n_out; }
            else if (base == 'A') { ++n_out; }
        }
    }
}

/* HCH read off the reconstructed span.  ``k_hch`` counts UNCONVERTED
 * cytosines, so k/n is the apparent HCH methylation -- the same orientation as
 * every other context, and one minus the conversion rate. */
static inline void count_hch(
    const std::string& ref_span, const std::vector<int32_t>& r2q,
    const uint8_t* seq, const uint8_t* qual, bool have_qual, int baseq_min,
    uint8_t informative_ref, int64_t lo_off, int64_t hi_off,
    int32_t& k_out, int32_t& n_out)
{
    const int64_t ref_len = (int64_t)ref_span.size();
    const int64_t lo = std::max<int64_t>(lo_off, 1);
    const int64_t hi = std::min<int64_t>(hi_off, ref_len - 1);

    for (int64_t r = lo; r < hi; ++r) {
        const char c = ref_span[(size_t)r];
        const char prev = ref_span[(size_t)(r - 1)];
        const char next = ref_span[(size_t)(r + 1)];
        if (prev == 'N' || next == 'N') continue;   // MD ran short; unknowable

        if (informative_ref == 'C') {
            // + strand HCH: not GCH (prev G), not HCG (next G), not GCG.
            if (c != 'C' || prev == 'G' || next == 'G') continue;
        } else {
            // - strand HCH is revcomp(prev,C,next) = HCH, i.e. neither
            // neighbour may be the C that would make it a GCH or HCG there.
            if (c != 'G' || prev == 'C' || next == 'C') continue;
        }
        const int32_t q = r2q[(size_t)r];
        if (q < 0) continue;
        if (have_qual && qual[q] < baseq_min) continue;
        const char base = bam_base_at(seq, q);
        if (informative_ref == 'C') {
            if (base == 'C') { ++k_out; ++n_out; }      // unconverted
            else if (base == 'T') { ++n_out; }          // converted
        } else {
            if (base == 'G') { ++k_out; ++n_out; }
            else if (base == 'A') { ++n_out; }
        }
    }
}

/* ------------------------------------------------------------------------ */

struct ChromCtx {
    std::string name;
    const int64_t* pos[MAX_CTX] = {nullptr};
    const uint8_t* ref[MAX_CTX] = {nullptr};
    int64_t n[MAX_CTX] = {0};
    const int64_t* snp = nullptr;
    int64_t n_snp = 0;
};

struct EndRec {
    std::string qname;
    int32_t  ci = -1;
    int64_t  pos = 0, endpos = 0;
    int16_t  mapq = 0;
    int8_t   rev = 0;
    int8_t   conv = 0;
    int32_t  tlen = 0;
    ReadCounts c;
};

static py::dict parse_cell_joint_ends(
    const std::string& bam_path,
    py::list contexts,                  // [{chrom: (pos int64, ref uint8)}, ...]
    py::dict snp_positions,             // {chrom: pos int64}; may be empty
    int trim_ends,
    int mapq_min,
    int baseq_min,
    int strand_min_evidence,
    bool keep_supplementary,
    bool count_hch_flag,
    bool emit_read_id,
    int64_t min_distance,
    bool cis_only)
{
    const int n_ctx_sets = (int)contexts.size();
    if (n_ctx_sets > MAX_CTX)
        throw std::runtime_error("at most 8 contexts per call");

    // ---- bind the per-chromosome arrays, keeping the Python owners alive ---
    samFile* fp = sam_open(bam_path.c_str(), "r");
    if (!fp) throw std::runtime_error("cannot open BAM: " + bam_path);
    bam_hdr_t* hdr = sam_hdr_read(fp);
    if (!hdr) { sam_close(fp); throw std::runtime_error("cannot read BAM header"); }

    std::vector<ChromCtx> by_tid((size_t)hdr->n_targets);
    std::vector<py::object> keepalive;
    for (int t = 0; t < hdr->n_targets; ++t) by_tid[(size_t)t].name = hdr->target_name[t];

    for (int ci = 0; ci < n_ctx_sets; ++ci) {
        py::dict d = contexts[(size_t)ci].cast<py::dict>();
        for (auto item : d) {
            const std::string chrom = py::str(item.first);
            const int t = bam_name2id(hdr, chrom.c_str());
            if (t < 0) continue;
            py::tuple arrs = py::cast<py::tuple>(item.second);
            auto pa = arrs[0].cast<py::array_t<int64_t>>();
            auto ra = arrs[1].cast<py::array_t<uint8_t>>();
            keepalive.push_back(arrs[0]); keepalive.push_back(arrs[1]);
            if (pa.size() == 0) continue;
            by_tid[(size_t)t].pos[ci] = pa.data();
            by_tid[(size_t)t].ref[ci] = ra.data();
            by_tid[(size_t)t].n[ci]   = pa.size();
        }
    }
    for (auto item : snp_positions) {
        const std::string chrom = py::str(item.first);
        const int t = bam_name2id(hdr, chrom.c_str());
        if (t < 0) continue;
        auto pa = item.second.cast<py::array_t<int64_t>>();
        keepalive.push_back(py::reinterpret_borrow<py::object>(item.second));
        if (pa.size() == 0) continue;
        by_tid[(size_t)t].snp = pa.data();
        by_tid[(size_t)t].n_snp = pa.size();
    }

    // ---- outputs, one row per END (two consecutive rows per pair) ----------
    std::vector<std::string> out_qname;
    std::vector<int64_t> out_pair, out_p5, out_p3, out_dist;
    std::vector<int32_t> out_chrom, out_frag;
    std::vector<int8_t>  out_end, out_rev, out_conv, out_cis;
    std::vector<int16_t> out_mapq;
    std::vector<int32_t> out_k[MAX_CTX], out_n[MAX_CTX];
    std::vector<int32_t> out_khch, out_nhch, out_nsnp;

    uint64_t n_reads = 0, n_pairs = 0, n_unpaired = 0,
             n_no_md = 0, n_strand_unknown = 0, n_short = 0;
    size_t peak_buffer = 0;

    {
    py::gil_scoped_release release;

    const uint16_t skip_flags = keep_supplementary ? SKIP_FLAGS_BASE : SKIP_FLAGS;
    std::unordered_map<std::string, EndRec> buf;
    bam1_t* b = bam_init1();
    std::string ref_span;
    std::vector<int32_t> r2q;

    // One sequential pass.  Reading the whole file in coordinate order pairs
    // cis and trans mates in the same loop and never touches a read twice; the
    // per-chromosome index queries the older extractors use would have to be
    // repeated per context set, and would drop trans contacts outright.
    while (sam_read1(fp, hdr, b) >= 0) {
        const uint16_t flag = b->core.flag;
        if (flag & skip_flags) continue;
        if (b->core.qual < mapq_min) continue;
        if (b->core.tid < 0 || b->core.mtid < 0) continue;
        if (cis_only && b->core.mtid != b->core.tid) continue;
        ++n_reads;

        EndRec rec;
        rec.qname.assign(bam_get_qname(b));
        rec.ci = b->core.tid;
        rec.pos = b->core.pos;
        rec.endpos = bam_endpos(b);
        rec.mapq = (int16_t)b->core.qual;
        rec.rev = (flag & BAM_FREVERSE) ? 1 : 0;
        rec.tlen = (int32_t)std::abs((long)b->core.isize);

        const int strand = infer_conversion_strand(b, strand_min_evidence);
        rec.conv = (int8_t)strand;

        if (strand == STRAND_UNKNOWN) {
            ++n_strand_unknown;                 // counts stay zero, end kept
        } else if (!reconstruct_reference(b, ref_span, r2q)) {
            ++n_no_md;
        } else {
            const uint8_t informative_ref = (strand == STRAND_CT) ? 'C' : 'G';
            const int64_t lo = rec.pos + trim_ends;
            const int64_t hi = rec.endpos - trim_ends;
            if (lo < hi) {
                const uint8_t* seq = bam_get_seq(b);
                const uint8_t* qual = bam_get_qual(b);
                const bool have_qual =
                    (baseq_min > 0 && qual != nullptr && qual[0] != 0xff);
                const ChromCtx& cc = by_tid[(size_t)b->core.tid];
                for (int ci = 0; ci < n_ctx_sets; ++ci) {
                    count_one_context(cc.pos[ci], cc.ref[ci], cc.n[ci],
                                      rec.pos, r2q, seq, qual, have_qual,
                                      baseq_min, informative_ref, lo, hi,
                                      rec.c.k[ci], rec.c.n[ci]);
                }
                if (count_hch_flag) {
                    count_hch(ref_span, r2q, seq, qual, have_qual, baseq_min,
                              informative_ref, lo - rec.pos, hi - rec.pos,
                              rec.c.k_hch, rec.c.n_hch);
                }
            }
        }

        // Independent of the strand call: an end that contributes no counts
        // still has a span, and whether that span was SNP-masked is the
        // reason to trust or distrust it.
        {
            const ChromCtx& cc = by_tid[(size_t)b->core.tid];
            const int64_t lo = rec.pos + trim_ends;
            const int64_t hi = rec.endpos - trim_ends;
            if (cc.n_snp && lo < hi) {
                const int64_t* f = std::lower_bound(cc.snp, cc.snp + cc.n_snp, lo);
                const int64_t* l = std::lower_bound(cc.snp, cc.snp + cc.n_snp, hi);
                rec.c.n_snp = (int32_t)(l - f);
            }
        }

        auto it = buf.find(rec.qname);
        if (it == buf.end()) {
            const std::string key = rec.qname;
            buf.emplace(key, std::move(rec));
            peak_buffer = std::max(peak_buffer, buf.size());
            continue;
        }

        EndRec mate = std::move(it->second);
        buf.erase(it);

        // Canonical order: lower chromosome index, then lower start.
        EndRec* r1 = &mate;
        EndRec* r2 = &rec;
        if (rec.ci < mate.ci || (rec.ci == mate.ci && rec.pos < mate.pos))
            std::swap(r1, r2);

        const bool is_cis = (r1->ci == r2->ci);
        const int64_t mid1 = (r1->pos + r1->endpos) / 2;
        const int64_t mid2 = (r2->pos + r2->endpos) / 2;
        const int64_t dist = is_cis ? std::abs(mid2 - mid1) : -1;
        if (is_cis && dist < min_distance) { ++n_short; continue; }

        const int64_t pair_idx = (int64_t)n_pairs;
        ++n_pairs;
        EndRec* ends[2] = {r1, r2};
        for (int e = 0; e < 2; ++e) {
            const EndRec& r = *ends[e];
            if (emit_read_id) out_qname.push_back(r.qname);
            out_pair.push_back(pair_idx);
            out_end.push_back((int8_t)(e + 1));
            out_chrom.push_back(r.ci);
            // 5' and 3' in reference coordinates, so strand is not lost.
            out_p5.push_back(r.rev ? r.endpos - 1 : r.pos);
            out_p3.push_back(r.rev ? r.pos : r.endpos - 1);
            out_rev.push_back(r.rev);
            out_conv.push_back(r.conv);
            out_mapq.push_back(r.mapq);
            out_frag.push_back(r.tlen);
            out_dist.push_back(dist);
            out_cis.push_back(is_cis ? 1 : 0);
            for (int ci = 0; ci < n_ctx_sets; ++ci) {
                out_k[ci].push_back(r.c.k[ci]);
                out_n[ci].push_back(r.c.n[ci]);
            }
            out_khch.push_back(r.c.k_hch);
            out_nhch.push_back(r.c.n_hch);
            out_nsnp.push_back(r.c.n_snp);
        }
    }
    n_unpaired = buf.size();
    bam_destroy1(b);
    }   // GIL reacquired

    bam_hdr_destroy(hdr);
    sam_close(fp);

    auto to_np = [](auto& v) {
        using T = typename std::remove_reference<decltype(v)>::type::value_type;
        py::array_t<T> a((py::ssize_t)v.size());
        if (!v.empty()) std::copy(v.begin(), v.end(), a.mutable_data());
        return a;
    };

    py::dict out;
    out["pair_idx"] = to_np(out_pair);
    out["end_idx"]  = to_np(out_end);
    out["chrom"]    = to_np(out_chrom);
    out["pos_5p"]   = to_np(out_p5);
    out["pos_3p"]   = to_np(out_p3);
    out["is_reverse"]  = to_np(out_rev);
    out["conv_strand"] = to_np(out_conv);
    out["mapq"]        = to_np(out_mapq);
    out["frag_len"]    = to_np(out_frag);
    out["contact_dist"] = to_np(out_dist);
    out["is_cis"]       = to_np(out_cis);
    py::list ks, ns;
    for (int ci = 0; ci < n_ctx_sets; ++ci) { ks.append(to_np(out_k[ci])); ns.append(to_np(out_n[ci])); }
    out["k"] = ks;
    out["n"] = ns;
    out["k_hch"] = to_np(out_khch);
    out["n_hch"] = to_np(out_nhch);
    out["n_masked_snp"] = to_np(out_nsnp);
    if (emit_read_id) out["read_id"] = py::cast(out_qname);

    py::list names;
    for (const ChromCtx& c : by_tid) names.append(c.name);
    out["chrom_names"] = names;
    py::dict st;
    st["n_reads_kept"] = py::int_(n_reads);
    st["n_pairs"] = py::int_(n_pairs);
    st["n_unpaired"] = py::int_(n_unpaired);
    st["n_no_md"] = py::int_(n_no_md);
    st["n_strand_unknown"] = py::int_(n_strand_unknown);
    st["n_below_min_distance"] = py::int_(n_short);
    st["peak_buffered_reads"] = py::int_((uint64_t)peak_buffer);
    out["stats"] = st;
    return out;
}

/* Exposed so a test can check the MD/CIGAR reconstruction against pysam's
 * get_reference_sequence() on real alignments rather than trusting it. */
static py::tuple reference_span(const std::string& bam_path, int64_t which)
{
    samFile* fp = sam_open(bam_path.c_str(), "r");
    if (!fp) throw std::runtime_error("cannot open BAM: " + bam_path);
    bam_hdr_t* hdr = sam_hdr_read(fp);
    bam1_t* b = bam_init1();
    std::string span; std::vector<int32_t> r2q;
    int64_t i = 0;
    bool found = false;
    while (sam_read1(fp, hdr, b) >= 0) {
        if (b->core.flag & (FLAG_UNMAPPED | FLAG_SECONDARY | FLAG_SUPPLEMENTARY)) continue;
        if (i++ != which) continue;
        found = reconstruct_reference(b, span, r2q);
        break;
    }
    bam_destroy1(b); bam_hdr_destroy(hdr); sam_close(fp);
    if (!found) return py::make_tuple(py::none(), py::none());
    return py::make_tuple(span, py::cast(r2q));
}

PYBIND11_MODULE(_joint_state_core, m) {
    m.doc() = "Joint multi-context per-read-end extraction (design Part A).";
    m.def("parse_cell_joint_ends", &parse_cell_joint_ends,
          py::arg("bam_path"), py::arg("contexts"),
          py::arg("snp_positions") = py::dict(),
          py::arg("trim_ends") = 5, py::arg("mapq_min") = 30,
          py::arg("baseq_min") = 0, py::arg("strand_min_evidence") = 1,
          py::arg("keep_supplementary") = false,
          py::arg("count_hch") = true, py::arg("emit_read_id") = true,
          py::arg("min_distance") = 1000, py::arg("cis_only") = true);
    m.def("reference_span", &reference_span,
          py::arg("bam_path"), py::arg("which") = 0);
}
