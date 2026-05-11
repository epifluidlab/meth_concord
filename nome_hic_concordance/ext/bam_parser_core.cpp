/*
 * C++ / htslib backend for BAM parsing.
 *
 * Replaces the pysam-based Python loop with direct htslib calls.
 * For each cell's BAM, processes one chromosome at a time:
 *   1. Reads through all records with hts_itr_next
 *   2. Filters on flag, MAPQ, intra-chromosomal
 *   3. Infers bisulfite conversion strand (C→T vs G→A) from the MD tag
 *   4. Pairs mates via a hash map keyed by query name
 *   5. For each pair, counts methylation at context sites (only from
 *      the informative strand) using binary search + direct 4-bit
 *      sequence access
 *
 * Compiled as a Python extension via pybind11.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <htslib/sam.h>
#include <htslib/hts.h>

#include <cctype>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>

namespace py = pybind11;

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

// BAM flag bits
static constexpr uint16_t FLAG_UNMAPPED      = 0x4;
static constexpr uint16_t FLAG_MATE_UNMAPPED = 0x8;
static constexpr uint16_t FLAG_PROPER_PAIR   = 0x2;
static constexpr uint16_t FLAG_SECONDARY     = 0x100;
static constexpr uint16_t FLAG_DUPLICATE     = 0x400;
static constexpr uint16_t FLAG_SUPPLEMENTARY = 0x800;
static constexpr uint16_t SKIP_FLAGS =
    FLAG_UNMAPPED | FLAG_MATE_UNMAPPED |
    FLAG_SECONDARY | FLAG_DUPLICATE | FLAG_SUPPLEMENTARY;

// Conversion strand constants
static constexpr int STRAND_CT      = 1;  // C→T conversion
static constexpr int STRAND_GA      = 2;  // G→A conversion
static constexpr int STRAND_UNKNOWN = 0;

// Extract a base from the 4-bit packed BAM sequence.
// Returns 'A','C','G','T','N' (uppercase).
static inline char bam_base_at(const uint8_t* seq, int i) {
    static const char TABLE[] = "=ACMGRSVTWYHKDBN";
    uint8_t nib = bam_seqi(seq, i);
    return TABLE[nib];
}

// Compute the query alignment start (number of soft/hard clipped bases
// at the left end).
static inline int query_alignment_start(const uint32_t* cigar, int n_cigar) {
    int qstart = 0;
    for (int i = 0; i < n_cigar; ++i) {
        int op = bam_cigar_op(cigar[i]);
        if (op == BAM_CSOFT_CLIP || op == BAM_CHARD_CLIP)
            qstart += bam_cigar_oplen(cigar[i]);
        else
            break;
    }
    return qstart;
}

// Compute reference_length and query_alignment_length from CIGAR.
static inline void cigar_lengths(const uint32_t* cigar, int n_cigar,
                                  int& ref_len, int& qry_len) {
    ref_len = 0; qry_len = 0;
    for (int i = 0; i < n_cigar; ++i) {
        int op  = bam_cigar_op(cigar[i]);
        int len = bam_cigar_oplen(cigar[i]);
        if (op == BAM_CMATCH || op == BAM_CDEL || op == BAM_CREF_SKIP ||
            op == BAM_CEQUAL || op == BAM_CDIFF)
            ref_len += len;
        if (op == BAM_CMATCH || op == BAM_CINS ||
            op == BAM_CEQUAL || op == BAM_CDIFF)
            qry_len += len;
    }
}

// --------------------------------------------------------------------------
// Bisulfite conversion strand inference from the MD tag
// --------------------------------------------------------------------------

// Two-step approach (simple and correct):
//
// Step 1: Walk the CIGAR to build a ref_to_query[] array that maps
//         each reference-consuming position to its query (read) position.
//         Insertions advance query only; deletions get -1 (no query base).
//
// Step 2: Walk the MD tag.  At each mismatch position, look up the
//         query position from ref_to_query[] and compare the reference
//         base (from MD, uppercased) with the read base (from BAM seq).
//         Count C→T vs G→A.
//
// Returns STRAND_CT, STRAND_GA, or STRAND_UNKNOWN.
static int infer_conversion_strand(const bam1_t* b) {
    const uint8_t* md_raw = bam_aux_get(b, "MD");
    if (!md_raw) return STRAND_UNKNOWN;
    const char* md = bam_aux2Z(md_raw);
    if (!md) return STRAND_UNKNOWN;

    const uint8_t* seq = bam_get_seq(b);
    int seq_len = b->core.l_qseq;
    if (seq_len == 0) return STRAND_UNKNOWN;

    const uint32_t* cigar = bam_get_cigar(b);
    int n_cigar = b->core.n_cigar;

    // ---- Step 1: build ref_to_query mapping from CIGAR ----
    // For each reference-consuming base, store the corresponding query
    // position.  Deletions (ref-only) get -1.
    std::vector<int> ref_to_query;
    {
        int qpos = 0;
        for (int ci = 0; ci < n_cigar; ++ci) {
            int op  = bam_cigar_op(cigar[ci]);
            int len = bam_cigar_oplen(cigar[ci]);
            if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
                for (int x = 0; x < len; ++x)
                    ref_to_query.push_back(qpos++);
            } else if (op == BAM_CINS || op == BAM_CSOFT_CLIP) {
                qpos += len;  // query only
            } else if (op == BAM_CDEL || op == BAM_CREF_SKIP) {
                for (int x = 0; x < len; ++x)
                    ref_to_query.push_back(-1);  // ref only, no query base
            }
            // BAM_CHARD_CLIP: consumes neither
        }
    }
    int ref_len = (int)ref_to_query.size();

    // ---- Step 2: walk MD tag, count C→T / G→A at mismatches ----
    int ct_count = 0;
    int ga_count = 0;
    int ref_idx = 0;   // index into ref_to_query
    const char* p = md;

    while (*p) {
        if (*p >= '0' && *p <= '9') {
            // Matching run — advance ref_idx by the number.
            int n = 0;
            while (*p >= '0' && *p <= '9') {
                n = n * 10 + (*p - '0');
                ++p;
            }
            ref_idx += n;
        } else if (*p == '^') {
            // Deletion: ^SEQ (letters are deleted ref bases).
            ++p;  // skip '^'
            while (*p && !(*p >= '0' && *p <= '9')) {
                ++ref_idx;
                ++p;
            }
        } else {
            // Mismatch: single character = reference base.
            char ref_base = std::toupper((unsigned char)*p);
            ++p;

            if (ref_idx >= 0 && ref_idx < ref_len) {
                int qp = ref_to_query[ref_idx];
                if (qp >= 0 && qp < seq_len) {
                    char read_base = bam_base_at(seq, qp);
                    if (ref_base == 'C' && read_base == 'T') ++ct_count;
                    else if (ref_base == 'G' && read_base == 'A') ++ga_count;
                }
            }
            ++ref_idx;
        }
    }

    if (ct_count > ga_count) return STRAND_CT;
    if (ga_count > ct_count) return STRAND_GA;
    return STRAND_UNKNOWN;
}

// --------------------------------------------------------------------------
// Per-read context counting (strand-aware)
// --------------------------------------------------------------------------

// Count methylation at context sites covered by a read, using only
// sites from the informative strand.
//
// strand:     STRAND_CT → only count ref=C sites
//             STRAND_GA → only count ref=G sites
//             STRAND_UNKNOWN → skip (return 0, 0)
static void count_context(
    const int64_t* ctx_pos, const uint8_t* ctx_ref, int64_t n_ctx,
    const bam1_t* b, int trim_ends, int strand,
    int& k_out, int& n_out)
{
    k_out = 0; n_out = 0;

    if (strand == STRAND_UNKNOWN) return;

    // Which ref_base is informative for this read's conversion strand?
    uint8_t informative_ref = (strand == STRAND_CT) ? 67 : 71; // 'C' or 'G'

    int64_t ref_start = b->core.pos + trim_ends;
    int64_t ref_end = bam_endpos(b) - trim_ends;
    if (ref_start >= ref_end) return;

    // Binary search for context sites in [ref_start, ref_end).
    const int64_t* lo = std::lower_bound(ctx_pos, ctx_pos + n_ctx, ref_start);
    const int64_t* hi = std::lower_bound(ctx_pos, ctx_pos + n_ctx, ref_end);
    if (lo >= hi) return;

    int i0 = (int)(lo - ctx_pos);
    int i1 = (int)(hi - ctx_pos);

    const uint8_t* seq = bam_get_seq(b);
    int seq_len = b->core.l_qseq;
    if (seq_len == 0) return;

    const uint32_t* cigar = bam_get_cigar(b);
    int n_cigar = b->core.n_cigar;

    int ref_len, qry_len;
    cigar_lengths(cigar, n_cigar, ref_len, qry_len);

    if (ref_len == qry_len) {
        // Fast path: no indels.
        int offset = query_alignment_start(cigar, n_cigar) - (int)b->core.pos;

        for (int j = i0; j < i1; ++j) {
            uint8_t rb = ctx_ref[j];
            if (rb != informative_ref) continue;  // wrong strand

            int qp = (int)ctx_pos[j] + offset;
            if (qp < 0 || qp >= seq_len) continue;

            char base = bam_base_at(seq, qp);

            if (rb == 67) {        // ref=C, C→T strand
                if (base == 'C')      { ++k_out; ++n_out; }
                else if (base == 'T') { ++n_out; }
            } else if (rb == 71) { // ref=G, G→A strand
                if (base == 'G')      { ++k_out; ++n_out; }
                else if (base == 'A') { ++n_out; }
            }
        }
    } else {
        // Slow path: build ref→query mapping from CIGAR.
        std::unordered_map<int64_t, int> ref_to_query;
        ref_to_query.reserve(ref_len);
        int qpos = query_alignment_start(cigar, n_cigar);
        int rpos = b->core.pos;

        for (int ci = 0; ci < n_cigar; ++ci) {
            int op  = bam_cigar_op(cigar[ci]);
            int len = bam_cigar_oplen(cigar[ci]);

            if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
                for (int x = 0; x < len; ++x)
                    ref_to_query[rpos + x] = qpos + x;
                rpos += len; qpos += len;
            } else if (op == BAM_CINS) {
                qpos += len;
            } else if (op == BAM_CDEL || op == BAM_CREF_SKIP) {
                rpos += len;
            } else if (op == BAM_CSOFT_CLIP || op == BAM_CHARD_CLIP) {
                if (ci == 0) { /* leading: already in query_alignment_start */ }
                // trailing: no action needed
            }
        }

        for (int j = i0; j < i1; ++j) {
            uint8_t rb = ctx_ref[j];
            if (rb != informative_ref) continue;

            auto it = ref_to_query.find(ctx_pos[j]);
            if (it == ref_to_query.end()) continue;
            int qp = it->second;
            if (qp < 0 || qp >= seq_len) continue;

            char base = bam_base_at(seq, qp);

            if (rb == 67) {
                if (base == 'C')      { ++k_out; ++n_out; }
                else if (base == 'T') { ++n_out; }
            } else if (rb == 71) {
                if (base == 'G')      { ++k_out; ++n_out; }
                else if (base == 'A') { ++n_out; }
            }
        }
    }
}

// --------------------------------------------------------------------------
// Per-cell BAM parser
// --------------------------------------------------------------------------

struct ReadInfo {
    int32_t  tid;
    int64_t  pos;
    int64_t  end_pos;
    int      mapq;
    int      k;
    int      n;

    ReadInfo() : tid(-1), pos(0), end_pos(0), mapq(0), k(0), n(0) {}
};

static py::dict parse_cell_bam_cpp(
    const std::string& bam_path,
    py::dict context_positions,
    int trim_ends,
    int mapq_min)
{
    hts_set_log_level(HTS_LOG_OFF);

    htsFile* fp = hts_open(bam_path.c_str(), "r");
    if (!fp)
        throw std::runtime_error("Cannot open BAM file: " + bam_path);

    sam_hdr_t* hdr = sam_hdr_read(fp);
    if (!hdr) { hts_close(fp); throw std::runtime_error("Cannot read BAM header: " + bam_path); }

    hts_idx_t* idx = sam_index_load(fp, bam_path.c_str());
    if (!idx) { sam_hdr_destroy(hdr); hts_close(fp); throw std::runtime_error("Cannot load BAM index for: " + bam_path); }

    std::vector<std::string> out_chrom;
    std::vector<int64_t> out_pos1, out_pos2, out_dist;
    std::vector<int32_t> out_k1, out_n1, out_k2, out_n2;

    size_t reserve = 2000000;
    out_chrom.reserve(reserve);
    out_pos1.reserve(reserve); out_pos2.reserve(reserve);
    out_k1.reserve(reserve);   out_n1.reserve(reserve);
    out_k2.reserve(reserve);   out_n2.reserve(reserve);
    out_dist.reserve(reserve);

    bam1_t* b = bam_init1();

    for (auto item : context_positions) {
        std::string chrom = py::cast<std::string>(item.first);
        py::tuple arrays = py::cast<py::tuple>(item.second);

        py::array_t<int64_t> pos_arr = arrays[0].cast<py::array_t<int64_t>>();
        py::array_t<uint8_t> ref_arr = arrays[1].cast<py::array_t<uint8_t>>();

        auto pos_buf = pos_arr.unchecked<1>();
        auto ref_buf = ref_arr.unchecked<1>();
        int64_t n_ctx = pos_buf.shape(0);
        if (n_ctx == 0) continue;

        const int64_t* ctx_pos = pos_buf.data(0);
        const uint8_t* ctx_ref = ref_buf.data(0);

        int tid = sam_hdr_name2tid(hdr, chrom.c_str());
        if (tid < 0) continue;

        std::unordered_map<std::string, ReadInfo> read_buffer;
        read_buffer.reserve(500000);

        hts_itr_t* iter = sam_itr_queryi(idx, tid, 0, INT64_MAX);
        if (!iter) continue;

        while (sam_itr_next(fp, iter, b) >= 0) {
            uint16_t flag = b->core.flag;
            if (flag & SKIP_FLAGS) continue;
            if (!(flag & FLAG_PROPER_PAIR)) continue;
            if (b->core.qual < mapq_min) continue;
            if (b->core.mtid != tid) continue;

            // Infer bisulfite conversion strand from MD tag.
            int strand = infer_conversion_strand(b);

            const char* qname = bam_get_qname(b);

            auto it = read_buffer.find(qname);
            if (it == read_buffer.end()) {
                ReadInfo ri;
                ri.tid = tid;
                ri.pos = b->core.pos;
                ri.end_pos = bam_endpos(b);
                ri.mapq = b->core.qual;
                count_context(ctx_pos, ctx_ref, n_ctx, b, trim_ends, strand, ri.k, ri.n);
                read_buffer.emplace(std::string(qname), ri);
                continue;
            }

            ReadInfo mate = it->second;
            read_buffer.erase(it);

            if (mate.mapq < mapq_min) continue;

            int k2, n2;
            count_context(ctx_pos, ctx_ref, n_ctx, b, trim_ends, strand, k2, n2);

            int64_t pos_this = b->core.pos;
            int64_t end_this = bam_endpos(b);

            int64_t r1_pos, r1_end, r2_pos, r2_end;
            int r1_k, r1_n, r2_k, r2_n;

            if (mate.pos <= pos_this) {
                r1_pos = mate.pos; r1_end = mate.end_pos;
                r1_k = mate.k;    r1_n = mate.n;
                r2_pos = pos_this; r2_end = end_this;
                r2_k = k2;        r2_n = n2;
            } else {
                r1_pos = pos_this; r1_end = end_this;
                r1_k = k2;        r1_n = n2;
                r2_pos = mate.pos; r2_end = mate.end_pos;
                r2_k = mate.k;    r2_n = mate.n;
            }

            int64_t mid1 = (r1_pos + r1_end) / 2;
            int64_t mid2 = (r2_pos + r2_end) / 2;

            out_chrom.push_back(chrom);
            out_pos1.push_back(mid1);
            out_k1.push_back(r1_k);
            out_n1.push_back(r1_n);
            out_pos2.push_back(mid2);
            out_k2.push_back(r2_k);
            out_n2.push_back(r2_n);
            out_dist.push_back(std::abs(mid2 - mid1));
        }

        hts_itr_destroy(iter);
        read_buffer.clear();
    }

    bam_destroy1(b);
    hts_idx_destroy(idx);
    sam_hdr_destroy(hdr);
    hts_close(fp);

    size_t N = out_pos1.size();
    py::array_t<int64_t> np_pos1(N), np_pos2(N), np_dist(N);
    py::array_t<int32_t> np_k1(N), np_n1(N), np_k2(N), np_n2(N);

    if (N > 0) {
        std::memcpy(np_pos1.mutable_data(), out_pos1.data(), N * sizeof(int64_t));
        std::memcpy(np_pos2.mutable_data(), out_pos2.data(), N * sizeof(int64_t));
        std::memcpy(np_dist.mutable_data(), out_dist.data(), N * sizeof(int64_t));
        std::memcpy(np_k1.mutable_data(),   out_k1.data(),   N * sizeof(int32_t));
        std::memcpy(np_n1.mutable_data(),   out_n1.data(),   N * sizeof(int32_t));
        std::memcpy(np_k2.mutable_data(),   out_k2.data(),   N * sizeof(int32_t));
        std::memcpy(np_n2.mutable_data(),   out_n2.data(),   N * sizeof(int32_t));
    }

    py::list chrom_list;
    for (auto& s : out_chrom) chrom_list.append(s);

    py::dict result;
    result["chrom"]    = chrom_list;
    result["pos1"]     = np_pos1;
    result["k1"]       = np_k1;
    result["n1"]       = np_n1;
    result["pos2"]     = np_pos2;
    result["k2"]       = np_k2;
    result["n2"]       = np_n2;
    result["distance"] = np_dist;

    return result;
}

// --------------------------------------------------------------------------
// pybind11 module
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Diagnostic: expose strand inference internals for debugging
// --------------------------------------------------------------------------

static py::list debug_strand_inference(const std::string& bam_path,
                                        const std::string& chrom,
                                        int max_reads)
{
    hts_set_log_level(HTS_LOG_OFF);
    htsFile* fp = hts_open(bam_path.c_str(), "r");
    if (!fp) throw std::runtime_error("Cannot open BAM: " + bam_path);
    sam_hdr_t* hdr = sam_hdr_read(fp);
    hts_idx_t* idx = sam_index_load(fp, bam_path.c_str());
    if (!hdr || !idx) { hts_close(fp); throw std::runtime_error("Cannot read header/index"); }

    int tid = sam_hdr_name2tid(hdr, chrom.c_str());
    if (tid < 0) { hts_close(fp); throw std::runtime_error("Chrom not found: " + chrom); }

    bam1_t* b = bam_init1();
    hts_itr_t* iter = sam_itr_queryi(idx, tid, 0, INT64_MAX);
    py::list results;
    int count = 0;

    while (iter && sam_itr_next(fp, iter, b) >= 0 && count < max_reads) {
        uint16_t flag = b->core.flag;
        if (flag & SKIP_FLAGS) continue;
        if (!(flag & FLAG_PROPER_PAIR)) continue;
        if (b->core.qual < 30) continue;

        // Get MD tag
        const uint8_t* md_raw = bam_aux_get(b, "MD");
        bool has_md = (md_raw != nullptr);
        std::string md_str;
        if (has_md) {
            const char* md = bam_aux2Z(md_raw);
            if (md) md_str = md;
            else { has_md = false; md_str = "bam_aux2Z_failed"; }
        }

        // Run strand inference
        int strand = infer_conversion_strand(b);

        // Also manually count C->T and G->A for verification
        int manual_ct = 0, manual_ga = 0;
        if (has_md && md_str != "bam_aux2Z_failed") {
            const uint8_t* seq = bam_get_seq(b);
            int seq_len = b->core.l_qseq;
            const uint32_t* cigar = bam_get_cigar(b);
            int n_cigar = b->core.n_cigar;

            // Build ref_to_query (same as infer_conversion_strand)
            std::vector<int> r2q;
            int qp = 0;
            for (int ci = 0; ci < n_cigar; ++ci) {
                int op = bam_cigar_op(cigar[ci]);
                int len = bam_cigar_oplen(cigar[ci]);
                if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
                    for (int x = 0; x < len; ++x) r2q.push_back(qp++);
                } else if (op == BAM_CINS || op == BAM_CSOFT_CLIP) {
                    qp += len;
                } else if (op == BAM_CDEL || op == BAM_CREF_SKIP) {
                    for (int x = 0; x < len; ++x) r2q.push_back(-1);
                }
            }

            // Walk MD
            const char* p = md_str.c_str();
            int ri = 0;
            while (*p) {
                if (*p >= '0' && *p <= '9') {
                    int n = 0;
                    while (*p >= '0' && *p <= '9') { n = n*10 + (*p-'0'); ++p; }
                    ri += n;
                } else if (*p == '^') {
                    ++p;
                    while (*p && !(*p >= '0' && *p <= '9')) { ++ri; ++p; }
                } else {
                    char rb = std::toupper((unsigned char)*p); ++p;
                    if (ri >= 0 && ri < (int)r2q.size()) {
                        int q = r2q[ri];
                        if (q >= 0 && q < seq_len) {
                            char rdbase = bam_base_at(seq, q);
                            if (rb == 'C' && rdbase == 'T') ++manual_ct;
                            else if (rb == 'G' && rdbase == 'A') ++manual_ga;
                        }
                    }
                    ++ri;
                }
            }
        }

        py::dict d;
        d["qname"] = std::string(bam_get_qname(b));
        d["has_md"] = has_md;
        d["md"] = md_str.substr(0, 60);  // truncate for readability
        d["strand"] = (strand == STRAND_CT ? "CT" :
                       strand == STRAND_GA ? "GA" : "unknown");
        d["manual_ct"] = manual_ct;
        d["manual_ga"] = manual_ga;
        results.append(d);
        ++count;
    }

    if (iter) hts_itr_destroy(iter);
    bam_destroy1(b);
    hts_idx_destroy(idx);
    sam_hdr_destroy(hdr);
    hts_close(fp);
    return results;
}


PYBIND11_MODULE(_bam_parser_core, m) {
    m.doc() = "C++/htslib backend for BAM read-pair context counting (strand-aware)";

    m.def("parse_cell_bam", &parse_cell_bam_cpp,
          py::arg("bam_path"),
          py::arg("context_positions"),
          py::arg("trim_ends") = 5,
          py::arg("mapq_min") = 30,
          R"doc(
          Parse a single cell's BAM file using htslib with bisulfite
          strand inference from the MD tag.

          Parameters
          ----------
          bam_path : str
          context_positions : dict  {chrom: (pos_int64, ref_base_uint8)}
          trim_ends : int
          mapq_min : int

          Returns
          -------
          dict with numpy arrays: chrom, pos1, k1, n1, pos2, k2, n2, distance
          )doc");

    m.def("debug_strand_inference", &debug_strand_inference,
          py::arg("bam_path"),
          py::arg("chrom"),
          py::arg("max_reads") = 10,
          "Debug: return strand inference details for the first N reads.");
}
