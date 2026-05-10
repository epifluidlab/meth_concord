/*
 * C++ / htslib backend for BAM parsing.
 *
 * Replaces the pysam-based Python loop with direct htslib calls.
 * For each cell's BAM, processes one chromosome at a time:
 *   1. Reads through all records with hts_itr_next
 *   2. Filters on flag, MAPQ, intra-chromosomal
 *   3. Pairs mates via a hash map keyed by query name
 *   4. For each pair, counts methylation at context sites using
 *      binary search + direct 4-bit sequence access
 *
 * Compiled as a Python extension via pybind11.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <htslib/sam.h>
#include <htslib/hts.h>

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

// Extract a base from the 4-bit packed BAM sequence.
// Returns 'A','C','G','T','N' (uppercase).
static inline char bam_base_at(const uint8_t* seq, int i) {
    static const char TABLE[] = "=ACMGRSVTWYHKDBN";
    uint8_t nib = bam_seqi(seq, i);
    return TABLE[nib];
}

// Check whether the CIGAR has any I or D operations.
static inline bool cigar_has_indel(const uint32_t* cigar, int n_cigar) {
    for (int i = 0; i < n_cigar; ++i) {
        int op = bam_cigar_op(cigar[i]);
        if (op == BAM_CINS || op == BAM_CDEL) return true;
    }
    return false;
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
        // Reference-consuming ops: M, D, N, =, X
        if (op == BAM_CMATCH || op == BAM_CDEL || op == BAM_CREF_SKIP ||
            op == BAM_CEQUAL || op == BAM_CDIFF)
            ref_len += len;
        // Query-consuming ops: M, I, =, X
        if (op == BAM_CMATCH || op == BAM_CINS ||
            op == BAM_CEQUAL || op == BAM_CDIFF)
            qry_len += len;
    }
}

// --------------------------------------------------------------------------
// Per-read context counting (the hot function)
// --------------------------------------------------------------------------

// Count methylation at context sites covered by a read.
//
// ctx_pos:    sorted array of context-site positions (int64)
// ctx_ref:    reference base at each site (uint8: 'C'=67, 'G'=71)
// n_ctx:      number of context sites
// b:          bam1_t* for the read
// trim_ends:  bases to trim from each end
//
// Returns (k_methylated, n_total) via output parameters.
static void count_context(
    const int64_t* ctx_pos, const uint8_t* ctx_ref, int64_t n_ctx,
    const bam1_t* b, int trim_ends,
    int& k_out, int& n_out)
{
    k_out = 0; n_out = 0;

    int64_t ref_start = b->core.pos + trim_ends;
    int64_t ref_end_raw = bam_endpos(b);
    int64_t ref_end = ref_end_raw - trim_ends;
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
        // Fast path: no indels — direct offset calculation.
        int offset = query_alignment_start(cigar, n_cigar) - (int)b->core.pos;

        for (int j = i0; j < i1; ++j) {
            int qp = (int)ctx_pos[j] + offset;
            if (qp < 0 || qp >= seq_len) continue;

            char base = bam_base_at(seq, qp);
            uint8_t rb = ctx_ref[j];

            if (rb == 67) {        // 'C', + strand site
                if (base == 'C')      { ++k_out; ++n_out; }
                else if (base == 'T') { ++n_out; }
            } else if (rb == 71) { // 'G', − strand site
                if (base == 'G')      { ++k_out; ++n_out; }
                else if (base == 'A') { ++n_out; }
            }
        }
    } else {
        // Slow path: build ref→query mapping from CIGAR.
        // Walk the CIGAR to map reference positions to query positions.
        std::unordered_map<int64_t, int> ref_to_query;
        ref_to_query.reserve(ref_len);
        int qpos = query_alignment_start(cigar, n_cigar);
        int rpos = b->core.pos;

        for (int ci = 0; ci < n_cigar; ++ci) {
            int op  = bam_cigar_op(cigar[ci]);
            int len = bam_cigar_oplen(cigar[ci]);

            if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
                for (int x = 0; x < len; ++x) {
                    ref_to_query[rpos + x] = qpos + x;
                }
                rpos += len; qpos += len;
            } else if (op == BAM_CINS) {
                qpos += len;
            } else if (op == BAM_CDEL || op == BAM_CREF_SKIP) {
                rpos += len;
            } else if (op == BAM_CSOFT_CLIP || op == BAM_CHARD_CLIP) {
                // Already accounted for in query_alignment_start for leading clips.
                // Trailing clips don't affect the mapping.
                if (ci > 0 && op == BAM_CSOFT_CLIP) {
                    // trailing soft clip — skip
                }
            }
        }

        for (int j = i0; j < i1; ++j) {
            auto it = ref_to_query.find(ctx_pos[j]);
            if (it == ref_to_query.end()) continue;
            int qp = it->second;
            if (qp < 0 || qp >= seq_len) continue;

            char base = bam_base_at(seq, qp);
            uint8_t rb = ctx_ref[j];

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

// Lightweight struct to hold a read's essential fields for mate pairing.
// We copy the minimum needed to avoid keeping the full bam1_t alive.
struct ReadInfo {
    int32_t  tid;
    int64_t  pos;         // reference_start
    int64_t  end_pos;     // reference_end (exclusive)
    int      mapq;
    int      k;           // methylated count
    int      n;           // total informative count

    ReadInfo() : tid(-1), pos(0), end_pos(0), mapq(0), k(0), n(0) {}
};

// Parse a single cell's BAM file and return columnar numpy arrays.
//
// context_positions: dict of { chrom_str: (pos_array_int64, ref_base_array_uint8) }
//
// Returns a dict with numpy arrays: chrom, pos1, k1, n1, pos2, k2, n2, distance
static py::dict parse_cell_bam_cpp(
    const std::string& bam_path,
    py::dict context_positions,
    int trim_ends,
    int mapq_min)
{
    // Open BAM file.
    htsFile* fp = hts_open(bam_path.c_str(), "r");
    if (!fp) {
        throw std::runtime_error("Cannot open BAM file: " + bam_path);
    }

    sam_hdr_t* hdr = sam_hdr_read(fp);
    if (!hdr) {
        hts_close(fp);
        throw std::runtime_error("Cannot read BAM header: " + bam_path);
    }

    hts_idx_t* idx = sam_index_load(fp, bam_path.c_str());
    if (!idx) {
        sam_hdr_destroy(hdr);
        hts_close(fp);
        throw std::runtime_error("Cannot load BAM index for: " + bam_path);
    }

    // Output collectors.
    std::vector<std::string> out_chrom;
    std::vector<int64_t> out_pos1, out_pos2, out_dist;
    std::vector<int32_t> out_k1, out_n1, out_k2, out_n2;

    // Reserve space (rough estimate: ~2M pairs per cell).
    size_t reserve = 2000000;
    out_chrom.reserve(reserve);
    out_pos1.reserve(reserve); out_pos2.reserve(reserve);
    out_k1.reserve(reserve);   out_n1.reserve(reserve);
    out_k2.reserve(reserve);   out_n2.reserve(reserve);
    out_dist.reserve(reserve);

    bam1_t* b = bam_init1();

    // Process per-chromosome.
    for (auto item : context_positions) {
        std::string chrom = py::cast<std::string>(item.first);
        py::tuple arrays = py::cast<py::tuple>(item.second);

        // Unpack numpy arrays.
        py::array_t<int64_t> pos_arr = arrays[0].cast<py::array_t<int64_t>>();
        py::array_t<uint8_t> ref_arr = arrays[1].cast<py::array_t<uint8_t>>();

        auto pos_buf = pos_arr.unchecked<1>();
        auto ref_buf = ref_arr.unchecked<1>();
        int64_t n_ctx = pos_buf.shape(0);
        if (n_ctx == 0) continue;

        const int64_t* ctx_pos = pos_buf.data(0);
        const uint8_t* ctx_ref = ref_buf.data(0);

        // Get chromosome tid from BAM header.
        int tid = sam_hdr_name2tid(hdr, chrom.c_str());
        if (tid < 0) continue;  // Chromosome not in BAM.

        // Mate-pairing buffer: qname → ReadInfo.
        std::unordered_map<std::string, ReadInfo> read_buffer;
        read_buffer.reserve(500000);

        // Iterate reads on this chromosome.
        hts_itr_t* iter = sam_itr_queryi(idx, tid, 0, INT64_MAX);
        if (!iter) continue;

        while (sam_itr_next(fp, iter, b) >= 0) {
            uint16_t flag = b->core.flag;

            // Combined flag filter.
            if (flag & SKIP_FLAGS) continue;
            if (!(flag & FLAG_PROPER_PAIR)) continue;
            if (b->core.qual < mapq_min) continue;

            // Intra-chromosomal only.
            if (b->core.mtid != tid) continue;

            // Get query name.
            const char* qname = bam_get_qname(b);

            auto it = read_buffer.find(qname);
            if (it == read_buffer.end()) {
                // First mate seen — count context sites and store.
                ReadInfo ri;
                ri.tid = tid;
                ri.pos = b->core.pos;
                ri.end_pos = bam_endpos(b);
                ri.mapq = b->core.qual;
                count_context(ctx_pos, ctx_ref, n_ctx, b, trim_ends, ri.k, ri.n);
                read_buffer.emplace(std::string(qname), ri);
                continue;
            }

            // Second mate — retrieve and remove buffered mate.
            ReadInfo mate = it->second;
            read_buffer.erase(it);

            // Re-check mate MAPQ.
            if (mate.mapq < mapq_min) continue;

            // Count context for this read.
            int k2, n2;
            count_context(ctx_pos, ctx_ref, n_ctx, b, trim_ends, k2, n2);

            // Canonical ordering: r1 has smaller pos.
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
            int64_t dist = std::abs(mid2 - mid1);

            out_chrom.push_back(chrom);
            out_pos1.push_back(mid1);
            out_k1.push_back(r1_k);
            out_n1.push_back(r1_n);
            out_pos2.push_back(mid2);
            out_k2.push_back(r2_k);
            out_n2.push_back(r2_n);
            out_dist.push_back(dist);
        }

        hts_itr_destroy(iter);
        read_buffer.clear();
    }

    bam_destroy1(b);
    hts_idx_destroy(idx);
    sam_hdr_destroy(hdr);
    hts_close(fp);

    // Convert to numpy arrays.
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

    // Build chrom string list.
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

PYBIND11_MODULE(_bam_parser_core, m) {
    m.doc() = "C++/htslib backend for BAM read-pair context counting";

    m.def("parse_cell_bam", &parse_cell_bam_cpp,
          py::arg("bam_path"),
          py::arg("context_positions"),
          py::arg("trim_ends") = 5,
          py::arg("mapq_min") = 30,
          R"doc(
          Parse a single cell's BAM file using htslib.

          Parameters
          ----------
          bam_path : str
              Path to coordinate-sorted, indexed BAM.
          context_positions : dict
              {chrom: (pos_array_int64, ref_base_array_uint8)}
          trim_ends : int
          mapq_min : int

          Returns
          -------
          dict with numpy arrays: chrom, pos1, k1, n1, pos2, k2, n2, distance
          )doc");
}
