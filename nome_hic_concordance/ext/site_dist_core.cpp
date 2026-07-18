/*
 * C++ / htslib backend for the site-level distance-correlation analysis.
 *
 * The whole hot path lives here, not just BAM reading: with ~1% of pairs being
 * long-range, the analysis needs FULL BAMs (~1e6 pairs/cell), and a Python
 * per-pair loop would dominate no matter how fast the parsing is.  So for each
 * cell this does
 *
 *   1. htslib read iteration + flag/MAPQ/cis filtering + mate pairing
 *   2. per-cytosine (position, 0/1) extraction, MD-tag strand inference
 *   3. site-pair enumeration (within-read AND cross-end)
 *   4. log-distance binning
 *   5. accumulation of Pearson sufficient statistics
 *   6. the ±window non-linked null
 *
 * and returns only the small accumulator arrays.  Filtering and the per-site
 * rules mirror bam_parser_core.cpp exactly, so a read contributing (k, n) there
 * contributes those same sites here.
 *
 * Compiled as a Python extension via pybind11.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <htslib/sam.h>
#include <htslib/hts.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

static constexpr uint16_t FLAG_UNMAPPED      = 0x4;
static constexpr uint16_t FLAG_MATE_UNMAPPED = 0x8;
static constexpr uint16_t FLAG_SECONDARY     = 0x100;
static constexpr uint16_t FLAG_DUPLICATE     = 0x400;
static constexpr uint16_t FLAG_SUPPLEMENTARY = 0x800;
static constexpr uint16_t SKIP_FLAGS_BASE =
    FLAG_UNMAPPED | FLAG_MATE_UNMAPPED | FLAG_SECONDARY | FLAG_DUPLICATE;
static constexpr uint16_t SKIP_FLAGS = SKIP_FLAGS_BASE | FLAG_SUPPLEMENTARY;

static constexpr int STRAND_CT = 1, STRAND_GA = 2, STRAND_UNKNOWN = 0;

static inline char bam_base_at(const uint8_t* seq, int i) {
    return "=ACMGRSVTWYHKDBN"[bam_seqi(seq, i)];
}

static inline int query_alignment_start(const uint32_t* cigar, int n_cigar) {
    int q = 0;
    for (int i = 0; i < n_cigar; ++i) {
        int op = bam_cigar_op(cigar[i]);
        if (op == BAM_CSOFT_CLIP) q += bam_cigar_oplen(cigar[i]);
        else if (op != BAM_CHARD_CLIP) break;
    }
    return q;
}

static inline void cigar_lengths(const uint32_t* cigar, int n_cigar,
                                 int& ref_len, int& qry_len) {
    ref_len = 0; qry_len = 0;
    for (int i = 0; i < n_cigar; ++i) {
        int op = bam_cigar_op(cigar[i]), len = bam_cigar_oplen(cigar[i]);
        if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
            ref_len += len; qry_len += len;
        } else if (op == BAM_CINS) { qry_len += len; }
        else if (op == BAM_CDEL || op == BAM_CREF_SKIP) { ref_len += len; }
        else if (op == BAM_CSOFT_CLIP) { qry_len += len; }
    }
}

/* Infer C→T vs G→A from the MD tag, exactly as bam_parser_core does. */
static int infer_conversion_strand(const bam1_t* b) {
    const uint8_t* md_aux = bam_aux_get(b, "MD");
    if (!md_aux) return STRAND_UNKNOWN;
    const char* md = bam_aux2Z(md_aux);
    if (!md) return STRAND_UNKNOWN;

    const uint8_t* seq = bam_get_seq(b);
    const uint32_t* cigar = bam_get_cigar(b);
    int n_cigar = b->core.n_cigar;
    int qpos = query_alignment_start(cigar, n_cigar);
    int seq_len = b->core.l_qseq;

    int ct = 0, ga = 0;
    const char* p = md;
    while (*p) {
        if (*p >= '0' && *p <= '9') {
            long run = strtol(p, (char**)&p, 10);
            qpos += (int)run;
        } else if (*p == '^') {                 // deletion: skip ref bases
            ++p;
            while (*p && ((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z'))) ++p;
        } else {
            char refb = (char)toupper((unsigned char)*p);
            if (qpos >= 0 && qpos < seq_len) {
                char qb = bam_base_at(seq, qpos);
                if (refb == 'C' && qb == 'T') ++ct;
                else if (refb == 'G' && qb == 'A') ++ga;
            }
            ++qpos;
            ++p;
        }
    }
    if (ct > ga) return STRAND_CT;
    if (ga > ct) return STRAND_GA;
    return STRAND_UNKNOWN;
}

struct SiteVec {
    std::vector<int64_t> pos;
    std::vector<uint8_t> st;
    inline size_t size() const { return pos.size(); }
    inline void clear() { pos.clear(); st.clear(); }
};

/* Per-site analogue of count_context(): emits (position, state) per site.
 * ``baseq_min`` drops individual low-quality basecalls — a miscalled base is a
 * miscalled methylation state, and unlike coverage there is no downstream step
 * that can recover from it. */
static void site_calls(const int64_t* ctx_pos, const uint8_t* ctx_ref,
                       int64_t n_ctx, const bam1_t* b, int trim_ends,
                       int strand, int baseq_min, SiteVec& out) {
    out.clear();
    if (strand == STRAND_UNKNOWN) return;
    uint8_t informative_ref = (strand == STRAND_CT) ? 67 : 71;
    const uint8_t* qual = bam_get_qual(b);
    const bool have_qual = (qual != nullptr && qual[0] != 0xff);

    int64_t ref_start = b->core.pos + trim_ends;
    int64_t ref_end = bam_endpos(b) - trim_ends;
    if (ref_start >= ref_end) return;

    const int64_t* lo = std::lower_bound(ctx_pos, ctx_pos + n_ctx, ref_start);
    const int64_t* hi = std::lower_bound(ctx_pos, ctx_pos + n_ctx, ref_end);
    if (lo >= hi) return;
    int i0 = (int)(lo - ctx_pos), i1 = (int)(hi - ctx_pos);

    const uint8_t* seq = bam_get_seq(b);
    int seq_len = b->core.l_qseq;
    if (seq_len == 0) return;
    const uint32_t* cigar = bam_get_cigar(b);
    int n_cigar = b->core.n_cigar;
    int ref_len, qry_len;
    cigar_lengths(cigar, n_cigar, ref_len, qry_len);

    auto emit = [&](int64_t p, char base, uint8_t rb) {
        if (rb == 67) {
            if (base == 'C') { out.pos.push_back(p); out.st.push_back(1); }
            else if (base == 'T') { out.pos.push_back(p); out.st.push_back(0); }
        } else {
            if (base == 'G') { out.pos.push_back(p); out.st.push_back(1); }
            else if (base == 'A') { out.pos.push_back(p); out.st.push_back(0); }
        }
    };

    if (ref_len == qry_len) {
        int offset = query_alignment_start(cigar, n_cigar) - (int)b->core.pos;
        for (int j = i0; j < i1; ++j) {
            uint8_t rb = ctx_ref[j];
            if (rb != informative_ref) continue;
            int64_t qp = ctx_pos[j] + offset;
            if (qp < 0 || qp >= seq_len) continue;
            if (baseq_min > 0 && have_qual && qual[qp] < baseq_min) continue;
            emit(ctx_pos[j], bam_base_at(seq, (int)qp), rb);
        }
    } else {
        std::unordered_map<int64_t, int> ref_to_query;
        ref_to_query.reserve(ref_len);
        int qpos = query_alignment_start(cigar, n_cigar);
        int64_t rpos = b->core.pos;
        for (int ci = 0; ci < n_cigar; ++ci) {
            int op = bam_cigar_op(cigar[ci]), len = bam_cigar_oplen(cigar[ci]);
            if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
                for (int x = 0; x < len; ++x) ref_to_query[rpos + x] = qpos + x;
                rpos += len; qpos += len;
            } else if (op == BAM_CINS) { qpos += len; }
            else if (op == BAM_CDEL || op == BAM_CREF_SKIP) { rpos += len; }
        }
        for (int j = i0; j < i1; ++j) {
            uint8_t rb = ctx_ref[j];
            if (rb != informative_ref) continue;
            auto it = ref_to_query.find(ctx_pos[j]);
            if (it == ref_to_query.end()) continue;
            int qp = it->second;
            if (qp < 0 || qp >= seq_len) continue;
            if (baseq_min > 0 && have_qual && qual[qp] < baseq_min) continue;
            emit(ctx_pos[j], bam_base_at(seq, qp), rb);
        }
    }
}

/* Pearson sufficient statistics per distance bin: n, Sx, Sy, Sxx, Syy, Sxy. */
struct Acc {
    std::vector<double> v;
    int nb;
    explicit Acc(int n) : v(6 * (size_t)n, 0.0), nb(n) {}
    inline void add(int b, double x, double y) {
        v[(size_t)0 * nb + b] += 1.0;
        v[(size_t)1 * nb + b] += x;
        v[(size_t)2 * nb + b] += y;
        v[(size_t)3 * nb + b] += x * x;
        v[(size_t)4 * nb + b] += y * y;
        v[(size_t)5 * nb + b] += x * y;
    }
};

static inline int bin_of(const std::vector<double>& edges, int64_t d) {
    double dd = (double)d;
    if (dd < edges.front() || dd > edges.back()) return -1;
    int idx = (int)(std::upper_bound(edges.begin(), edges.end(), dd)
                    - edges.begin()) - 1;
    if (idx < 0) idx = 0;
    if (idx > (int)edges.size() - 2) idx = (int)edges.size() - 2;
    return idx;
}

/* Enumerate (or randomly sample, when capped) site pairs and accumulate. */
static void pair_accumulate(Acc& acc, const SiteVec& A, const SiteVec& B,
                            bool same, const std::vector<double>& edges,
                            int cap, std::mt19937_64& rng) {
    size_t na = A.size(), nbs = B.size();
    if (na == 0 || nbs == 0) return;
    size_t total = same ? (na * (na - 1)) / 2 : na * nbs;
    if (total == 0) return;

    auto one = [&](size_t i, size_t j) {
        int64_t d = B.pos[j] > A.pos[i] ? B.pos[j] - A.pos[i]
                                        : A.pos[i] - B.pos[j];
        int b = bin_of(edges, d);
        if (b >= 0) acc.add(b, (double)A.st[i], (double)B.st[j]);
    };

    if (cap > 0 && total > (size_t)cap) {
        for (int t = 0; t < cap; ++t) {
            size_t i = rng() % na, j = rng() % nbs;
            if (same) {
                if (i == j) continue;
                if (i > j) std::swap(i, j);
            }
            one(i, j);
        }
        return;
    }
    if (same) {
        for (size_t i = 0; i < na; ++i)
            for (size_t j = i + 1; j < na; ++j) one(i, j);
    } else {
        for (size_t i = 0; i < na; ++i)
            for (size_t j = 0; j < nbs; ++j) one(i, j);
    }
}

struct PairRec {
    int64_t mid1, mid2;
    SiteVec g1, g2, h1, h2;
};

/* Accumulate every measure for one (end1, end2) record pair. */
static void add_pair(Acc& gg, Acc& hh, Acc& gh,
                     const SiteVec& g1, const SiteVec& g2,
                     const SiteVec& h1, const SiteVec& h2,
                     const std::vector<double>& edges, int cap,
                     std::mt19937_64& rng, bool cross_only) {
    if (!cross_only) {                    // WITHIN-read arm (short range)
        pair_accumulate(gg, g1, g1, true, edges, cap, rng);
        pair_accumulate(gg, g2, g2, true, edges, cap, rng);
        pair_accumulate(hh, h1, h1, true, edges, cap, rng);
        pair_accumulate(hh, h2, h2, true, edges, cap, rng);
        pair_accumulate(gh, h1, g1, false, edges, cap, rng);
        pair_accumulate(gh, h2, g2, false, edges, cap, rng);
    }
    // CROSS-end arm (long range)
    pair_accumulate(gg, g1, g2, false, edges, cap, rng);
    pair_accumulate(hh, h1, h2, false, edges, cap, rng);
    pair_accumulate(gh, h1, g2, false, edges, cap, rng);
    pair_accumulate(gh, h2, g1, false, edges, cap, rng);
}

static py::array_t<double> to_numpy(const Acc& a) {
    py::array_t<double> out({(py::ssize_t)6, (py::ssize_t)a.nb});
    std::copy(a.v.begin(), a.v.end(), out.mutable_data());
    return out;
}

/* True iff [start, end) hits any interval of a sorted, merged blacklist. */
static inline bool hits_blacklist(const std::vector<int64_t>& starts,
                                  const std::vector<int64_t>& ends,
                                  int64_t s, int64_t e) {
    if (starts.empty()) return false;
    size_t i = (size_t)(std::upper_bound(starts.begin(), starts.end(), e - 1)
                        - starts.begin());
    return i > 0 && ends[i - 1] > s;      // merged ⇒ only the predecessor can hit
}

static py::dict parse_cell_site_distance(
    const std::string& bam_path,
    py::dict ctx_gch, py::dict ctx_hcg,
    std::vector<double> edges,
    int trim_ends, int mapq_min, int null_window,
    int max_site_pairs_per_read, uint64_t seed,
    bool keep_supplementary, int baseq_min, py::dict blacklist)
{
    const int nb = (int)edges.size() - 1;
    Acc gg_l(nb), hh_l(nb), gh_l(nb), gg_n(nb), hh_n(nb), gh_n(nb);
    // MATCHED: the TRUE partner, but only for the pairs that were null-eligible.
    // The null can only use a pair that HAS another end within +-window, which
    // at single-cell coverage is a minority of pairs and is biased toward
    // read-dense loci.  Comparing that against the all-pairs linked curve
    // therefore compares two different sets of loci, not two partner choices.
    // This third arm restores the paired contrast: same pairs, same anchor end,
    // cross-end only -- the ONLY difference from the null is whether the
    // partner is the true mate.
    Acc gg_m(nb), hh_m(nb), gh_m(nb);
    std::mt19937_64 rng(seed);
    // Separate stream, so adding this arm leaves the linked and null draws --
    // and hence any previously computed result -- bit-identical.
    std::mt19937_64 rng_m(seed ^ 0x9E3779B97F4A7C15ULL);
    uint64_t n_pairs_total = 0, n_null_eligible = 0;

    // ---- GIL HELD: pull every Python-owned buffer out up front -------------
    // Nothing below may touch the interpreter, so the whole parse can run with
    // the GIL released.  That lets callers use a THREAD pool and share this
    // cell's (large) context tables instead of pickling them to each process.
    struct ChromCtx {
        std::string chrom;
        const int64_t *gp = nullptr, *hp = nullptr;
        const uint8_t *gr = nullptr, *hr = nullptr;
        int64_t ng = 0, nh = 0;
        std::vector<int64_t> bl_start, bl_end;
    };
    std::vector<ChromCtx> ctxs;
    std::vector<py::object> keepalive;      // hold refs so .data() stays valid

    std::vector<std::string> chroms;
    for (auto item : ctx_gch) chroms.push_back(py::str(item.first));
    for (auto item : ctx_hcg) {
        std::string c = py::str(item.first);
        if (std::find(chroms.begin(), chroms.end(), c) == chroms.end())
            chroms.push_back(c);
    }
    std::sort(chroms.begin(), chroms.end());

    for (const std::string& chrom : chroms) {
        ChromCtx cc; cc.chrom = chrom;
        if (ctx_gch.contains(chrom.c_str())) {
            auto t = ctx_gch[chrom.c_str()].cast<py::tuple>();
            auto pa = t[0].cast<py::array_t<int64_t>>();
            auto ra = t[1].cast<py::array_t<uint8_t>>();
            cc.gp = pa.data(); cc.gr = ra.data(); cc.ng = pa.size();
            keepalive.push_back(pa); keepalive.push_back(ra);
        }
        if (ctx_hcg.contains(chrom.c_str())) {
            auto t = ctx_hcg[chrom.c_str()].cast<py::tuple>();
            auto pa = t[0].cast<py::array_t<int64_t>>();
            auto ra = t[1].cast<py::array_t<uint8_t>>();
            cc.hp = pa.data(); cc.hr = ra.data(); cc.nh = pa.size();
            keepalive.push_back(pa); keepalive.push_back(ra);
        }
        if (cc.ng == 0 && cc.nh == 0) continue;
        if (blacklist.contains(chrom.c_str())) {
            auto t = blacklist[chrom.c_str()].cast<py::tuple>();
            auto sa = t[0].cast<py::array_t<int64_t>>();
            auto ea = t[1].cast<py::array_t<int64_t>>();
            cc.bl_start.assign(sa.data(), sa.data() + sa.size());
            cc.bl_end.assign(ea.data(), ea.data() + ea.size());
        }
        ctxs.push_back(std::move(cc));
    }

    // ---- GIL RELEASED for all htslib + compute work ------------------------
    {
    py::gil_scoped_release release;

    samFile* fp = sam_open(bam_path.c_str(), "r");
    if (!fp) throw std::runtime_error("cannot open BAM: " + bam_path);
    bam_hdr_t* hdr = sam_hdr_read(fp);
    if (!hdr) { sam_close(fp); throw std::runtime_error("cannot read header"); }
    hts_idx_t* idx = sam_index_load(fp, bam_path.c_str());
    if (!idx) {
        bam_hdr_destroy(hdr); sam_close(fp);
        throw std::runtime_error("cannot load BAM index for " + bam_path);
    }
    const uint16_t skip_flags = keep_supplementary ? SKIP_FLAGS_BASE : SKIP_FLAGS;

    for (const ChromCtx& cc : ctxs) {
        const std::string& chrom = cc.chrom;
        const int64_t *gp = cc.gp, *hp = cc.hp;
        const uint8_t *gr = cc.gr, *hr = cc.hr;
        const int64_t ng = cc.ng, nh = cc.nh;
        const std::vector<int64_t>& bl_start = cc.bl_start;
        const std::vector<int64_t>& bl_end = cc.bl_end;

        int tid = bam_name2id(hdr, chrom.c_str());
        if (tid < 0) continue;
        hts_itr_t* iter = sam_itr_queryi(idx, tid, 0, hdr->target_len[tid]);
        if (!iter) continue;

        std::unordered_map<std::string, bam1_t*> buf;
        std::vector<PairRec> pairs;
        bam1_t* b = bam_init1();

        while (sam_itr_next(fp, iter, b) >= 0) {
            if (b->core.flag & skip_flags) continue;
            if (b->core.qual < mapq_min) continue;
            if (b->core.mtid != b->core.tid) continue;      // cis only

            std::string qname = bam_get_qname(b);
            auto it = buf.find(qname);
            if (it == buf.end()) {
                bam1_t* keep = bam_dup1(b);
                buf.emplace(std::move(qname), keep);
                continue;
            }
            bam1_t* mate = it->second;
            buf.erase(it);
            if (mate->core.qual < mapq_min) { bam_destroy1(mate); continue; }

            const bam1_t* r1 = (b->core.pos <= mate->core.pos) ? b : mate;
            const bam1_t* r2 = (r1 == b) ? mate : b;

            // Drop the PAIR if EITHER end touches a blacklist interval, matching
            // bam_parser's read-level blacklist filter.
            if (!bl_start.empty() &&
                (hits_blacklist(bl_start, bl_end, r1->core.pos, bam_endpos(r1)) ||
                 hits_blacklist(bl_start, bl_end, r2->core.pos, bam_endpos(r2)))) {
                bam_destroy1(mate);
                continue;
            }

            PairRec rec;
            rec.mid1 = (r1->core.pos + bam_endpos(r1)) / 2;
            rec.mid2 = (r2->core.pos + bam_endpos(r2)) / 2;
            int s1 = infer_conversion_strand(r1), s2 = infer_conversion_strand(r2);
            if (ng) {
                site_calls(gp, gr, ng, r1, trim_ends, s1, baseq_min, rec.g1);
                site_calls(gp, gr, ng, r2, trim_ends, s2, baseq_min, rec.g2);
            }
            if (nh) {
                site_calls(hp, hr, nh, r1, trim_ends, s1, baseq_min, rec.h1);
                site_calls(hp, hr, nh, r2, trim_ends, s2, baseq_min, rec.h2);
            }
            bam_destroy1(mate);
            if (rec.g1.size() || rec.g2.size() || rec.h1.size() || rec.h2.size()) {
                pairs.push_back(std::move(rec));
                ++n_pairs_total;
            }
        }
        bam_destroy1(b);
        for (auto& kv : buf) bam_destroy1(kv.second);
        hts_itr_destroy(iter);

        // ---- LINKED ----
        for (const PairRec& p : pairs)
            add_pair(gg_l, hh_l, gh_l, p.g1, p.g2, p.h1, p.h2, edges,
                     max_site_pairs_per_read, rng, false);

        // ---- NULL: partner replaced by a different end within ±window ----
        if (null_window > 0 && pairs.size() > 1) {
            std::vector<size_t> order(pairs.size());
            for (size_t i = 0; i < order.size(); ++i) order[i] = i;
            std::sort(order.begin(), order.end(),
                      [&](size_t a, size_t c) { return pairs[a].mid2 < pairs[c].mid2; });
            std::vector<int64_t> mids(order.size());
            for (size_t i = 0; i < order.size(); ++i) mids[i] = pairs[order[i]].mid2;

            for (size_t i = 0; i < pairs.size(); ++i) {
                int64_t m = pairs[i].mid2;
                size_t lo = (size_t)(std::lower_bound(mids.begin(), mids.end(),
                                                      m - null_window) - mids.begin());
                size_t hi = (size_t)(std::upper_bound(mids.begin(), mids.end(),
                                                      m + null_window) - mids.begin());
                if (hi - lo < 2) continue;          // only itself in the window
                // Deterministic non-self pick: draw once, and if it lands on
                // this pair step to the next slot in the window.  The previous
                // "4 random attempts, else give up" could fail with p=1/16 when
                // the window holds only 2 ends, silently dropping an ELIGIBLE
                // pair from the null and making the eligible count depend on
                // the RNG stream (it differed between the two backends).
                const size_t span = hi - lo;
                size_t k = lo + (size_t)(rng() % span);
                if (order[k] == i) k = lo + ((k - lo) + 1) % span;
                const size_t pick = order[k];
                ++n_null_eligible;
                add_pair(gg_n, hh_n, gh_n, pairs[i].g1, pairs[pick].g2,
                         pairs[i].h1, pairs[pick].h2, edges,
                         max_site_pairs_per_read, rng, true);
                // Same pair, TRUE partner, cross-end only: the like-for-like
                // baseline for the null line directly above.
                add_pair(gg_m, hh_m, gh_m, pairs[i].g1, pairs[i].g2,
                         pairs[i].h1, pairs[i].h2, edges,
                         max_site_pairs_per_read, rng_m, true);
            }
        }
    }

    hts_idx_destroy(idx);
    bam_hdr_destroy(hdr);
    sam_close(fp);
    }   // GIL reacquired here

    py::dict out;
    out["gg_linked"] = to_numpy(gg_l);
    out["hh_linked"] = to_numpy(hh_l);
    out["gh_linked"] = to_numpy(gh_l);
    out["gg_null"] = to_numpy(gg_n);
    out["hh_null"] = to_numpy(hh_n);
    out["gh_null"] = to_numpy(gh_n);
    out["gg_matched"] = to_numpy(gg_m);
    out["hh_matched"] = to_numpy(hh_m);
    out["gh_matched"] = to_numpy(gh_m);
    out["n_pairs"] = py::int_(n_pairs_total);
    out["n_null_eligible"] = py::int_(n_null_eligible);
    return out;
}

PYBIND11_MODULE(_site_dist_core, m) {
    m.doc() = "htslib backend for site-level distance correlation";
    m.def("parse_cell_site_distance", &parse_cell_site_distance,
          py::arg("bam_path"), py::arg("ctx_gch"), py::arg("ctx_hcg"),
          py::arg("edges"), py::arg("trim_ends") = 5, py::arg("mapq_min") = 30,
          py::arg("null_window") = 100,
          py::arg("max_site_pairs_per_read") = 64, py::arg("seed") = 42,
          py::arg("keep_supplementary") = false, py::arg("baseq_min") = 0,
          py::arg("blacklist") = py::dict(),
          "Parse one cell's BAM and return per-distance-bin Pearson "
          "sufficient statistics for GCH-GCH / HCG-HCG / HCG-GCH, linked and "
          "non-linked.");
}
