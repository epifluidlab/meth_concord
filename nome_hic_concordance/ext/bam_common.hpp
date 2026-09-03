/* Shared htslib read-level helpers.
 *
 * These four functions exist verbatim inside both bam_parser_core.cpp and
 * site_dist_core.cpp.  joint_state_core.cpp includes them from here instead of
 * making a third copy, and tests/test_joint_state_core.py pins the conversion
 * -strand call against _site_dist_core on the same reads so the copies that do
 * remain cannot silently diverge.  Folding the two older files onto this
 * header is a mechanical follow-up; it is deliberately not done in the same
 * change that introduces the new extractor, because both of them feed
 * published outputs.
 */
#pragma once

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include <htslib/sam.h>

namespace nhc {

static constexpr uint16_t FLAG_UNMAPPED      = 0x4;
static constexpr uint16_t FLAG_MATE_UNMAPPED = 0x8;
static constexpr uint16_t FLAG_PROPER_PAIR   = 0x2;
static constexpr uint16_t FLAG_SECONDARY     = 0x100;
static constexpr uint16_t FLAG_DUPLICATE     = 0x400;
static constexpr uint16_t FLAG_SUPPLEMENTARY = 0x800;
static constexpr uint16_t SKIP_FLAGS_BASE =
    FLAG_UNMAPPED | FLAG_MATE_UNMAPPED | FLAG_SECONDARY | FLAG_DUPLICATE;
static constexpr uint16_t SKIP_FLAGS = SKIP_FLAGS_BASE | FLAG_SUPPLEMENTARY;

static constexpr int STRAND_UNKNOWN = 0;
static constexpr int STRAND_CT      = 1;   // C->T conversion strand
static constexpr int STRAND_GA      = 2;   // G->A conversion strand

/* 4-bit BAM sequence encoding -> character. */
static inline char bam_base_at(const uint8_t* seq, int i) {
    static const char TABLE[] = "=ACMGRSVTWYHKDBN";
    return TABLE[bam_seqi(seq, i)];
}

/* Query offset of the first aligned base (leading soft/hard clips). */
static inline int query_alignment_start(const uint32_t* cigar, int n_cigar) {
    int q = 0;
    for (int i = 0; i < n_cigar; ++i) {
        int op = bam_cigar_op(cigar[i]);
        if (op == BAM_CSOFT_CLIP || op == BAM_CHARD_CLIP) {
            q += (op == BAM_CSOFT_CLIP) ? (int)bam_cigar_oplen(cigar[i]) : 0;
            continue;
        }
        break;
    }
    return q;
}

/* Bisulfite conversion strand from the MD tag's mismatch spectrum.
 *
 * ``min_margin`` is how far C->T must beat G->A (or vice versa) before the
 * call is made; below it the read is STRAND_UNKNOWN and contributes nothing,
 * which is the safe direction -- a misassigned strand reads the reference base
 * at every site of the other strand and calls it methylated. */
static inline int infer_conversion_strand(const bam1_t* b, int min_margin = 1) {
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
        } else if (*p == '^') {                 // deletion: consumes ref only
            ++p;
            while (*p && isalpha((unsigned char)*p)) ++p;
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
    if (min_margin < 1) min_margin = 1;
    if (ct - ga >= min_margin) return STRAND_CT;
    if (ga - ct >= min_margin) return STRAND_GA;
    return STRAND_UNKNOWN;
}

}  // namespace nhc
