#!/usr/bin/env bash
# Download and prepare the public data the Figure 5 and Figure 6 notebooks use
# (listed in Supplementary Table 7).
#
#   bash reproduce/download_public_data.sh ANNOT_DIR [mm10|hg38|all]
#
# mm10 is what figure5_cortex.ipynb needs, hg38 what figure6_xci.ipynb needs.
# Files already present are kept, so the script can be re-run after an
# interruption.  Needs curl, seqkit, bgzip/tabix (htslib), samtools,
# bedtools, faToTwoBit (UCSC) and the repository installed with its bigwig
# extra (pip install -e ".[bigwig]"):
#
#   conda install -c bioconda -c conda-forge seqkit htslib samtools bedtools \
#       ucsc-fatotwobit
#
# The mouse full-stack ChromHMM model, GENCODE vM25 and the UCSC CpG islands
# are fetched by annotation_class_concordance.py itself (--genome mm10) into
# ANNOT_DIR/cache on its first run.  The 1000 Genomes panel and the IMR-90
# assembly variants are fetched by scripts/phased_snps.py in
# figure6_xci.ipynb (about 32 GB, removed as it is used).
set -euo pipefail

ANNOT=${1:?usage: download_public_data.sh ANNOT_DIR [mm10|hg38|all]}
WHICH=${2:-all}
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python3}
mkdir -p "$ANNOT"
ANNOT=$(cd "$ANNOT" && pwd)

UCSC=https://hgdownload.soe.ucsc.edu/goldenPath
FDN=https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput
ENC=https://www.encodeproject.org/files

md5of() { if command -v md5sum >/dev/null; then md5sum "$1" | cut -d' ' -f1;
          else md5 -q "$1"; fi; }

# fetch URL DEST [MD5]
fetch() {
    local url=$1 dest=$2 want=${3:-}
    if [ -s "$dest" ]; then
        echo "  have $(basename "$dest")"
    else
        echo "  get  $(basename "$dest")"
        curl -fL --retry 5 --retry-delay 10 -o "$dest.part" "$url"
        mv "$dest.part" "$dest"
    fi
    if [ -n "$want" ]; then
        local got; got=$(md5of "$dest")
        [ "$got" = "$want" ] || { echo "md5 mismatch for $dest: $got != $want" >&2; exit 1; }
    fi
}

# motifs FASTA OUTDIR TAG: GCH and HCG cytosines on both strands, as seqkit
# locate reports them, one BED per context (bgzipped, tabix-indexed).  The
# BED start is the cytosine's own 0-based position on either strand, so the
# pipeline reads them with its default --context-bed-offset 0; GCG matches
# neither pattern.
motifs() {
    local fa=$1 out=$2 tag=$3 ctx pat
    for ctx in GCH HCG; do
        [ "$ctx" = GCH ] && pat="GC[ATC]" || pat="[ATC]CG"
        local bed=$out/${ctx}_motif_seqkit.$tag.bed
        if [ -s "$bed.gz.tbi" ]; then echo "  have $(basename "$bed").gz"; continue; fi
        echo "  make $(basename "$bed").gz (seqkit locate -p '$pat')"
        seqkit locate -i -r -p "$pat" -j 8 "$fa" \
            | grep -v "^seqID" | cut -f1,4-7 \
            | perl -ne 'chomp;@f=split "\t";$s=$f[2];$e=$f[3]-1;print "$f[0]\t$s\t$e\t$f[4]\t0\t$f[1]\n";' \
            | grep -v "_" | grep -v "chrEBV" \
            | LC_ALL=C sort -k1,1 -k2,2n -k3,3n > "$bed"
        bgzip -f "$bed" && tabix -f -p bed "$bed.gz"
    done
}

# ------------------------------------------------------------------ mm10 --
mm10() {
    local d=$ANNOT/mm10
    mkdir -p "$d/repliseq"
    echo "== mm10 (Figure 5, Supplementary Figs. 9-12)"
    fetch "$UCSC/mm10/bigZips/mm10.2bit" "$d/mm10.2bit"
    fetch "$UCSC/mm10/bigZips/mm10.chrom.sizes" "$d/mm10.chrom.sizes"
    if [ ! -s "$d/mm10.fa" ]; then
        fetch "$UCSC/mm10/bigZips/mm10.fa.gz" "$d/mm10.fa.gz"
        gzip -dc "$d/mm10.fa.gz" > "$d/mm10.fa"
    fi
    motifs "$d/mm10.fa" "$d" mm10
    # ENCODE Blacklist v2 (Amemiya et al. 2019)
    if [ ! -s "$d/mm10-blacklist.v2.bed" ]; then
        fetch https://raw.githubusercontent.com/Boyle-Lab/Blacklist/master/lists/mm10-blacklist.v2.bed.gz \
              "$d/mm10-blacklist.v2.bed.gz"
        gzip -dc "$d/mm10-blacklist.v2.bed.gz" > "$d/mm10-blacklist.v2.bed"
    fi
    # common PMDs / HMDs (Zhou et al. 2018)
    fetch https://zhouserver.research.chop.edu/GenomeAnnotation/mm10/PMD_coordinates_mm10.bed.gz \
          "$d/PMD_coordinates_mm10.bed.gz"
    # 4DN 2-stage Repli-seq of neural precursor cells: three early and three
    # late tracks, as RPKM bedGraphs, converted to bigWig as the published
    # runs read them
    local rt=$d/repliseq
    while read -r acc name md5 url; do
        fetch "$url" "$rt/$name.$acc.bedGraph.gz" "$md5"
        if [ ! -s "$rt/$name.$acc.bw" ]; then
            "$PY" - "$rt/$name.$acc.bedGraph.gz" "$d/mm10.chrom.sizes" "$rt/$name.$acc.bw" <<'PYEOF'
import sys, pandas as pd, pyBigWig
src, sizes, out = sys.argv[1:]
cs = pd.read_csv(sizes, sep="\t", header=None, names=["c", "n"])
bg = pd.read_csv(src, sep="\t", header=None, comment="#", dtype={0: str},
                 names=["c", "s", "e", "v"])
bg = bg[bg.c.isin(set(cs.c))]
order = [c for c in cs.c if c in set(bg.c)]
bw = pyBigWig.open(out, "w")
bw.addHeader([(c, int(n)) for c, n in zip(cs.c, cs.n) if c in set(order)])
for c in order:
    g = bg[bg.c == c].sort_values("s")
    bw.addEntries([c] * len(g), g.s.astype(int).tolist(),
                  ends=g.e.astype(int).tolist(), values=g.v.astype(float).tolist())
bw.close()
PYEOF
            echo "  made $name.$acc.bw"
        fi
    done <<EOF
4DNFINCFK1F9 F121-9-CASTx129_NPC.repliseq.early.mm10 c5e68fb22ee966bf6ea2ef13185acffd $FDN/537ca79c-679c-443b-8bf1-6a2802bc0d30/4DNFINCFK1F9.bedGraph.gz
4DNFIBA95HHW F121-9-CASTx129_NPC.repliseq.late.mm10 b2bc35b4eb969c62d912073411e16e7c $FDN/75aaa7db-dbb4-4f50-bdba-65063117135f/4DNFIBA95HHW.bedGraph.gz
4DNFIC5KD21S Cas129-CD1-129_CAST.repliseq.early.rep1 037e7a0d5066846af9e312b7339bbbd2 $FDN/67be57ba-76e5-40e7-b259-cdbb1f118d27/4DNFIC5KD21S.bedGraph.gz
4DNFI2D6RJL8 Cas129-CD1-129_CAST.repliseq.early.rep2 7c4078810e16379d14068bae50fbf286 $FDN/c2c30807-cdc8-475f-8693-62065f540623/4DNFI2D6RJL8.bedGraph.gz
4DNFIDSG5BWJ Cas129-CD1-129_CAST.repliseq.late.rep1 3d77ef9bd6d50b24313fbd9885e01509 $FDN/862df51a-d201-489d-9fe7-6c1da1dbdd28/4DNFIDSG5BWJ.bedGraph.gz
4DNFIDJYEKHB Cas129-CD1-129_CAST.repliseq.late.rep2 8854e3b9971125ce9c959b8f55e4e230 $FDN/b17dbfdf-b863-4727-bdb2-f7289ce981be/4DNFIDJYEKHB.bedGraph.gz
EOF
}

# ------------------------------------------------------------------ hg38 --
hg38() {
    local d=$ANNOT/hg38 x
    x=$d/xci_annotations
    mkdir -p "$x"
    echo "== hg38 (Figure 6)"
    local base=GCA_000001405.15_GRCh38_no_alt_analysis_set
    if [ ! -s "$d/$base.fa" ]; then
        fetch "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/$base.fna.gz" \
              "$d/$base.fna.gz"
        gzip -dc "$d/$base.fna.gz" > "$d/$base.fa"
    fi
    [ -s "$d/$base.fa.fai" ] || samtools faidx "$d/$base.fa"
    [ -s "$d/$base.2bit" ] || faToTwoBit "$d/$base.fa" "$d/$base.2bit"
    motifs "$d/$base.fa" "$d" hg38
    # UCSC CpG islands
    if [ ! -s "$d/UCSC.cpgIsland.hg38.sort.bed" ]; then
        fetch "$UCSC/hg38/database/cpgIslandExt.txt.gz" "$d/cpgIslandExt.hg38.txt.gz"
        gzip -dc "$d/cpgIslandExt.hg38.txt.gz" | awk 'BEGIN{OFS="\t"}{print $2,$3,$4,$5}' \
            | grep -v "_" | LC_ALL=C sort -k1,1 -k2,2n > "$d/UCSC.cpgIsland.hg38.sort.bed"
    fi
    # ENCODE v4 cCREs of GM12878: promoters (PLS), enhancers (pELS, dELS),
    # CTCF sites (CA-CTCF)
    fetch "$ENC/ENCFF068GHQ/@@download/ENCFF068GHQ.bed.gz" "$d/ENCFF068GHQ.bed.gz"
    local t n c
    for t in "promoter:PLS" "enhancer:pELS,dELS" "ctcf:CA-CTCF"; do
        n=${t%%:*}; c=${t#*:}
        [ -s "$d/gm12878_$n.bed" ] && continue
        gzip -dc "$d/ENCFF068GHQ.bed.gz" \
            | awk -F'\t' -v c="$c" 'BEGIN{split(c,a,",");for(i in a)k[a[i]]=1} ($10 in k)' \
            | cut -f1-3 | LC_ALL=C sort -k1,1 -k2,2n > "$d/gm12878_$n.bed"
        echo "  made gm12878_$n.bed"
    done
    # NCBI RefSeq curated genes (UCSC)
    fetch "$UCSC/hg38/database/ncbiRefSeqCurated.txt.gz" "$x/ncbiRefSeqCurated.txt.gz"
    # 4DN GM12878 compartments (in situ Hi-C, MboI; 250 kb), as a bedGraph
    fetch "$FDN/7635b7f8-4bfd-4ebc-9cd2-1d64fe8c84ee/4DNFILYQ1PAY.bw" "$x/4DNFILYQ1PAY.bw" \
          ba73062524e460539390cd6513a068d1
    if [ ! -s "$x/gm12878_compartments.hg38.bedGraph" ]; then
        "$PY" - "$x/4DNFILYQ1PAY.bw" "$x/gm12878_compartments.hg38.bedGraph" <<'PYEOF'
import math, sys, pyBigWig
bw = pyBigWig.open(sys.argv[1])
with open(sys.argv[2], "w") as out:
    for c in bw.chroms():
        if c == "chrY":
            continue
        for s, e, v in (bw.intervals(c) or []):
            if not math.isnan(v):
                out.write(f"{c}\t{s}\t{e}\t{v:.5g}\n")
PYEOF
        echo "  made gm12878_compartments.hg38.bedGraph"
    fi
    # 4DN GM12878 TAD boundaries, and those with a GM12878 CTCF ChIP-seq
    # summit (ENCODE conservative IDR peaks) within the bin +/- 5 kb
    fetch "$FDN/15b1a7db-6d72-4134-a90f-fcfeb644b92f/4DNFIVK5JOFU.bed.gz" "$x/4DNFIVK5JOFU.bed.gz" \
          ebdf5462f5d150b6818c2bf2ab570ea7
    fetch "$ENC/ENCFF796WRU/@@download/ENCFF796WRU.bed.gz" "$x/ENCFF796WRU.bed.gz" \
          3f36655eb9a421a542ff031fc7b5e9aa
    if [ ! -s "$x/gm12878_tad_boundaries_ctcf.hg38.bed" ]; then
        gzip -dc "$x/4DNFIVK5JOFU.bed.gz" | grep -v "^#" | LC_ALL=C sort -k1,1 -k2,2n \
            > "$x/gm12878_tad_boundaries.hg38.bed"
        gzip -dc "$x/ENCFF796WRU.bed.gz" | awk 'BEGIN{OFS="\t"}{s=$2+$10; print $1,s,s+1}' \
            | LC_ALL=C sort -k1,1 -k2,2n > "$x/gm12878_ctcf_summits.hg38.bed"
        bedtools window -a "$x/gm12878_tad_boundaries.hg38.bed" -b "$x/gm12878_ctcf_summits.hg38.bed" \
            -w 5000 -u > "$x/gm12878_tad_boundaries_ctcf.hg38.bed"
        bedtools window -a "$x/gm12878_tad_boundaries.hg38.bed" -b "$x/gm12878_ctcf_summits.hg38.bed" \
            -w 5000 -v > "$x/gm12878_tad_boundaries_noctcf.hg38.bed"
        echo "  made gm12878_tad_boundaries{,_ctcf,_noctcf}.hg38.bed" \
             "($(wc -l < "$x/gm12878_tad_boundaries.hg38.bed"), $(wc -l < "$x/gm12878_tad_boundaries_ctcf.hg38.bed")," \
             "$(wc -l < "$x/gm12878_tad_boundaries_noctcf.hg38.bed"); expected 7553, 4510, 3043)"
    fi
    # 4DN human Repli-seq (log2 early/late, 250 kb) and LaminB1 DamID LADs,
    # read by the chrX class run
    [ -s "$d/rt_4dn/signal_track_flags.txt" ] || \
        "$PY" "$REPO/scripts/fetch_4dn_repliseq.py" --output-dir "$d/rt_4dn" \
            --derive-partition-bed --keep-fractions
    [ -s "$d/lad_4dn/cLAD.bed" ] || \
        "$PY" "$REPO/scripts/fetch_4dn_damid.py" --output-dir "$d/lad_4dn" --derive-lad-bed
}

case $WHICH in
    mm10) mm10 ;;
    hg38) hg38 ;;
    all) mm10; hg38 ;;
    *) echo "unknown genome: $WHICH (mm10, hg38 or all)" >&2; exit 2 ;;
esac
echo "done: $ANNOT"
