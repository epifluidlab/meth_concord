# Reproducing Figures 5 and 6

Two Jupyter notebooks rebuild Figure 5, Figure 6 and Supplementary Figures 9–12 of the scNOMe-HiC manuscript from the per-cell BAM files. Each one ends by checking key numbers against the manuscript.

| Notebook | Cells | Reference | Reproduces |
|---|---|---|---|
| [`figure5_cortex.ipynb`](figure5_cortex.ipynb) | 1,577 mouse cortex nuclei | mm10 | Fig. 5a–h; Supplementary Figs. 9, 10, 11, 12 |
| [`figure6_xci.ipynb`](figure6_xci.ipynb) | 469 GM12878 and 160 IMR-90 nuclei | GRCh38 | Fig. 6a–i |

The commands in the notebooks are those of the published runs. Given those runs' outputs, the figure and check steps reproduce the manuscript's figures and every checked number exactly.

Files in this folder:

| File | What it is |
|---|---|
| `figure5_cortex.ipynb`, `figure6_xci.ipynb` | the analyses, one command per run |
| `download_public_data.sh` | downloads and prepares the public data (genome, context sites, annotations; Supplementary Table 7) |
| `samples/mouse_cortex.samples.tsv` | the 1,577 cortex cells: BAM path, cell type, experiment and library batch, colour |
| `samples/gm12878_imr90.samples.tsv` | the 629 human cells: BAM path, cell line, batch |
| `repro.py` | the notebooks' helpers: run or skip a command, show a PDF, compare with the manuscript |

## Workflow

```
raw reads (GEO GSE346089)
  │  sc_NOMeHiC_pipeline: demultiplex, trim, bisulfite Hi-C alignment, duplicates, MD tags
  ▼
per-cell BAM files ─────────────────────────────── $SCNOMEHIC_DATA/bams/<dataset>/<cell>.bam
  │  this repository's pipeline (python -m nome_hic_concordance, steps 1–4):
  │  each read pair's two ends, their GCH / HCG sites, and each end's posterior
  │  of being open (GCH) or methylated (HCG)
  ▼
end posteriors ──────────────────────────────────── results/<dataset>/pipeline/context_<ctx>/04_posteriors
  │  scripts/*.py: concordance by distance, by chromatin class, per cell,
  │  per chromosome copy (Fig. 6)
  ▼
tables ──► figures (scripts/figure*.py) ──► checks against the manuscript
```

Public annotations come from `download_public_data.sh`. A few more are fetched by the scripts themselves on first use (see [Public data](#public-data)).

## 1. Install

The pipeline compiles a small C++ extension against htslib. Install the command-line tools and a compiler with conda, then install the repository:

```bash
conda create -n scnomehic -c conda-forge -c bioconda python=3.9 \
    htslib samtools bedtools seqkit ucsc-fatotwobit poppler pybind11 cxx-compiler
conda activate scnomehic
git clone <this repository> meth_concord
cd meth_concord
pip install -e ".[bigwig]"
pip install jupyterlab nbclient
```

Check that the extension was built. The published runs used it; without it the pipeline falls back to a slower pure-Python BAM parser:

```bash
python -c "import nome_hic_concordance._bam_parser_core, nome_hic_concordance._site_dist_core; print('ok')"
```

`poppler` (`pdftoppm`) is optional; it only draws the figures inline in the notebooks. The test suite runs with `pip install -e ".[dev]"` and `pytest -n auto`.

Tested with Python 3.9.18, numpy 1.24.4, pandas 2.1.0, scipy 1.11.2, matplotlib 3.8.0, pysam 0.23.3, pyarrow 21.0.0, scikit-learn 1.6.1, statsmodels 0.14.0, pyBigWig 0.3.25, samtools and htslib 1.21, and bedtools 2.30.0.

## 2. Data

Everything lives under one folder; point the notebooks at it with `SCNOMEHIC_DATA`:

```
$SCNOMEHIC_DATA/
├── bams/
│   ├── mouse_cortex/<cell>.bam, <cell>.bam.bai      1,577 cells
│   └── gm12878_imr90/<cell>.bam, <cell>.bam.bai     629 cells
├── annotations/                                     download_public_data.sh
│   ├── mm10/  hg38/  cache/
└── results/                                         written by the notebooks
    ├── mouse_cortex/
    └── gm12878_imr90/
```

### From raw reads to per-cell BAM files

The raw reads are in GEO under GSE346089 (SRA `<SRA accession>`). They are processed with sc_NOMeHiC_pipeline v1.0.0 (`<sc_NOMeHiC_pipeline URL>`), as described in the Methods:

1. demultiplex into cells (`demultiplex_ecker_scWGBS.pl`, bisulfiteHiC);
2. trim with Trim Galore! 0.6.7: `--paired_end --clip_R1 15 --clip_R2 15 --three_prime_clip_R1 5 --three_prime_clip_R2 5`;
3. align with `bhmem` (bisulfiteHiC 0.38) to the C>T and G>A converted genome (Bismark `genome_preparation`, BWA 0.7.18 index): `-pbat -nonDirectional -outputMateDiffChr`. The human reference is the GRCh38 no-alt analysis set (GCA_000001405.15) and the mouse reference is mm10;
4. `samtools sort -n`, `fixmate -m`, `sort`, `markdup`, then `samtools calmd` (samtools 1.21).

The notebooks need, per cell, a BAM file that is coordinate-sorted and indexed, with duplicates flagged and MD tags against the reference. The MD tags matter because each read's bisulfite strand is read from them. Name each file after the cell, as in the sample sheet's `cell_name` column, under `bams/mouse_cortex/` or `bams/gm12878_imr90/`.

### Sample sheets

`samples/*.samples.tsv` list the cells the figures use, with their BAM paths relative to `$SCNOMEHIC_DATA`. The notebooks write a copy with absolute paths into `results/<dataset>/`.

| Column | `mouse_cortex` | `gm12878_imr90` |
|---|---|---|
| `cell_name` | cell barcode | cell barcode |
| `bam_path` | `bams/mouse_cortex/<cell>.bam` | `bams/gm12878_imr90/<cell>.bam` |
| `group` | cell type (IT-L2/3 … PC/EC) | GM12878 or IMR90 |
| batches | `exp_batch`, `lib_batch` | `batch` |
| `color` | the cell type's colour in the figures | — |

### Public data

```bash
bash reproduce/download_public_data.sh "$SCNOMEHIC_DATA/annotations" all   # or mm10 / hg38
```

The script is safe to re-run: it keeps files already present and checks md5 sums where the source publishes them. It writes:

| Genome | Files | Source |
|---|---|---|
| mm10 | `mm10.2bit`, `mm10.chrom.sizes`, `mm10.fa` | UCSC |
| mm10 | `GCH_motif_seqkit.mm10.bed.gz`, `HCG_motif_seqkit.mm10.bed.gz`: every GCH and HCG cytosine, both strands, bgzipped and tabix-indexed (GCG is in neither) | `seqkit locate` on `mm10.fa` |
| mm10 | `mm10-blacklist.v2.bed` | ENCODE Blacklist v2 (Amemiya et al. 2019) |
| mm10 | `PMD_coordinates_mm10.bed.gz` | common PMDs/HMDs (Zhou et al. 2018) |
| mm10 | `repliseq/*.bw`: early and late Repli-seq of neural precursor cells, three tracks each | 4DN (4DNFINCFK1F9, 4DNFIC5KD21S, 4DNFI2D6RJL8, 4DNFIBA95HHW, 4DNFIDSG5BWJ, 4DNFIDJYEKHB) |
| hg38 | `GCA_000001405.15_GRCh38_no_alt_analysis_set.{fa,fa.fai,2bit}` | NCBI |
| hg38 | `GCH_motif_seqkit.hg38.bed.gz`, `HCG_motif_seqkit.hg38.bed.gz` | as for mm10 |
| hg38 | `UCSC.cpgIsland.hg38.sort.bed` | UCSC `cpgIslandExt` |
| hg38 | `gm12878_{promoter,enhancer,ctcf}.bed`: PLS; pELS and dELS; CA-CTCF | ENCODE v4 cCREs of GM12878 (ENCFF068GHQ) |
| hg38 | `xci_annotations/ncbiRefSeqCurated.txt.gz` | UCSC (NCBI RefSeq curated) |
| hg38 | `xci_annotations/gm12878_compartments.hg38.bedGraph` | 4DN GM12878 compartments (4DNFILYQ1PAY) |
| hg38 | `xci_annotations/gm12878_tad_boundaries{,_ctcf,_noctcf}.hg38.bed`: all TAD boundaries, those with a CTCF summit within 5 kb, and those without | 4DN GM12878 boundaries (4DNFIVK5JOFU); ENCODE GM12878 CTCF peaks (ENCFF796WRU) |
| hg38 | `rt_4dn/`, `lad_4dn/`: Repli-seq (log2 early/late) with constitutive and variable timing classes; LaminB1 DamID with cLAD, fLAD and ciLAD | 4DN (`scripts/fetch_4dn_repliseq.py`, `scripts/fetch_4dn_damid.py`) |

The scripts fetch the rest on first use:

- `annotation_class_concordance.py` fetches the full-stack chromHMM model (Vu & Ernst), the GENCODE genes (vM25 for mm10, v44 for hg38), the UCSC CpG islands and, for hg38, the common PMDs. They are cached in `annotations/cache/`, where `figure5_cortex.py` also reads the GENCODE genes and the chromHMM segmentation for Fig. 5h.
- `phased_snps.py` fetches the GM12878 (NA12878) SNVs from the 1000 Genomes high-coverage phased panel. It downloads about 32 GB, chromosome by chromosome, and deletes each file once it is used.
- `phased_snps.py` also fetches the IMR-90 SNVs from its haplotype-resolved assemblies (SyRI calls against T2T-CHM13, on figshare) and UCSC's `hs1ToHg38` chain.

## 3. Run the notebooks

```bash
export SCNOMEHIC_DATA=/path/to/scnomehic_data
cd reproduce
jupyter lab            # open figure5_cortex.ipynb or figure6_xci.ipynb and run all cells
```

or without a browser:

```bash
cd reproduce
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    figure5_cortex.ipynb --output figure5_cortex.run.ipynb
```

The configuration cell at the top of each notebook sets:

- `DATA`, which defaults to `$SCNOMEHIC_DATA`;
- `THREADS`, which defaults to 10;
- `DRY_RUN`: set it to `True` to print every command and the inputs still missing, without running anything.

In the Figure 5 notebook it also sets `CHROMHMM_BIVALENT` (see [Notes](#notes)).

Each command is skipped when its outputs already exist. A notebook can therefore be re-run, or resumed after an interruption, without redoing finished work. To redo a step, delete its output folder. Every command's output is also written to `results/<dataset>/logs/`. The figure commands always run; they take seconds.

The mouse notebook needs only the mm10 data and the human notebook only the hg38 data. The two can run in either order.

## 4. What each step makes

### `figure5_cortex.ipynb` (results in `results/mouse_cortex/`)

| Step | Script | Output | Used in |
|---|---|---|---|
| 1 | `python -m nome_hic_concordance --steps 1,2,3,4` (GCH, HCG) | `pipeline/context_{GCH,HCG}/` | everything |
| 2 | `read_pair_stats.py` | `figures/FigureS10_{all,gt1kb,gt20kb}.pdf` | Suppl. Fig. 10 |
| 3 | `end_posterior_distance_correlation.py`, under three non-linked controls | `distance/control1_same_cell_near_copies_refused/` | Fig. 5c,d; Suppl. Figs. 11, 12a,b |
| | | `distance/control2_same_cell_near_copies_kept/` | Fig. 5a,b; Suppl. Fig. 11 |
| | | `distance/control3_same_place_other_cells/` | Suppl. Fig. 11 |
| 4 | `annotation_class_concordance.py` (A/B, replication timing, PMD/HMD, chromHMM) | `classes/near_copies_refused/` | Fig. 5e–h |
| | | `classes/near_copies_kept/` | Suppl. Fig. 12c,d |
| 5 | `figure5_cortex.py`, `figure_s_null_controls.py`, `figure_s9_concordance_model.py` | `figures/Figure5.pdf`, `FigureS9.pdf`, `FigureS11.pdf`, `FigureS12.pdf` | |
| | | `figures/Figure5h_variable_bins.tsv`: every bin of Fig. 5h (276, ranked as drawn) with its genes | Fig. 5h |
| 6 | checks | 19 numbers against the manuscript | |

### `figure6_xci.ipynb` (results in `results/gm12878_imr90/`)

| Step | Script | Output | Used in |
|---|---|---|---|
| 1 | `phased_snps.py 1kgp`, `phased_snps.py syri` | `snps/NA12878.het.vcf.gz`, `snps/IMR90.het.vcf.gz` | everything below |
| 2 | `python -m nome_hic_concordance`, in two passes (see Notes) | `pipeline_autosomes/` (the fit), `pipeline/` (the posteriors) | Steps 4, 7 |
| 3 | `allelic_imbalance.py`, both lines | `haplotypes/` (each read pair's copy), `allelic_phased/allelic_cells.tsv` (each IMR-90 cell's Xi) | Steps 4, 5, 7 |
| | `allelic_imbalance.py`, GM12878 elements | `allelic_elements/allelic_summary.tsv`, `allelic_snps.tsv` | Fig. 6d; Steps 5, 6 |
| 4 | `annotation_class_concordance.py --chrx --haplotypes` | `chrx_classes/chrx_*.tsv` | Fig. 6b,h; Steps 5, 6 |
| 5 | `haplotype_contacts.py` | `haplotype_contacts/` | Fig. 6b–g |
| 6 | `copy_mixing_excess.py` | `copy_mixing_excess/` | Fig. 6h |
| 7 | `end_posterior_distance_correlation.py --haplotypes` | `units/end_posterior_units_GM12878.tsv` | Fig. 6i |
| 8 | `figure6_xci.py` | `figures/Figure6.pdf` | |
| 9 | checks | 16 numbers against the manuscript | |

## 5. Checks

The last cell of each notebook prints a table with four columns:

- `value`: what is checked;
- `manuscript`: the number as printed in the manuscript;
- `this run`: this run's value, rounded to the same number of decimals;
- `matches`.

Every sampling step uses a fixed seed (each script's default), so a rerun should match on every row. The sampling steps are the mixture fit's subsample and restarts, the non-linked draws, and the bootstraps and permutations.

## Notes

- **Memory.** Control 3 of Figure 5 draws each contact's stand-in from other cells. It does so one stratum (cell type × batch) at a time, and holds at most `--pool-max-cells` cells' ends at once. At the default of 300 cells it needs roughly 9–12 GB; lower it if memory is short.
- **chromHMM grouping (Figure 5).** `CHROMHMM_BIVALENT = "polycomb"` keeps the bivalent-promoter states inside the Polycomb group, giving the six groups of Fig. 5f and Supplementary Fig. 12d. `"own"` gives them a group of their own; the class runs then go to `classes_bivalent_own/`.
- **Fig. 5h.** The bins are those whose ΔC × w (each contact's w from its A/B class pair, as in Fig. 5e) differs between the ten cell types beyond sampling, in the class run's `locus_variability.tsv.gz`. Their annotations are tested as Fig. 5g's hot bins are, against genome rotations. h takes its chromHMM rows from the full-stack states themselves, to split promoters into TSS, flanking, poised and bivalent states; there, unlike in Fig. 5f, bivalent promoters count as promoters rather than polycomb.
- **chrX in the human posteriors.** The pipeline leaves chrX and chrY out by default, so the human posteriors are made in two passes. Pass 1 fits the beta-binomial mixture on the autosomes (`pipeline_autosomes/`). Pass 2 reruns steps 2 and 4 on every chromosome with `--exclude-chrom none`, using that fit, so X is scored under the autosomes' model (`pipeline/`). The mouse analyses use the autosomes only.
- **Which X is inactive.** In GM12878 the inactive X is hap1, the paternal copy: chrX is phased by transmission from NA12878's parents, NA12891 and NA12892. In IMR-90 each cell's inactive X is called from its own CpG-island methylation (`allelic_cells.tsv`).
- **dbSNP.** The pipeline's `--dbsnp-vcf` filter acts only when context sites are extracted from the `.2bit` reference. With `--context-bed`, as here and in the published runs, context sites are not filtered against dbSNP. The notebooks leave the option out.
- **Runs without the notebooks.** Every command is printed before it runs, and is written at the top of its log in `results/<dataset>/logs/`. You can run the steps from a shell or a cluster scheduler and come back to the notebook for the figures and checks. Finished steps are then skipped.

## Citation

If you use this code, please cite the scNOMe-HiC manuscript (`<citation>`).
