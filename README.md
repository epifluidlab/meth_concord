# meth_concord

Long-range methylation / accessibility concordance analysis from NOMe-HiC data.

This repository implements the **NOMe-HiC Long-Range Chromatin Accessibility
Concordance Pipeline** described in [`design/NOME_HIC_CONCORDANCE_PIPELINE_DESIGN.md`](design/NOME_HIC_CONCORDANCE_PIPELINE_DESIGN.md).

## Install

```
pip install -e .
```

The pipeline depends on `numpy`, `scipy`, `pandas`, `pyarrow`, `pysam`,
`py2bit`, `scikit-learn`, `umap-learn`, `matplotlib`, `seaborn`, `joblib`, and
`statsmodels` (see `requirements.txt`).

## Run

```
nome_hic_concordance \
    --samples samples.tsv \
    --reference hg38.2bit \
    --output-dir output/ \
    --n-threads 8
```

`samples.tsv` is a tab-delimited manifest with columns
`cell_name`, `bam_path`, and an optional `group`. See
[`design/NOME_HIC_CONCORDANCE_PIPELINE_DESIGN.md`](design/NOME_HIC_CONCORDANCE_PIPELINE_DESIGN.md)
for the complete I/O specification, parameter reference, and module-by-module
algorithm descriptions.

The pipeline runs in seven steps (1 = context extraction, 2 = BAM parsing,
3 = mixture-model fit, 4 = posteriors, 5 = concordance + distance decay,
6 = shrinkage, 7 = matrix processing + UMAP). Pass `--steps 4,5,6,7` to
resume from intermediate outputs.

## Tests

```
pytest tests/
```
