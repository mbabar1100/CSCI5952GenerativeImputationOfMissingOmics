# Cross-Modal Recovery of Missing Genomic Signals: Benchmarking Imputation Strategies Under Structured Missingness in Pan-Cancer Multi-Omics Data

## Title
**Cross-Modal Recovery of Missing Genomic Signals: Benchmarking Imputation Strategies Under Structured Missingness in Pan-Cancer Multi-Omics Data**

Corresponding author: Muhammad Babar (muhammad.babar@ucdenver.edu)

## Description
This repository contains the analysis code and processed data supporting a benchmark of **within-modality** (mean, KNN) and **cross-modal** (linear regression, ridge regression, and a nonlinear mRNA→CNV neural encoder–decoder) imputation strategies for recovering partially masked copy-number variation (CNV) measurements from transcriptomic (mRNA) input in TCGA pan-cancer data. The study evaluates:

1. **Structured missingness benchmark** — random, sample-targeted, and feature-targeted CNV masking at 5%, 10%, and 20%, each repeated across five random seeds.
2. **Leakage-controlled confirmatory evaluation** — 60/15/25 train/validation/test split with training-only preprocessing.
3. **Downstream cancer-type classification** using Random Forest.
4. **Gene-level reconstruction** at cancer-associated loci and across all CNV features.
5. **Functional enrichment** (pre-ranked GSEA, GO Biological Process 2023 and KEGG 2021 Human).
6. **Prognostic validation** via TCGA-CDR overall survival, cancer-stratified Cox models.
7. **Architecture ablation** across six neural configurations, five seeds.
8. **Leave-one-cancer-type-out (LOCO)** cross-cancer generalization across 31 held-out cancer types.

## Dataset Information

### Source
All molecular and clinical data are derived from **The Cancer Genome Atlas (TCGA)** Pan-Cancer Atlas and the **TCGA Pan-Cancer Clinical Data Resource (TCGA-CDR)**.

- Molecular data portal: https://portal.gdc.cancer.gov/
- Clinical outcomes: TCGA-CDR (Liu et al., *Cell* 2018, doi:10.1016/j.cell.2018.02.052)

### Processed cohort used in this study
| Component | Layer | Samples | Features | Role |
|---|---|---|---|---|
| mRNA | Transcriptomics | 8,314 | 3,217 | Cross-modal predictor |
| CNV | Genomics | 8,314 | 3,105 | Reconstruction target |
| Cancer label | Phenotype | 8,314 | 1 | Downstream multiclass evaluation |
| OS time / OS event | Clinical | 8,204 (usable) | 2 | Survival analysis |

- 32 cancer types represented (see Table 2 of manuscript).
- GBM is not present in the aligned cohort; OV contains 9 samples and is excluded only as a LOCO held-out target.
- Cancer-type labels were verified against TCGA-CDR patient barcodes (100% mapping purity).

### Files provided
- `data/mrna_processed.csv` — standardized mRNA matrix (samples × 3,217 features) after training-only scaling.
- `data/cnv_processed.csv` — standardized CNV matrix (samples × 3,105 features).
- `data/cancer_labels.csv` — verified TCGA cancer codes.
- `data/survival_tcga_cdr.csv` — OS time, OS event, cancer code.
- `data/splits/` — train/validation/test indices per seed.
- `data/masks/` — saved binary CNV masks for each (pattern, ratio, seed) combination.

If you cannot redistribute the processed matrices, the `scripts/download_and_preprocess.py` script rebuilds them from the public GDC files (see Usage).

## Code Information

### Repository layout
.
├── README.md
├── LICENSE
├── requirements.txt
├── data/ # processed inputs (see Dataset Information)
├── masks/ # saved CNV masks (pattern, ratio, seed)
├── scripts/
│ ├── download_and_preprocess.py # pull TCGA files, align modalities, training-only scaling
│ ├── make_masks.py # generate random / sample / feature targeted masks
│ ├── run_benchmark.py # 9-scenario × 5-seed benchmark (mean, KNN, linear, ridge, neural)
│ ├── run_confirmatory.py # leakage-controlled 5% random held-out evaluation
│ ├── run_classification.py # Random Forest downstream evaluation
│ ├── run_gene_level.py # per-gene Pearson r and MSE
│ ├── run_enrichment.py # GSEApy pre-ranked GO / KEGG
│ ├── run_survival.py # PCA + cancer-stratified Cox, KM, log-rank
│ ├── run_ablation.py # 6 architectures × 5 seeds
│ └── run_loco.py # leave-one-cancer-type-out
├── src/
│ ├── models.py # NeuralEncoderDecoder, Ridge wrapper, KNN wrapper
│ ├── metrics.py # MSE, R², Pearson, within-cancer C-index
│ └── utils.py # seeding, splits, IO
└── results/
├── tables/ # CSV mirrors of manuscript tables
└── figures/ # generated figures

text

### What each major script does
- **`download_and_preprocess.py`** — downloads mRNA and CNV from GDC, reduces barcodes to TCGA patient IDs, matches cancer labels and TCGA-CDR, writes `mrna_processed.csv`, `cnv_processed.csv`.
- **`make_masks.py`** — writes a binary mask file per `(pattern ∈ {random, sample, feature}, ratio ∈ {0.05, 0.10, 0.20}, seed ∈ {42, 123, 456, 789, 1011})`. Masks are saved *before* imputation so every method is scored on identical entries.
- **`run_benchmark.py`** — reproduces the original nine-scenario exploratory benchmark (Table 3) across all five methods and five seeds.
- **`run_confirmatory.py`** — reproduces Table 4 (KNN MSE 0.320, R² 0.689, r 0.836; Neural MSE 0.533, R² 0.481, r 0.694) using the mutually exclusive 60/15/25 split and training-only standardization.
- **`run_classification.py`** — trains a Random Forest on reconstructed training CNV and evaluates on reconstructed test CNV (Table 6).
- **`run_gene_level.py`** — computes per-gene Pearson r and MSE on masked test entries (Table 8, Figure 3).
- **`run_enrichment.py`** — pre-ranked GSEA via GSEApy with 1,000 permutations, min size 10, max size 500, seed 42, GO BP 2023 and KEGG 2021 Human.
- **`run_survival.py`** — 50-PC PCA on original training CNV, cancer-stratified Cox with L2 penalizer 0.1, training-derived within-cancer median risk thresholds, held-out KM/log-rank (Table 11, Figure 4).
- **`run_ablation.py`** — retrains six architectures across five seeds with paired t-tests + Holm correction (Table 12).
- **`run_loco.py`** — one training run per held-out cancer type, 80/20 development split (split seed 42), training-only standardization (Table 13, Figure 5).

## Usage Instructions

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
