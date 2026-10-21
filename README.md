# Cross-Modal Recovery of Missing Genomic Signals: Benchmarking Imputation Strategies Under Structured Missingness in Pan-Cancer Multi-Omics Data

**Corresponding author:** Muhammad Babar ([muhammad.babar@ucdenver.edu](mailto:muhammad.babar@ucdenver.edu))

Developed as part of **CSCI 5952 — Big Data Science** (team: Muhammad Babar, Kathryn Eron, Kavya Avula, Susmitha Rachuri).

---

## Contents

1. [Description](#description)
2. [Dataset Information](#dataset-information)
3. [Code Information](#code-information)
4. [Requirements](#requirements)
5. [Usage Instructions](#usage-instructions)
6. [Methodology](#methodology)
7. [Results Files](#results-files)
8. [Citation](#citation)
9. [License & Contribution Guidelines](#license--contribution-guidelines)

---

## Description

This repository contains the analysis code and results for a benchmark of imputation strategies that recover partially masked **copy-number variation (CNV)** measurements from **transcriptomic (mRNA)** input in TCGA pan-cancer data.

Two families of methods are compared:

| Family | Methods |
|---|---|
| **Within-modality** | Mean imputation, K-nearest neighbours (KNN) |
| **Cross-modal** (mRNA → CNV) | Linear regression, ridge regression, nonlinear neural encoder–decoder |

The study evaluates:

1. **Structured missingness benchmark** — random, sample-targeted, and feature-targeted CNV masking at 5%, 10%, and 20%, each repeated across five random seeds.
2. **Leakage-controlled confirmatory evaluation** — 60/15/25 train/validation/test split with training-only preprocessing.
3. **Downstream cancer-type classification** using Random Forest.
4. **Gene-level reconstruction** at cancer-associated loci and across all CNV features.
5. **Functional enrichment** — pre-ranked GSEA against GO Biological Process 2023 and KEGG 2021 Human.
6. **Prognostic validation** — TCGA-CDR overall survival with cancer-stratified Cox models.
7. **Architecture ablation** — six neural configurations, five seeds.
8. **Leave-one-cancer-type-out (LOCO)** — cross-cancer generalisation across 31 held-out cancer types.

---

## Dataset Information

### Source

All molecular and clinical data are derived from **The Cancer Genome Atlas (TCGA)**. Two third-party datasets are used:

| Dataset | Used for | Source | Licence |
|---|---|---|---|
| **MLOmics** (Pan-cancer classification dataset, *Original* scale) | mRNA, CNV and cancer-type labels | Figshare: [doi:10.6084/m9.figshare.28729127](https://doi.org/10.6084/m9.figshare.28729127) · Hugging Face: <https://huggingface.co/datasets/AIBIC/MLOmics> (`Main_Dataset/Classification_datasets/Pan-cancer/Original/`) · Paper: Yang et al., *Scientific Data* 2025, [doi:10.1038/s41597-025-05235-x](https://doi.org/10.1038/s41597-025-05235-x) | CC BY 4.0 |
| **TCGA Pan-Cancer Clinical Data Resource (TCGA-CDR)** | Overall survival (OS time, OS event) | <https://gdc.cancer.gov/about-data/publications/PanCan-Clinical-2018> · Paper: Liu et al., *Cell* 2018, [doi:10.1016/j.cell.2018.02.052](https://doi.org/10.1016/j.cell.2018.02.052) | Public (GDC open access) |

### Processed cohort

| Component | Layer | Samples | Features | Role |
|---|---|---|---|---|
| mRNA | Transcriptomics | 8,314 | 3,217 | Cross-modal predictor |
| CNV | Genomics | 8,314 | 3,105 | Reconstruction target |
| Cancer label | Phenotype | 8,314 | 1 | Downstream multiclass evaluation |
| OS time / OS event | Clinical | 8,204 (usable) | 2 | Survival analysis |

**Cohort notes**

- 32 cancer types are represented (see Table 2 of the manuscript).
- GBM is not present in the aligned cohort.
- OV contains 9 samples and is excluded **only** as a LOCO held-out target.
- Cancer-type labels were verified against TCGA-CDR patient barcodes (100% mapping purity).

### Input files

| File | Contents | In this repo? |
|---|---|---|
| `Pan-cancer_mRNA.csv` | mRNA expression matrix (8,314 × 3,217) | No — download from MLOmics (see above) |
| `Pan-cancer_CNV.csv` | CNV matrix (8,314 × 3,105) | No — download from MLOmics (see above) |
| `Pan-cancer_label_num.csv` | Cancer-type labels (32 classes) | No — download from MLOmics (see above) |
| `TCGA-CDR-SupplementalTableS1.xlsx` | TCGA-CDR clinical outcomes (OS time, OS event) | Yes |

The three MLOmics files are used exactly as distributed; all preprocessing (label alignment, standardisation, masking) is done inside the notebooks.

---

## Code Information

### Repository layout

```text
.
├── README.md
├── LICENSE                                            # MIT licence
├── requirements.txt                                   # Python dependencies
│
├── CSI5952_GenerativeImputationOfMissingOmics.ipynb   # course-project notebook: EDA, masking, baselines, autoencoder, classification
├── Babar analysis-paper-1.ipynb                       # manuscript analysis notebook (confirmatory, gene-level, GSEA, survival, ablation, LOCO)
├── Babar analysis-paper-1.html                        # rendered HTML export of the manuscript notebook
├── GitHub Repo URL.txt
├── TCGA-CDR-SupplementalTableS1.xlsx                  # clinical outcomes input
│
├── *.csv, *.png, *.pdf, *.svg                         # result tables and figures (see Results Files)
│
├── reproduce_analysis.py                              # supplementary BioNeuralNet graph/module analysis
├── build_figures.py                                   # figures for the supplementary analysis
├── metrics.json, feature_results.npz,
│   pure_module_tests.csv, hybrid_module_tests.csv     # supplementary analysis outputs
├── analysis/                                          # copy of the supplementary scripts and outputs
├── figures/                                           # figure1_workflow, figure2_umap, figure3_metrics (PDF)
└── manuscript.tex
```

### Notebooks

**`CSI5952_GenerativeImputationOfMissingOmics.ipynb`** — the original course-project pipeline:

| Step | Content |
|---|---|
| 1–2 | Environment setup and data loading |
| 3–5 | Label cleaning and alignment, exploratory analysis, preprocessing |
| 6 | Simulating missing omics data (CNV masking) |
| 7 | Baseline imputations (mean, KNN, regression) |
| 8 | Autoencoder for generative cross-modal imputation |
| 9 | Evaluation (MSE, R², Pearson), performance plots, best/worst cases |
| 10 | Downstream Random Forest classification |

**`Babar analysis-paper-1.ipynb`** — the extended manuscript analyses:

| Section | Content | Main outputs |
|---|---|---|
| Cross-modal model | Leakage-controlled 60/15/25 split, KNN vs neural model on identical masked entries | Confirmatory metrics |
| Gene-level reconstruction | Per-gene Pearson r and MSE (incl. ERBB2, EGFR, MYC, CDKN2A, PTEN) | `gene_level_cnv_validation*.svg/png` |
| Functional enrichment | Pre-ranked GSEA (GO BP 2023, KEGG 2021 Human) | `functional_enrichment_gene_ranking.csv`, `GO_*`, `KEGG_*` |
| Survival / prognostic | Cancer-stratified Cox, KM, log-rank | `survival_cindex_by_cancer.csv`, `KM_*.png` |
| Ablation | Six architectures × five seeds | `ablation_raw_5seeds.csv`, `ablation_summary_5seeds.csv` |
| LOCO | Leave-one-cancer-type-out generalisation | `LOCO_cross_cancer_generalization.*` |

### Supplementary analysis

`reproduce_analysis.py` and `build_figures.py` run a separate graph-based module analysis (similarity graphs, GNN embeddings, Leiden and k-means clustering, UMAP) using the **BioNeuralNet** library. They expect a local clone of BioNeuralNet in a `BioNeuralNet/` folder next to the scripts.

---

## Requirements

- Python 3.10+ with Jupyter
- Python packages listed in [`requirements.txt`](requirements.txt): `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `torch`, `tqdm`, `umap-learn`, `lifelines`, `gseapy`, `openpyxl`
- Supplementary analysis only: `networkx` and [BioNeuralNet](https://github.com/UCD-BDLab/BioNeuralNet)
- A GPU is optional; the neural models fall back to CPU automatically.
- Enrichment analysis (`gseapy`) downloads GO / KEGG gene-set libraries, so it needs an internet connection.

---

## Usage Instructions

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare the data

1. Download `Pan-cancer_mRNA.csv`, `Pan-cancer_CNV.csv` and `Pan-cancer_label_num.csv` from MLOmics (`Main_Dataset/Classification_datasets/Pan-cancer/Original/` on [Hugging Face](https://huggingface.co/datasets/AIBIC/MLOmics), or the [Figshare archive](https://doi.org/10.6084/m9.figshare.28729127)).
2. Place them in one folder together with `TCGA-CDR-SupplementalTableS1.xlsx` from this repository.

### 3. Set the data path

The notebooks point to a local folder. Before running, change the `BASE` variable (and any hard-coded `C:\Users\...` paths) to your data folder:

```python
BASE = r"path/to/your/data"
```

### 4. Run the notebooks

```bash
jupyter notebook
```

1. Run `CSI5952_GenerativeImputationOfMissingOmics.ipynb` top to bottom for the baseline benchmark and classification.
2. Run `Babar analysis-paper-1.ipynb` top to bottom for the confirmatory, gene-level, enrichment, survival, ablation and LOCO analyses. Outputs are written to the `BASE` folder (and a `mandatory_analysis/` subfolder).

### 5. (Optional) Supplementary analysis

```bash
git clone https://github.com/UCD-BDLab/BioNeuralNet.git
python reproduce_analysis.py
python build_figures.py
```

---

## Methodology

Key protocol settings:

| Analysis | Settings |
|---|---|
| **Masking** | Patterns: random, sample-targeted, feature-targeted · Ratios: 0.05, 0.10, 0.20 · Seeds: 42, 123, 456, 789, 1011 |
| **Confirmatory split** | 60/15/25 train/validation/test, mutually exclusive, training-only standardisation |
| **Confirmatory result** (Table 4) | KNN — MSE 0.320, R² 0.689, r 0.836 · Neural — MSE 0.533, R² 0.481, r 0.694 |
| **Downstream classification** | Random Forest trained on reconstructed training CNV, evaluated on reconstructed test CNV |
| **Enrichment** | GSEApy pre-ranked, 1,000 permutations, gene-set size 10–500, seed 42 |
| **Survival** | 50-PC PCA on original training CNV · cancer-stratified Cox, L2 penalizer 0.1 · training-derived within-cancer median risk thresholds · held-out KM / log-rank |
| **Ablation** | Six architectures × five seeds · paired t-tests with Holm correction |
| **LOCO** | One run per held-out cancer type · 80/20 development split (split seed 42) · training-only standardisation |

**Leakage control:** all scaling and model fitting use training data only, and masks are fixed before imputation so every method is evaluated on identical held-out entries.

---

## Results Files

| File(s) | Analysis |
|---|---|
| `gene_level_cnv_validation.svg`, `gene_level_cnv_validation_corrected.svg`, `gene_level_cnv_validation_corrected_preview.png` | Gene-level reconstruction |
| `functional_enrichment_gene_ranking.csv`, `GO_KEGG_gene_reconstruction_ranking.csv` | Gene ranking for enrichment |
| `GO_BP_ranked_enrichment.csv`, `KEGG_ranked_enrichment.csv`, `GO_BP_final_neural_r_GSEA.csv`, `KEGG_final_neural_r_GSEA.csv` | GO / KEGG enrichment |
| `survival_cindex_by_cancer.csv`, `KM_original_CNV.png`, `KM_neural_reconstructed_CNV.png` | Survival / prognostic validation |
| `ablation_raw_5seeds.csv`, `ablation_summary_5seeds.csv` | Architecture ablation |
| `LOCO_cross_cancer_generalization.csv` / `.png` / `.pdf` | LOCO generalisation |
| `figures/`, `Fig 3-Pure Leiden Colors.png`, `metrics.json`, `*_module_tests.csv` | Supplementary BioNeuralNet analysis |

---

## Citation

If you use this code, please cite the accompanying manuscript:

- Babar M, et al. Cross-Modal Recovery of Missing Genomic Signals: Benchmarking Imputation Strategies Under Structured Missingness in Pan-Cancer Multi-Omics Data. Submitted to *PeerJ Computer Science*.

and the datasets it builds on:

- Yang Z, Kotoge R, Piao X, Chen Z, Zhu L, Gao P, Matsubara Y, Sakurai Y, Sun J. MLOmics: Cancer Multi-Omics Database for Machine Learning. *Scientific Data* 12, 913 (2025). doi:10.1038/s41597-025-05235-x
- Liu J, et al. An Integrated TCGA Pan-Cancer Clinical Data Resource to Drive High-Quality Survival Outcome Analytics. *Cell* 173(2), 400–416 (2018). doi:10.1016/j.cell.2018.02.052

---

## License & Contribution Guidelines

### License

The code in this repository is released under the **MIT License** — see [`LICENSE`](LICENSE).

Third-party data keep their original terms: MLOmics is distributed under **CC BY 4.0**, and TCGA-CDR is open-access data from the NCI Genomic Data Commons. Please cite both sources (see [Citation](#citation)) when reusing them.

### Contributing

Contributions, bug reports and questions are welcome.

1. **Issues:** open a [GitHub issue](https://github.com/mbabar1100/CSCI5952GenerativeImputationOfMissingOmics/issues) describing the problem, the steps to reproduce it, and your environment (OS, Python and package versions).
2. **Pull requests:** fork the repository, create a branch from `main`, make focused changes with clear commit messages, and open a pull request explaining what changed and why.
3. **Reproducibility:** changes to analysis code should keep the fixed random seeds and the training-only preprocessing described in [Methodology](#methodology), so results stay comparable with the manuscript.

For questions about the study, contact the corresponding author, Muhammad Babar ([muhammad.babar@ucdenver.edu](mailto:muhammad.babar@ucdenver.edu)).
