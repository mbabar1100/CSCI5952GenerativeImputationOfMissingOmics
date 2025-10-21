# CSCI5952 Generative Imputation Of Missing Omics
Generative imputation for missing multi-omics — reconstruct incomplete omics profiles with VAEs, GANs, and diffusion models and evaluate impact on prediction and subtyping.
Proejct Description
This project explores Generative Imputation of Missing Omics data using The Cancer Genome Atlas (TCGA) Pan-Cancer dataset. Multi-omics data, including genomics, transcriptomics, and clinical profiles, often suffer from missing modalities due to cost and experimental constraints. Our goal is to build and evaluate generative models (VAEs, GANs) to reconstruct missing omics profiles by leveraging existing modalities.

We aim to:

Build preprocessing pipelines for multi-omics integration.

Implement baseline and generative imputation models.

Evaluate reconstruction quality and its impact on downstream tasks like disease subtyping and survival prediction.

Project Steps
1. Environment Setup and Library Installation
Installed essential Python libraries: pandas, numpy, scikit-learn, torch, matplotlib, and seaborn.
Set up Google Colab environment with access to Google Drive for dataset storage.
Verified GPU availability for deep learning experiments.
2. Data Access and Download (TCGA)
Accessed RNA-Seq, Copy Number Variation, and Clinical data from the GDC Data Portal.
Organized downloaded data by modality (transcriptomics, genomics, clinical).
Ensured consistent naming and formatting of files for automated loading.
3. Data Preprocessing and Exploratory Analysis
Cleaned, normalized, and merged multi-omics data.
Handled missing values and inconsistent identifiers.
Performed batch-wise processing to handle large files efficiently.
Conducted initial exploratory analysis (missingness patterns, value distributions).
4. Baseline Imputation Models
Implemented simple yet effective imputation methods:

Mean Imputation – replaced missing values with feature-wise means.
KNN Imputation – used KNNImputer from scikit-learn with optimized neighbors.
Matrix Factorization (optional) – approximated data with low-rank factors.
5. Generative Model Development
Built neural-based imputation methods:

Autoencoder (AE) – trained to reconstruct missing data from compressed representations.
Variational Autoencoder (VAE) – captured probabilistic latent features for imputation.
(Optionally extendable to GAN-based models for synthetic feature generation.)
6. Evaluation Metrics and Visualization
Evaluated models using:

RMSE – Root Mean Squared Error between true and imputed values.
Pearson Correlation – measured biological consistency of imputed expression.
Visualizations:

Distribution of imputation errors.
PCA or t-SNE plots to assess structure preservation.
Comparison of modality-specific performance.
7. Reporting and Reproducibility
Documented all steps with clear markdown sections and experiment logs.
Recorded each team member’s contributions.
Summarized results into a final report with figures and performance tables.
Exported notebook as PDF and HTML, and pushed it to the GitHub branch for review.
