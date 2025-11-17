🧬 Generative Imputation of Missing Multi-Omics Data
Cross-Modal Recovery of CNV from mRNA Using Deep Learning

Biomedical Multimodal Learning — University of Colorado Denver

📖 Overview

This project focuses on generative imputation for multi-omics data, specifically reconstructing Copy Number Variation (CNV) profiles from mRNA gene expression. Missing omics data is a common issue in clinical and biomedical research, and recovering these signals accurately can improve downstream analyses including cancer subtype prediction.

Using TCGA Pan-Cancer data, we:

Simulate multiple missingness scenarios

Apply classical imputation methods

Train a deep autoencoder to reconstruct CNV

Evaluate imputation quality

Test downstream classification performance

The results show that deep learning–based imputation significantly outperforms traditional baselines.

🎯 Project Objectives

Simulate random, sample-targeted, and feature-targeted missingness (5%, 10%, 20%)

Implement baseline imputations: Mean, KNN, Linear Regression

Train a Generative Autoencoder for cross-modal prediction (mRNA → CNV)

Compare reconstruction accuracy using:

Mean Squared Error (MSE)

R² Score

Pearson Correlation

Evaluate impact on downstream Random Forest cancer classification

Analyze and visualize structural relationships between omics modalities

📂 Dataset Description

Source: TCGA Pan-Cancer (multi-omics benchmark)

Modality	Type	Samples	Features	Description
mRNA	Transcriptomics	8,314	3,217	Gene expression levels
CNV	Genomics	8,314	3,105	Copy number variation
Labels	Cancer subtype	8,314	1	32 encoded cancer types

All modalities are aligned using consistent TCGA sample IDs.

⚙️ Repository Structure
project/
│
├── data/
│   ├── Pan-cancer_mRNA.csv
│   ├── Pan-cancer_CNV.csv
│   └── Pan-cancer_label_num.csv
│
├── notebooks/
│   └── Generative_Imputation_PanCancer.ipynb
│
├── results/
│   ├── reconstruction_visuals/
│   ├── classification_reports/
│   └── figures/
│
├── baseline_imputations/
├── autoencoder_imputations/
├── missingness_versions/
│
├── requirements.txt
└── README.md

🚀 Setup Instructions
1. Clone Repository
git clone https://github.com/yourusername/Generative-Imputation-MultiOmics.git
cd Generative-Imputation-MultiOmics

2. Install Dependencies
pip install -r requirements.txt

3. (Colab / Jupyter) Install extra packages
!pip install pandas numpy scikit-learn matplotlib seaborn torch torchvision torchaudio umap-learn tqdm

4. Load Dataset
mRNA = pd.read_csv("data/Pan-cancer_mRNA.csv", index_col=0)
CNV = pd.read_csv("data/Pan-cancer_CNV.csv", index_col=0)
labels = pd.read_csv("data/Pan-cancer_label_num.csv")["Label"].values

🧪 Methodology
1. Preprocessing

Align samples across modalities

Scale features using StandardScaler

Create simulated missingness at 5%, 10%, 20%

2. Baseline Imputation Methods

Mean Imputation: column-wise mean filling

KNN Imputation: neighbor-based estimation

Linear Regression: CNV predicted from mRNA

3. Autoencoder-Based Generative Imputation

A neural architecture that learns cross-modal mappings:

mRNA → Encoder → Latent Space → Decoder → CNV reconstruction


Training details:

Layers: 512 → 128 → 512

Loss: MSE

Optimizer: Adam (lr = 0.001)

Epochs: 10

4. Evaluation Metrics
Metric	Meaning
MSE	Reconstruction error
R² Score	Explained variance
Correlation	Per-sample biological coherence
5. Downstream Cancer Classification

Algorithm: Random Forest

Inputs: Original & imputed CNV

Outputs: Accuracy, F1-macro

📈 Key Results
Imputation Quality (Across All Missingness Types)
Method	MSE ↓	R² ↑	Correlation ↑
Mean	~1.00	~0.00	~0.00
Regression	~1.00	~0.00	~0.02
KNN	~0.31	~0.68	~0.83
Autoencoder (Proposed)	~0.26	~0.73	~0.86

Autoencoder provided the best reconstruction on all metrics.

Downstream Classification Results (Random Forest)

Original CNV accuracy: 0.44

Method	Accuracy	F1-Macro
Mean	~0.43	~0.36
Regression	~0.44	~0.36
KNN	~0.45	~0.37
Autoencoder	0.46–0.48	0.38–0.39

🔍 Autoencoder-imputed CNV consistently outperforms all baselines.

📊 Visual Highlights

PCA projections show structural similarity between real and imputed CNV.

Missingness heatmaps illustrate how masking strategies impact recovery difficulty.

Correlation plots confirm stronger sample-level reconstruction from generative models.

Classification bar plots show improved downstream cancer prediction.

🧠 Assumptions on Noise

We assume missing values occur due to random measurement errors and not because of any specific cancer subtype.
Noise levels of 5%, 10%, and 20% simulate realistic signal loss in large genomics studies.

🤝 Team Members
Name
Muhammad Babar
Kathryn Eron
Kavya Avula
Susmitha Rachuri
