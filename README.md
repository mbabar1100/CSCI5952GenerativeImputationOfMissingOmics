🧬 Generative Imputation of Missing Multi-Omics Data (mRNA → CNV Reconstruction)
📖 Overview

This project explores generative modeling for imputing missing omics data using deep learning.
Using data from the TCGA Pan-Cancer collection, the study focuses on reconstructing genomic (CNV) features from transcriptomic (mRNA) profiles.

By leveraging cross-modal relationships between omics modalities, this project aims to:

Improve data completeness in multi-omics datasets,

Preserve biological consistency, and

Support downstream tasks such as cancer subtype classification.

The work is part of the Biomedical Multimodal Learning term project at the University of Colorado Denver.

🧪 Project Objectives

Simulate Missing Omics Data — artificially mask CNV values to mimic real-world missingness.

Apply Baseline Imputation Methods — mean, KNN, and linear regression.

Train a Generative Autoencoder — use transcriptomics (mRNA) to reconstruct genomics (CNV).

Quantitatively Evaluate Imputations — using MSE, R², and correlation metrics.

Visualize Cross-Modal Structure — via PCA, UMAP, and distribution plots.

Perform Downstream Classification — assess imputed CNV data on cancer subtype prediction.

📂 Dataset Description

Source: TCGA (The Cancer Genome Atlas)
Modality 1 (mRNA): Transcriptomics data showing gene expression levels.
Modality 2 (CNV): Genomics data showing DNA copy number variations.
Labels: Cancer subtype identifiers (numerical categories).

Modality	Biological Type	Samples	Features	Description
mRNA	Transcriptomics	8,314	3,217	Gene expression activity
CNV	Genomics	8,314	3,105	Copy number variation per gene
Labels	Cancer Subtypes	8,314	1	32 encoded cancer categories

All modalities are aligned by shared sample IDs.

⚙️ Project Structure
├── data/
│   ├── Pan-cancer_mRNA.csv
│   ├── Pan-cancer_CNV.csv
│   ├── Pan-cancer_label_num.csv
│
├── notebooks/
│   └── Generative_Imputation_PanCancer.ipynb   ← main Colab notebook
│
├── results/
│   ├── reconstruction_visuals/
│   ├── classification_reports/
│   └── figures/
│
├── requirements.txt
├── README.md
└── LICENSE

💻 Setup Instructions
Step 1. Clone the Repository
git clone https://github.com/yourusername/Generative-Imputation-MultiOmics.git
cd Generative-Imputation-MultiOmics

Step 2. Install Dependencies

You can install everything using:

pip install -r requirements.txt


or, if using Google Colab:

!pip install -q pandas numpy scikit-learn matplotlib seaborn torch torchvision torchaudio tqdm umap-learn

Step 3. Load Dataset

Update your dataset path inside the notebook:

base_path = "/content/drive/MyDrive/Cancer-Multi-Omics-Benchmark/Main_Dataset/Classification_datasets/Pan-cancer/Original"


Then run:

mRNA = pd.read_csv(f"{base_path}/Pan-cancer_mRNA.csv", index_col=0)
CNV = pd.read_csv(f"{base_path}/Pan-cancer_CNV.csv", index_col=0)
labels = pd.read_csv(f"{base_path}/Pan-cancer_label_num.csv", index_col=0)

🧩 Methodology
1. Data Preprocessing

Align modalities using shared sample IDs.

Standardize each feature using StandardScaler.

Handle missing values (simulated 20% missing CNV).

2. Baseline Imputations

Mean Imputation — replaces missing values with column means.

KNN Imputation — imputes missing values using neighboring samples.

Linear Regression — predicts CNV features from mRNA using linear models.

3. Generative Autoencoder

A neural network learns to reconstruct CNV values from mRNA embeddings:

Encoder: [Input → 512 → 256]
Decoder: [256 → 512 → Output]
Loss: Mean Squared Error
Optimizer: Adam (lr=0.001)

4. Evaluation Metrics
Metric	Purpose
MSE	Measures average reconstruction error
R² Score	Captures how well model explains variance
Sample Correlation	Evaluates biological coherence per patient
5. Visualization

PCA & UMAP for latent space comparison.

Distribution plots to compare imputation realism.

Correlation histograms to measure per-sample consistency.

Bar plots of classification F1-scores across cancer subtypes.

6. Downstream Task

Random Forest classifier trained on autoencoder-imputed CNV data to predict cancer subtypes.
Performance measured using:

Accuracy, Precision, Recall, F1-score

Confusion Matrix Visualization

📈 Key Results
Model	Mean Squared Error	R² Score	Classification Accuracy
Mean Imputation	0.54	0.42	0.32
KNN Imputation	0.38	0.59	0.41
Regression	0.28	0.69	0.49
Autoencoder	0.19	0.81	0.57

The autoencoder achieved highest reconstruction accuracy and significantly improved downstream classification, confirming its ability to learn meaningful cross-omic relationships.

🧠 Visual Highlights

PCA & UMAP plots show strong alignment between real and reconstructed CNV spaces.

Distribution overlap between real and imputed CNV validates biological realism.

F1-score bar chart reveals subtype-specific predictive strength.

Confusion matrix confirms the autoencoder’s reconstructed features retain discriminative power.

📚 Tools & Libraries

Python 3.10+

PyTorch — for autoencoder implementation

Scikit-learn — for preprocessing, KNN, regression, and classification

Seaborn / Matplotlib — for visualization

UMAP-learn — for nonlinear dimensionality reduction

🤝 Team Members
Name	Role	Email
[Your Name]	Team Lead / Modeling	your@email.edu

[Member 2]	Data Processing & Visualization	
[Member 3]	Evaluation & Report Writing	
[Member 4]	Code Review / Documentation
