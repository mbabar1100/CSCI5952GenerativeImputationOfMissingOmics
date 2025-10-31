<h1 align="center">🧬 Generative Imputation of Missing Multi-Omics Data (mRNA → CNV Reconstruction)</h1>

<p align="center">
<b>Biomedical Multimodal Learning — University of Colorado Denver</b>  
</p>

---

## 📖 Overview
This project explores **generative modeling for imputing missing omics data** using deep learning.  
Using data from the **TCGA Pan-Cancer** collection, the study focuses on **reconstructing genomic (CNV) features** from **transcriptomic (mRNA)** profiles.

By leveraging cross-modal relationships between omics modalities, this project aims to:
- Improve **data completeness** in multi-omics datasets  
- Preserve **biological consistency**  
- Support **downstream tasks** such as cancer subtype classification  

---

## 🧪 Project Objectives
1. **Simulate Missing Omics Data** — artificially mask CNV values to mimic real-world missingness  
2. **Apply Baseline Imputation Methods** — mean, KNN, and linear regression  
3. **Train a Generative Autoencoder** — use transcriptomics (mRNA) to reconstruct genomics (CNV)  
4. **Quantitatively Evaluate Imputations** — using MSE, R², and correlation metrics  
5. **Visualize Cross-Modal Structure** — via PCA, UMAP, and distribution plots  
6. **Perform Downstream Classification** — assess imputed CNV data on cancer subtype prediction  

---

## 📂 Dataset Description
**Source:** TCGA (The Cancer Genome Atlas)  

**Modalities Used:**
- **mRNA (Transcriptomics):** Gene expression activity  
- **CNV (Genomics):** DNA copy number variation  
- **Labels:** Cancer subtype identifiers (32 encoded categories)  

| Modality | Biological Type | Samples | Features | Description |
|-----------|-----------------|----------|-----------|--------------|
| mRNA | Transcriptomics | 8,314 | 3,217 | Gene expression activity |
| CNV | Genomics | 8,314 | 3,105 | Copy number variation per gene |
| Labels | Cancer Subtypes | 8,314 | 1 | 32 encoded cancer categories |

All modalities are aligned by shared sample IDs.

---

## ⚙️ Project Structure
├── data/
│ ├── Pan-cancer_mRNA.csv
│ ├── Pan-cancer_CNV.csv
│ ├── Pan-cancer_label_num.csv
│
├── notebooks/
│ └── Generative_Imputation_PanCancer.ipynb ← main Colab notebook
│
├── results/
│ ├── reconstruction_visuals/
│ ├── classification_reports/
│ └── figures/
│
├── requirements.txt
├── README.md
└── LICENSE

yaml
Copy code

---

## 💻 Setup Instructions

### Step 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Generative-Imputation-MultiOmics.git
cd Generative-Imputation-MultiOmics
Step 2. Install Dependencies
bash
Copy code
pip install -r requirements.txt
or if using Google Colab:

python
Copy code
!pip install -q pandas numpy scikit-learn matplotlib seaborn torch torchvision torchaudio tqdm umap-learn
Step 3. Load Dataset
Update your dataset path inside the notebook:

python
Copy code
base_path = "/content/drive/MyDrive/Cancer-Multi-Omics-Benchmark/Main_Dataset/Classification_datasets/Pan-cancer/Original"
Then run:

python
Copy code
mRNA = pd.read_csv(f"{base_path}/Pan-cancer_mRNA.csv", index_col=0)
CNV = pd.read_csv(f"{base_path}/Pan-cancer_CNV.csv", index_col=0)
labels = pd.read_csv(f"{base_path}/Pan-cancer_label_num.csv", index_col=0)
🧩 Methodology
1. Data Preprocessing
Align modalities using shared sample IDs

Standardize features with StandardScaler

Simulate 20% missing values in CNV to model incomplete data

2. Baseline Imputations
Mean Imputation: Replace missing values with mean

KNN Imputation: Use neighboring samples for replacement

Linear Regression: Predict CNV values using mRNA as input

3. Generative Autoencoder
A neural network learns to reconstruct CNV values from mRNA embeddings.

vbnet
Copy code
Encoder: [Input → 512 → 256]
Decoder: [256 → 512 → Output]
Loss: Mean Squared Error
Optimizer: Adam (lr=0.001)
4. Evaluation Metrics
Metric	Description
MSE	Average reconstruction error
R² Score	Explained variance measure
Sample Correlation	Biological coherence per patient

5. Visualization
PCA & UMAP: Compare latent spaces

KDE & Histogram: Compare data distributions

Heatmaps: Show missingness and correlation

Bar plots: Show classification F1-scores

6. Downstream Classification
Random Forest trained on autoencoder-imputed CNV data

Evaluated on real cancer subtype labels

Metrics: Precision, Recall, F1-score, Accuracy, and Confusion Matrix

📈 Key Results
Method	MSE ↓	R² ↑	Accuracy ↑
Mean Imputation	0.54	0.42	0.32
KNN Imputation	0.38	0.59	0.41
Linear Regression	0.28	0.69	0.49
Autoencoder (Proposed)	0.19	0.81	0.57

The Autoencoder achieved the best overall performance, showing that generative modeling effectively captures the complex cross-omic relationships between transcriptomics and genomics.

🧠 Visual Highlights
PCA and UMAP show overlapping manifolds for real and imputed CNV data.

KDE distributions confirm the reconstructed CNV retains biological variance.

Correlation heatmaps demonstrate consistency across patient samples.

Confusion matrix highlights moderate accuracy in cancer subtype classification.

📚 Tools & Libraries
Category	Tools
Deep Learning	PyTorch
ML & Preprocessing	Scikit-learn
Visualization	Matplotlib, Seaborn
Dimensionality Reduction	UMAP-learn, PCA
Environment	Python 3.10+, Google Colab

🤝 Team Members
Name
Muhammad Babar
Kathryn Eron
Kavya Avula
Susmitha Rachuri
