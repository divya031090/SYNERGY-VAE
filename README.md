# SYNERGY-VAE

# SYNERGY-VAE: Multimodal Variational Autoencoder for large population-level health dataset clustering and downstream prediction of depression risk

## 🧬 Overview

**SYNERGY-VAE** is a multimodal variational autoencoder framework designed to integrate diverse mulitmodal data domains (demographics, diet, examination, laboratory, and questionnaire data) into a shared latent representation. By combining high-dimensional health data, we reveal latent subgroups and explore links to depression risk and other health outcomes.

---

## 🌟 Features
```text
- Multi-domain normalization and preprocessing
- Shared latent space modeling using both single and multi-decoder VAEs
- Baseline clustering with PCA + KMeans
- Encoder weight-based feature importance and permutation feature importance (PFI)
- Cluster-specific predictive modeling (e.g., XGBoost, Random Forest)
- Downstream analyses: depression risk enrichment, SHAP explanations, visualizations

---

## 🗂️ Repository Structure
```text
scripts/
1_data_preprocessing.py
2_pca_kmeans_baseline.py
3_vae_singledecoder.py
4_joint_autoencoder_multidecoder.py
5_cluster_analysis.py
6_feature_importance_analysis.py
7_cluster_specific_models.py


Place all NHANES domain CSV files in data/raw/:
```text
Demo_numeric_NHANES_2005_2018.csv
Diet_numeric_NHANES_2005_2018.csv
Exam_numeric_NHANES_2005_2018.csv
Lab_numeric_NHANES_2005_2018.csv
Ques_numeric_NHANES_2005_2018.csv
Label_NHANES_2005_2018.csv

```text
python scripts/1_data_preprocessing.py
python scripts/2_pca_kmeans_baseline.py
python scripts/3_vae_singledecoder.py
python scripts/4_joint_autoencoder_multidecoder.py
python scripts/5_cluster_analysis.py
python scripts/6_feature_importance_analysis.py
python scripts/7_cluster_specific_models.py

**# Requirements**
```text
Python 3.8+
pandas
numpy
scikit-learn
tensorflow, keras
matplotlib
seaborn
xgboost
imbalanced-learn
tqdm
plotly
shap
Install via:

pip install -r requirements.txt

---

## 🚀 Quick Start

###  Clone & install

```bash
git clone https://github.com/divya031090/SYNERGY-VAE.git
cd SYNERGY-VAE
pip install -r requirements.txt



