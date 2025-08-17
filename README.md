Assignment 1 – Part D 

End-to-end training/evaluation for NeuMF, LightGCN, and SSR (SASRec-style) on Goodbooks-10k using implicit feedback (ratings ≥ 3).
Evaluation: Leave-One-Out (LOO) with 99 negatives + 1 positive per user. Metrics: HR@10, NDCG@10 (plus summary CSVs and plots).

Run on Google Colab

Open the notebook:

assignment_1d/notebooks/PartD_colab.ipynb (upload to Colab or open from GitHub).

Set GPU:

Runtime → Change runtime type → Hardware accelerator = GPU (T4/L4 both work).

Provide the dataset:

Ensure /content/data/ratings.csv exists. You can either upload or mount Drive.

Option A — Upload ratings.csv directly

from google.colab import files
import os, shutil

os.makedirs("/content/data", exist_ok=True)

uploaded = files.upload()  # choose ratings.csv from your computer

shutil.move(next(iter(uploaded.keys())), "/content/data/ratings.csv")

print("Saved to /content/data/ratings.csv")



Option B — Mount Google Drive

from google.colab import drive

drive.mount('/content/drive')  # follow the prompt

Run the notebook cells in order:

Setup (imports, AMP/TF32, seeds)

Load data (prints users/items/positives)

Train & Evaluate (runs NeuMF → SSR → LightGCN)

Figures (generates F1–F16)

Outputs (saved automatically)

CSV metrics → /content/assignment_1d/outputs/

metrics_neumf.csv, metrics_lightgcn.csv, metrics_ssr.csv

summary_all_models.csv, segment_evaluation.json ← (renamed)

Figures → /content/assignment_1d/figures/

F1_...png … F16_...png

Notes

Expected results (+- small variance):

- NeuMF: HR@10 ≈ 0.798, NDCG@10 ≈ 0.566

- LightGCN: HR@10 ≈ 0.687, NDCG@10 ≈ 0.428

- SSR: HR@10 ≈ 0.082, NDCG@10 ≈ 0.037 (low because timestamps are not available)

- LightGCN takes longer per epoch than NeuMF/SSR.
