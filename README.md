# DSSM-BERT-Reranker-Semantic-Retrieval-System
This repository implements a full pipeline for semantic retrieval using Deep Structured Semantic Model (DSSM) for match and Bert for rerank.
---

## 📘 Overview
DSSM maps queries and passages into the same embedding space and retrieves top-k relevant passages based on cosine similarity.

---

## 🧩 Pipeline
1. **Data Preparation** – Prepare CSVs with query, passage, and label.
2. **Training** – Train DSSM model on (query, passage) pairs.
3. **Embedding Generation** – Encode all passages to vector space.
4. **Inference** – Given a query, retrieve top-K most similar passages.

---

## ⚙️ Usage
### 1️⃣ Train the model
```bash
python train_dssm.py \
  --train data/train.csv \
  --val data/val.csv \
  --save models/dssm_weights.h5
```
### 2️⃣Encode passages
```bash
python encode_passages.py \
  --dssm_meta meta.npz \
  --dssm_ckpt models/dssm_weights.h5
```
### 3️⃣ Run inference
```bash
python infer_dssm.py \
  --query "who makes jammy dodgers?" \
  --passage_embs passage_embs.npz
```
### 📊 Evaluation
You can evaluate Recall@K on validation queries using:
```
python eval_dssm.py --val data/val.csv
```
### 📈 Visualization
To inspect embeddings distribution:

PCA or t-SNE projection of query/passages

Helps diagnose domain gaps between train/val sets
