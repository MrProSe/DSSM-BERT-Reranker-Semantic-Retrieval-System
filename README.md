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

