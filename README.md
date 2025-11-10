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
4. **Build faiss index** – Building efficient similarity search indexes based on passage embeddings
5. **query_embbedings Generation** – Encode all queries to vector space.
6. **Train bert_reranker** - Read (query, passage, label) pairs and fine-tune the BERT ranking model
7. **Inference** – Combine FAISS with BERT Reranker to perform the complete retrieval and ranking process, and output the final results

---

## ⚙️ Usage
### 1️⃣ Train the model
```bash
python train_dssm.py \
  --train data/train.csv \
  --val data/val.csv \
  ----ckpt_dir ./ckpt_dssm_csv
```
### 2️⃣Encode passages
```bash
python encode_passages.py \
  --dssm_meta ./ckpt_dssm_csv/meta_final.npz
  --dssm_ckpt ./ckpt_dssm_csv/dssm_final.ckpt
```
### 3️⃣Build faiss index
```bash
python build_faiss_index.py \
  --passage_embs ./ckpt_dssm_csv/meta_final_passage_embs.npz
  --index_out ./faiss_index.ivf
```
### 4️⃣query embbeding generation
```bash
python encode_train_queries.py \
  --train_csv train.csv
  --dssm_meta ./ckpt_dssm_csv/meta_final.npz
  --dssm_ckpt ./ckpt_dssm_csv/dssm_final.ckpt

python encode_val_queries.py \
  --val_csv val.csv
  --dssm_meta ./ckpt_dssm_csv/meta_final.npz
  --dssm_ckpt ./ckpt_dssm_csv/dssm_final.ckpt
```
### 5️⃣Train BERT Reranker
```bash
python train_bert_reranker.py \
  --train_csv train.csv
  --valid_csv valid.csv
  --dssm_meta ./ckpt_dssm_csv/meta_final.npz
  --passage_embs ./ckpt_dssm_csv/meta_final_passage_embs.npz
  --faiss_index ./faiss_index.ivf
  --train_query_embs ./ckpt_dssm_csv/meta_final_train_query_embs.npz
  --val_query_embs ./ckpt_dssm_csv/meta_final_val_query_embs.npz
```
### 6️⃣ Inference and evaluate
```bash
python python pipeline_rerank.py \
  --dssm_meta ./ckpt_dssm_csv/meta_final.npz -
  -dssm_ckpt ./ckpt_dssm_csv/dssm_final.ckpt
  --passage_embs ./ckpt_dssm_csv/meta_final_passage_embs.npz
  --faiss_index ./faiss_index.ivf
  --bert_ckpt ./ckpt_bert/bert_rerank.h5
```
### 📈 Visualization
To inspect embeddings distribution:

PCA or t-SNE projection of query/passages

Helps diagnose domain gaps between train/val sets
