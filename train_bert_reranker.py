# train_bert_reranker.py
import argparse, numpy as np, faiss
import os

import pandas as pd
import tf_keras.callbacks
from transformers import BertTokenizerFast, TFBertForSequenceClassification,create_optimizer
import tensorflow as tf
from utils import load_meta, SimpleTokenizer, load_csv_pairs
bert_path = './models'
def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--train_csv', required=True)
    p.add_argument('--valid_csv', required=True)
    p.add_argument('--dssm_meta', required=True)
    p.add_argument('--passage_embs', required=True)
    p.add_argument('--faiss_index', required=True)
    p.add_argument('--train_query_embs', required=True)
    p.add_argument('--val_query_embs', required=True)
    p.add_argument('--bert_model', default='bert-base-uncased')
    p.add_argument('--topk', type=int, default=50)
    p.add_argument('--neg_per_pos', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--out_ckpt', default='./ckpt_bert/bert_rerank.h5')
    return p.parse_args()

def load_faiss(path):
    return faiss.read_index(path)

def build_pairs_from_topk(train_pairs, passages_list, query_texts, query_embs, faiss_idx, topk=50, neg_per_pos=3):

    examples=[]
    D,I = faiss_idx.search(query_embs.astype('float32'), topk)
    for qi, q in enumerate(query_texts):
        pos_ids = [pid for pid in set(train_pairs.get(q, [])) if pid < len(passages_list)]
        if not pos_ids:
            print(f"[WARN] Query '{q}' has no valid positive passages after filtering.")
            continue
        cand_ids = I[qi].tolist()
        negs = [cid for cid in cand_ids if cid not in pos_ids and cid < len(passages_list)]
        if not negs:
            print(f"[WARN] Query '{q}' has no valid negative passages after filtering.")
            continue
        for pos in pos_ids:
            if pos >= len(passages_list):
                print(
                    f"[ERROR] POSITIVE INDEX OUT OF RANGE: query='{q}', pos={pos}, passages_list_len={len(passages_list)}")
                continue  # 跳过越界的正例
            for n in negs[:neg_per_pos]:
                examples.append((q, passages_list[pos], 1))
                examples.append((q, passages_list[n], 0))
    return examples

def main():
    args = parse_args()
    train_pairs, val_pairs, _ = load_csv_pairs(args.train_csv, args.valid_csv)
    meta = np.load(args.dssm_meta, allow_pickle=True)
    passages_meta = meta['passages'].tolist()
    word2id = meta['word2id'].item()
    # load passage embeddings and train query emb
    pen = np.load(args.passage_embs, allow_pickle=True)
    passage_embs = pen['embeddings']
    passages_list = pen['passages'].tolist()
    train_qmeta = np.load(args.train_query_embs, allow_pickle=True)
    train_query_texts = train_qmeta['queries'].tolist()
    train_query_embs = train_qmeta['embeddings'].astype('float32')
    val_qmeta = np.load(args.val_query_embs, allow_pickle=True)
    val_query_texts = val_qmeta['queries'].tolist()
    val_query_embs = val_qmeta['embeddings'].astype('float32')

    faiss_idx = load_faiss(args.faiss_index)
    train_examples = build_pairs_from_topk(train_pairs, passages_list, train_query_texts, train_query_embs, faiss_idx, args.topk, args.neg_per_pos)
    val_examples = build_pairs_from_topk(val_pairs, passages_list, val_query_texts, val_query_embs, faiss_idx, args.topk, args.neg_per_pos)
    print("Train examples:", len(train_examples))
    print("Val examples:", len(val_examples))
    # prepare for BERT
    tokenizer = BertTokenizerFast.from_pretrained(bert_path)
    model = TFBertForSequenceClassification.from_pretrained(bert_path, num_labels=2)
    print("Trainable params:", np.sum([np.prod(v.shape) for v in model.trainable_variables]))
    train_enc = tokenizer([e[0] for e in train_examples], [e[1] for e in train_examples], truncation=True, padding=True, max_length=256, return_tensors='np')
    val_enc = tokenizer([e[0] for e in val_examples], [e[1] for e in val_examples], truncation=True, padding=True, max_length=256, return_tensors='np')
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_enc), np.array([e[2] for e in train_examples], dtype=np.int32)))
    train_dataset = train_dataset.shuffle(10000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((dict(val_enc), np.array([e[2] for e in val_examples], dtype=np.int32)))
    val_dataset = val_dataset.shuffle(10000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    callbacks = [
        tf_keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        ),
        tf_keras.callbacks.ModelCheckpoint(
            filepath=args.out_ckpt,
            mode = 'min',
            save_weights_only=True,
            monitor='val_loss',
            save_best_only=True
        )
    ]
    steps_per_epoch = len(train_examples) // args.batch_size
    num_train_steps = steps_per_epoch * args.epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    optimizer, lr_schedule = create_optimizer(
        init_lr=2e-5,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        weight_decay_rate=0.01
    )
    model.compile(
        optimizer=optimizer,
        loss=tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks
    )
    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)
    model.save_weights(args.out_ckpt)
    print("Saved BERT reranker to", args.out_ckpt)

if __name__ == '__main__':
    main()
