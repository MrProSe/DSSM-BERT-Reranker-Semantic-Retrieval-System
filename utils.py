# utils.py
import numpy as np
import pandas as pd
from collections import Counter
import json

class SimpleTokenizer:
    def __init__(self, vocab_size=50000):
        # keep PAD and UNK in word2id by default
        self.word2id = {'<PAD>': 0, '<UNK>': 1}
        self.id2word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = vocab_size

    def fit_on_texts(self, texts):
        cnt = Counter()
        for t in texts:
            cnt.update(str(t).split())
        for idx, (w, _) in enumerate(cnt.most_common(self.vocab_size - len(self.word2id)), start=len(self.word2id)):
            if w not in self.word2id:
                self.word2id[w] = idx
                self.id2word[idx] = w

    def texts_to_sequences(self, texts, max_len):
        seqs = []
        for txt in texts:
            tokens = str(txt).split()
            ids = [self.word2id.get(t, self.word2id['<UNK>']) for t in tokens]
            if len(ids) >= max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [self.word2id['<PAD>']] * (max_len - len(ids))
            seqs.append(ids)
        return np.array(seqs, dtype=np.int32)

def load_csv_pairs(train_csv, valid_csv=None):
    df_train = pd.read_csv(train_csv)
    df_valid = pd.read_csv(valid_csv) if valid_csv is not None else None

    combined_passages = pd.concat([df_train['answers'], df_train['finalpassage']]) if df_valid is None else pd.concat([df_train['answers'], df_train['finalpassage'], df_valid['answers'], df_valid['finalpassage']])
    passages_list = []
    seen = set()
    for p in combined_passages.tolist():
        if isinstance(p, str) and p.strip() and p not in seen:
            seen.add(p); passages_list.append(p)

    passage2id = {p:i for i,p in enumerate(passages_list)}

    def build_pairs(df):
        pairs = {}
        for _, row in df.iterrows():
            q = row['query']
            a = row['answers']; p = row['finalpassage']
            pos = []
            if a in passage2id: pos.append(passage2id[a])
            if p in passage2id and passage2id[p] not in pos: pos.append(passage2id[p])
            pos = [pid for pid in pos if pid < len(passages_list)]
            if pos: pairs.setdefault(q, []).extend(pos)
        return pairs

    train_pairs = build_pairs(df_train)
    val_pairs = build_pairs(df_valid) if df_valid is not None else {}

    return train_pairs, val_pairs, passages_list

def save_meta(path, passages, word2id):
    np.savez(path, passages=np.array(passages, dtype=object), word2id=word2id)

def load_meta(path):
    data = np.load(path, allow_pickle=True)
    passages = data['passages'].tolist()
    word2id = data['word2id'].item()
    return passages, word2id
