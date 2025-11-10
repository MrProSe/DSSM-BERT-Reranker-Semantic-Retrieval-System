# encode_passages.py
import argparse, numpy as np
import tensorflow as tf
from utils import load_meta, SimpleTokenizer

def build_full_model(vocab_size, embed_dim=128, hidden_sizes=[256,128], dropout=0.2, max_len_q=32, max_len_p=128):
    q_in = tf.keras.Input((max_len_q,), dtype='int32')
    d_in = tf.keras.Input((max_len_p,), dtype='int32')
    embedding = tf.keras.layers.Embedding(vocab_size, embed_dim, name='tok_embed')
    q_seq = embedding(q_in); d_seq = embedding(d_in)
    q_vec = tf.reduce_mean(q_seq, axis=1); d_vec = tf.reduce_mean(d_seq, axis=1)
    for h in hidden_sizes:
        q_vec = tf.keras.layers.Dense(h, activation='relu')(q_vec); q_vec = tf.keras.layers.Dropout(dropout)(q_vec)
        d_vec = tf.keras.layers.Dense(h, activation='relu')(d_vec); d_vec = tf.keras.layers.Dropout(dropout)(d_vec)
    model = tf.keras.Model([q_in,d_in],[q_vec,d_vec])
    return model

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--dssm_meta', required=True)
    p.add_argument('--dssm_ckpt', required=True)
    p.add_argument('--out', default=None)
    p.add_argument('--max_len_passage', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=256)
    return p.parse_args()

def main():
    args = parse_args()
    passages, word2id = load_meta(args.dssm_meta)
    tok = SimpleTokenizer(vocab_size=len(word2id))
    tok.word2id = word2id
    tok.id2word = {v:k for k,v in word2id.items()}

    vocab_size = len(tok.word2id)
    model = build_full_model(vocab_size, max_len_q=32, max_len_p=args.max_len_passage)
    model.load_weights(args.dssm_ckpt)

    p_seqs = tok.texts_to_sequences(passages, args.max_len_passage)
    embs=[]
    for i in range(0, len(p_seqs), args.batch_size):
        b = p_seqs[i:i+args.batch_size]
        dummy_queries = np.zeros((len(b), 32), dtype=np.int32)
        _, d_emb = model([dummy_queries, b], training=False)
        d_emb = d_emb.numpy()
        d_emb = d_emb / (np.linalg.norm(d_emb, axis=1, keepdims=True)+1e-12)
        embs.append(d_emb)
    embs = np.vstack(embs)

    # 过滤 NaN 或非字符串 passage
    clean_passages = []
    clean_embs = []
    for p, e in zip(passages, embs):
        if isinstance(p, str) and p.strip():
            clean_passages.append(p)
            clean_embs.append(e)
        else:
            print(f"[WARN] Skipping invalid passage: {p}")
    clean_passages = np.array(clean_passages, dtype=object)
    clean_embs = np.vstack(clean_embs)

    out = args.out or args.dssm_meta.replace('.npz','_passage_embs.npz')
    np.savez(out, embeddings=clean_embs, passages=clean_passages)
    print(f"[INFO] Saved clean passage embeddings: {out}, total passages = {len(clean_passages)}")

if __name__ == '__main__':
    main()
