import argparse, numpy as np, pandas as pd, tensorflow as tf
import tf_keras
from utils import load_meta, SimpleTokenizer

def build_full_model(vocab_size, embed_dim=128, hidden_sizes=[256,128], dropout=0.2, max_len_q=32, max_len_p=128):
    q_in = tf_keras.Input((max_len_q,), dtype='int32')
    d_in = tf_keras.Input((max_len_p,), dtype='int32')
    embedding = tf_keras.layers.Embedding(vocab_size, embed_dim, name='tok_embed')
    q_seq = embedding(q_in); d_seq = embedding(d_in)
    q_vec = tf.reduce_mean(q_seq, axis=1); d_vec = tf.reduce_mean(d_seq, axis=1)
    for h in hidden_sizes:
        q_vec = tf_keras.layers.Dense(h, activation='relu')(q_vec); q_vec = tf_keras.layers.Dropout(dropout)(q_vec)
        d_vec = tf_keras.layers.Dense(h, activation='relu')(d_vec); d_vec = tf_keras.layers.Dropout(dropout)(d_vec)
    model = tf_keras.Model([q_in,d_in],[q_vec,d_vec])
    return model

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--valid_csv', required=True)
    p.add_argument('--dssm_meta', required=True)
    p.add_argument('--dssm_ckpt', required=True)
    p.add_argument('--out', default=None)
    p.add_argument('--max_len_query', type=int, default=32)
    p.add_argument('--batch_size', type=int, default=256)
    return p.parse_args()

def main():
    args = parse_args()
    passages, word2id = load_meta(args.dssm_meta)
    df = pd.read_csv(args.valid_csv)
    queries = df['query'].unique().tolist()

    tok = SimpleTokenizer(vocab_size=len(word2id))
    tok.word2id = word2id
    tok.id2word = {v:k for k,v in word2id.items()}

    model = build_full_model(len(tok.word2id), max_len_q=args.max_len_query, max_len_p=32)
    model.load_weights(args.dssm_ckpt)

    embs=[]
    for i in range(0,len(queries), args.batch_size):
        batch = queries[i:i+args.batch_size]
        seqs = tok.texts_to_sequences(batch, args.max_len_query)
        q_emb, _ = model([seqs, np.zeros((len(seqs),32),dtype=np.int32)], training=False)
        q_emb = q_emb.numpy()
        q_emb = q_emb / (np.linalg.norm(q_emb,axis=1,keepdims=True)+1e-12)
        embs.append(q_emb)
    embs = np.vstack(embs)
    out = args.out or args.dssm_meta.replace('.npz','_val_query_embs.npz')
    np.savez(out, queries=np.array(queries, dtype=object), embeddings=embs)
    print("Saved validation query embeddings:", out)

if __name__ == '__main__':
    main()
