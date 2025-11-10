# train_dssm.py
import argparse, os
import numpy as np, tensorflow as tf
from utils import SimpleTokenizer, load_csv_pairs, save_meta

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--train_csv', required=True)
    p.add_argument('--valid_csv', required=True)
    p.add_argument('--vocab_size', type=int, default=50000)
    p.add_argument('--max_len_query', type=int, default=32)
    p.add_argument('--max_len_passage', type=int, default=128)
    p.add_argument('--embed_dim', type=int, default=128)
    p.add_argument('--hidden_sizes', type=str, default='256,128')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--ckpt_dir', default='./ckpt_dssm')
    return p.parse_args()

def build_dssm(vocab_size, embed_dim, hidden_sizes, dropout=0.2, max_len_q=32, max_len_p=128):
    q_in = tf.keras.Input((max_len_q,), dtype='int32', name='query_input')
    d_in = tf.keras.Input((max_len_p,), dtype='int32', name='doc_input')
    embedding = tf.keras.layers.Embedding(vocab_size, embed_dim, name='tok_embed')
    q_seq = embedding(q_in); d_seq = embedding(d_in)
    q_vec = tf.reduce_mean(q_seq, axis=1); d_vec = tf.reduce_mean(d_seq, axis=1)
    for h in hidden_sizes:
        q_vec = tf.keras.layers.Dense(h, activation='relu')(q_vec)
        q_vec = tf.keras.layers.Dropout(dropout)(q_vec)
        d_vec = tf.keras.layers.Dense(h, activation='relu')(d_vec)
        d_vec = tf.keras.layers.Dropout(dropout)(d_vec)
    model = tf.keras.Model([q_in,d_in],[q_vec,d_vec])
    return model

def info_nce(q_emb, d_emb, temp=0.05):
    qn = tf.math.l2_normalize(q_emb, axis=1)
    dn = tf.math.l2_normalize(d_emb, axis=1)
    logits = tf.matmul(qn, dn, transpose_b=True) / temp
    labels = tf.range(tf.shape(logits)[0])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(loss)

def main():
    args = parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    train_pairs, val_pairs, passages_list = load_csv_pairs(args.train_csv, args.valid_csv)
    # build tokenizer
    tok = SimpleTokenizer(vocab_size=args.vocab_size)
    tok.fit_on_texts(list(train_pairs.keys()) + passages_list)
    # vocab_size = len(tok.word2id)  # consistent: no +1
    vocab_size = len(tok.word2id)
    hidden_sizes = list(map(int, args.hidden_sizes.split(',')))
    model = build_dssm(vocab_size, args.embed_dim, hidden_sizes, max_len_q=args.max_len_query, max_len_p=args.max_len_passage)
    opt = tf.keras.optimizers.Adam(args.lr)

    train_qs = list(train_pairs.keys())
    bs = args.batch_size
    for epoch in range(args.epochs):
        np.random.shuffle(train_qs)
        losses=[]
        for i in range(0, len(train_qs), bs):
            batch_q = train_qs[i:i+bs]
            q_texts=[]; d_texts=[]
            for q in batch_q:
                pos = train_pairs.get(q, [])
                if not pos: continue
                pid = np.random.choice(pos)
                q_texts.append(q); d_texts.append(passages_list[pid])
            if not q_texts: continue
            q_seqs = tok.texts_to_sequences(q_texts, args.max_len_query)
            d_seqs = tok.texts_to_sequences(d_texts, args.max_len_passage)
            with tf.GradientTape() as tape:
                q_emb, d_emb = model([q_seqs, d_seqs], training=True)
                loss = info_nce(q_emb, d_emb)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
            losses.append(float(loss.numpy()))
        print(f"Epoch {epoch+1}/{args.epochs} TrainLoss {np.mean(losses):.4f}")
        model.save_weights(os.path.join(args.ckpt_dir, f"dssm_epoch{epoch+1}.ckpt"))
        # save meta (passages and tokenizer)
        save_meta(os.path.join(args.ckpt_dir, f"meta_epoch{epoch+1}.npz"), passages_list, tok.word2id)
    # final
    model.save_weights(os.path.join(args.ckpt_dir, "dssm_final.ckpt"))
    save_meta(os.path.join(args.ckpt_dir, "meta_final.npz"), passages_list, tok.word2id)
    print("DSSM training done.")

if __name__ == '__main__':
    main()
