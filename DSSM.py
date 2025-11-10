import argparse
import pandas as pd
import tensorflow as tf
import numpy as np

# ------------------------
# Default hyperparameters
# ------------------------
DEFAULTS = {
    'vocab_size': 50000,
    'max_len': 32,
    'embed_dim': 128,
    'hidden_sizes': [256, 128],
    'dropout': 0.2,
    'batch_size': 128,
    'lr': 1e-3,
    'epochs': 3,
    'checkpoint_dir': './ckpt_dssm_csv',
    'temperature': 0.05
}

# ------------------------
# Parser
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='DSSM TensorFlow Repro')
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--valid_csv', type=str, required=True)
    parser.add_argument('--vocab_size', type=int, default=DEFAULTS['vocab_size'])
    parser.add_argument('--max_len', type=int, default=DEFAULTS['max_len'])
    parser.add_argument('--embed_dim', type=int, default=DEFAULTS['embed_dim'])
    parser.add_argument('--hidden_sizes', type=lambda s: list(map(int, s.split(','))), default=','.join(map(str, DEFAULTS['hidden_sizes'])))
    parser.add_argument('--dropout', type=float, default=DEFAULTS['dropout'])
    parser.add_argument('--batch_size', type=int, default=DEFAULTS['batch_size'])
    parser.add_argument('--lr', type=float, default=DEFAULTS['lr'])
    parser.add_argument('--epochs', type=int, default=DEFAULTS['epochs'])
    parser.add_argument('--checkpoint_dir', type=str, default=DEFAULTS['checkpoint_dir'])
    parser.add_argument('--temperature', type=float, default=DEFAULTS['temperature'])
    args = parser.parse_args()
    return args

# ------------------------
# Tokenizer
# ------------------------
class SimpleTokenizer:
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.word2id = {'<PAD>':0, '<UNK>':1}
        self.id2word = {0:'<PAD>', 1:'<UNK>'}

    def fit_on_texts(self, texts):
        from collections import Counter
        cnt = Counter()
        for txt in texts:
            cnt.update(str(txt).split())
        for idx, (w, _) in enumerate(cnt.most_common(self.vocab_size-2), start=2):
            self.word2id[w] = idx
            self.id2word[idx] = w

    def texts_to_sequences(self, texts, max_len):
        seqs = []
        for txt in texts:
            seq = [self.word2id.get(w, 1) for w in str(txt).split()]
            seq = seq[:max_len] + [0]*(max_len - len(seq))
            seqs.append(seq)
        return np.array(seqs)

# ------------------------
# Dataset Loader
# ------------------------
def load_csv_dataset(train_csv, valid_csv, tokenizer, max_len):
    # Read CSV
    df_train = pd.read_csv(train_csv)
    df_valid = pd.read_csv(valid_csv)

    # Combine answers + finalpassage as passages
    train_passages = pd.concat([df_train['answers'], df_train['finalpassage']]).unique().tolist()
    valid_passages = pd.concat([df_valid['answers'], df_valid['finalpassage']]).unique().tolist()
    passages_list = list(set(train_passages + valid_passages))

    # queries
    queries_list = df_train['query'].unique().tolist()

    # Tokenizer fit
    tokenizer.fit_on_texts(list(df_train['query']) + list(df_train['answers']) + list(df_train['finalpassage']))

    # Create train pairs dict (qid -> list of positive passage ids)
    passage2id = {p:i for i,p in enumerate(passages_list)}
    train_pairs = {}
    for _, row in df_train.iterrows():
        q = row['query']
        a = row['answers']
        p = row['finalpassage']
        qid = q
        pos_ids = list({passage2id[a], passage2id[p]})  # 去重
        train_pairs[qid] = pos_ids

    return df_train, df_valid, tokenizer, train_pairs, passage2id

# ------------------------
# DSSM Model
# ------------------------
def build_dssm(vocab_size, embed_dim, hidden_sizes, dropout):
    query_input = tf.keras.Input(shape=(None,), dtype='int32')
    doc_input = tf.keras.Input(shape=(None,), dtype='int32')

    embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
    q_embed = tf.reduce_mean(embedding(query_input), axis=1)
    d_embed = tf.reduce_mean(embedding(doc_input), axis=1)

    for h in hidden_sizes:
        q_embed = tf.keras.layers.Dense(h, activation='relu')(q_embed)
        d_embed = tf.keras.layers.Dense(h, activation='relu')(d_embed)
        q_embed = tf.keras.layers.Dropout(dropout)(q_embed)
        d_embed = tf.keras.layers.Dropout(dropout)(d_embed)

    model = tf.keras.Model(inputs=[query_input, doc_input], outputs=[q_embed, d_embed])
    return model

# ------------------------
# InfoNCE Loss
# ------------------------
def info_nce_loss(q_embed, d_embed, temperature=0.05):
    q_norm = tf.math.l2_normalize(q_embed, axis=1)
    d_norm = tf.math.l2_normalize(d_embed, axis=1)
    logits = tf.matmul(q_norm, d_norm, transpose_b=True) / temperature
    labels = tf.range(tf.shape(logits)[0])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(loss)

# ------------------------
# Training Loop
# ------------------------
def train(args):
    tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)
    df_train, df_valid, tokenizer, train_pairs, passage2id = load_csv_dataset(
        args.train_csv, args.valid_csv, tokenizer, args.max_len
    )

    passages_list = list(passage2id.keys())
    train_queries = list(train_pairs.keys())

    # 构建验证 pairs
    val_pairs = {}
    for _, row in df_valid.iterrows():
        q = row['query']
        a = row['answers']
        p = row['finalpassage']
        val_pairs[q] = list({passage2id.get(a, 0), passage2id.get(p, 0)})

    model = build_dssm(args.vocab_size, args.embed_dim, args.hidden_sizes, args.dropout)
    optimizer = tf.keras.optimizers.Adam(args.lr)

    batch_size = args.batch_size
    max_len = args.max_len

    for epoch in range(args.epochs):
        # ------------------------
        # Training
        # ------------------------
        np.random.shuffle(train_queries)
        train_loss = []
        for i in range(0, len(train_queries), batch_size):
            batch_q = train_queries[i:i+batch_size]
            q_seqs, d_seqs = [], []
            for q in batch_q:
                pos_ids = train_pairs[q]
                pid = np.random.choice(pos_ids)  # 随机选择一个正样本
                q_seqs.append(tokenizer.texts_to_sequences([q], max_len)[0])
                d_seqs.append(tokenizer.texts_to_sequences([passages_list[pid]], max_len)[0])
            q_seqs = np.array(q_seqs)
            d_seqs = np.array(d_seqs)

            with tf.GradientTape() as tape:
                q_emb, d_emb = model([q_seqs, d_seqs], training=True)
                loss = info_nce_loss(q_emb, d_emb, temperature=args.temperature)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss.append(loss.numpy())

        # ------------------------
        # Validation
        # ------------------------
        val_loss_list = []
        val_queries = list(val_pairs.keys())
        for i in range(0, len(val_queries), batch_size):
            batch_q = val_queries[i:i+batch_size]
            q_seqs, d_seqs = [], []
            for q in batch_q:
                pos_ids = val_pairs[q]
                pid = np.random.choice(pos_ids)  # 随机选择一个正样本
                q_seqs.append(tokenizer.texts_to_sequences([q], max_len)[0])
                d_seqs.append(tokenizer.texts_to_sequences([passages_list[pid]], max_len)[0])
            if len(q_seqs) == 0:
                continue
            q_seqs = np.array(q_seqs)
            d_seqs = np.array(d_seqs)
            q_emb, d_emb = model([q_seqs, d_seqs], training=False)
            val_loss = info_nce_loss(q_emb, d_emb, temperature=args.temperature)
            val_loss_list.append(val_loss.numpy())

        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss_list)

        recall5 = evaluate_recall_at_k(model, tokenizer, val_pairs, passages_list, max_len=args.max_len, k=5)
        print(
            f'Epoch {epoch + 1}/{args.epochs} Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f} Recall@5: {recall5:.4f}')

        model.save_weights(f'{args.checkpoint_dir}/dssm_epoch{epoch+1}.ckpt')


def evaluate_recall_at_k(model, tokenizer, val_pairs, passages_list, max_len=32, k=5, batch_size=128):
    """
    计算验证集 Recall@k
    """
    # 1. 编码所有 passages
    all_passage_seqs = tokenizer.texts_to_sequences(passages_list, max_len)
    all_passage_seqs = np.array(all_passage_seqs)
    passage_embs = []
    for i in range(0, len(all_passage_seqs), batch_size):
        batch = all_passage_seqs[i:i + batch_size]
        _, d_emb = model([np.zeros_like(batch), batch], training=False)  # query 输入随便给一个0填充
        passage_embs.append(d_emb.numpy())
    passage_embs = np.vstack(passage_embs)
    passage_embs = passage_embs / np.linalg.norm(passage_embs, axis=1, keepdims=True)

    # 2. 遍历每个 query，计算 top-k
    recalls = []
    for q, pos_ids in val_pairs.items():
        q_seq = tokenizer.texts_to_sequences([q], max_len)
        q_seq = np.array(q_seq)
        q_emb, _ = model([q_seq, np.zeros_like(q_seq)], training=False)
        q_emb = q_emb.numpy()
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

        sims = np.dot(q_emb, passage_embs.T)  # [1, num_passages]
        topk_idx = sims.argsort(axis=1)[:, -k:][:, ::-1]  # top-k indices
        # 检查正样本是否在 top-k
        hit = any(pid in topk_idx[0] for pid in pos_ids)
        recalls.append(hit)
    recall_at_k = np.mean(recalls)
    return recall_at_k


# ------------------------
# Main
# ------------------------
if __name__ == '__main__':
    args = parse_args()
    train(args)