# pipeline_rerank.py
import argparse, numpy as np, faiss, tensorflow as tf
from transformers import BertTokenizerFast, TFBertForSequenceClassification

from MS_MARCO.train_bert_reranker import bert_path
from utils import load_meta, SimpleTokenizer

def parse_args():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--dssm_meta', required=True)
    p.add_argument('--dssm_ckpt', required=True)
    p.add_argument('--passage_embs', required=True)
    p.add_argument('--faiss_index', required=True)
    p.add_argument('--bert_ckpt', required=True)
    p.add_argument('--max_len_query', type=int, default=32)
    p.add_argument('--topk', type=int, default=50)
    return p.parse_args()

def build_full_model(vocab_size, embed_dim=128, hidden_sizes=[256,128], max_len_q=32, max_len_p=128):
    q_in = tf.keras.Input((max_len_q,), dtype='int32')
    d_in = tf.keras.Input((max_len_p,), dtype='int32')
    embedding = tf.keras.layers.Embedding(vocab_size, embed_dim, name='tok_embed')
    q_seq = embedding(q_in); d_seq = embedding(d_in)
    q_vec = tf.reduce_mean(q_seq, axis=1); d_vec = tf.reduce_mean(d_seq, axis=1)
    for h in hidden_sizes:
        q_vec = tf.keras.layers.Dense(h, activation='relu')(q_vec)
        d_vec = tf.keras.layers.Dense(h, activation='relu')(d_vec)
    model = tf.keras.Model([q_in,d_in],[q_vec,d_vec])
    return model

def main():
    args = parse_args()
    passages, word2id = load_meta(args.dssm_meta)
    pen = np.load(args.passage_embs, allow_pickle=True)
    passage_embs = pen['embeddings'].astype('float32')
    passages_list = pen['passages'].tolist()
    # build dssm model to encode queries
    tok = SimpleTokenizer(vocab_size=len(word2id))
    tok.word2id = word2id
    tok.id2word = {v:k for k,v in word2id.items()}
    vocab_size = len(tok.word2id)
    dssm_model = build_full_model(vocab_size, max_len_q=args.max_len_query, max_len_p=128)
    dssm_model.load_weights(args.dssm_ckpt)
    # build/load faiss index
    idx = faiss.read_index(args.faiss_index)
    # load BERT
    bert_path = './models'
    bert_tok = BertTokenizerFast.from_pretrained(bert_path)
    bert = TFBertForSequenceClassification.from_pretrained(bert_path, num_labels=2)
    bert.load_weights(args.bert_ckpt)
    # interactive
    while True:
        q = input("Query (or 'exit'): ").strip()
        if q.lower() in ['exit','quit','q']: break
        q_seq = tok.texts_to_sequences([q], args.max_len_query)
        q_emb, _ = dssm_model([q_seq, np.zeros((1,128),dtype=np.int32)], training=False)
        q_emb = q_emb.numpy().astype('float32')
        D,I = idx.search(q_emb, args.topk)
        cand_ids = I[0].tolist()
        cand_texts = [passages_list[i] for i in cand_ids]
        # BERT rerank
        enc = bert_tok([q]*len(cand_texts), cand_texts, truncation=True, padding=True, max_length=256, return_tensors='tf')
        logits = bert(**enc).logits
        probs = tf.nn.softmax(logits, axis=-1)[:,1].numpy()
        order = probs.argsort()[::-1]
        for rank, idxpos in enumerate(order[:10], start=1):
            pid = cand_ids[idxpos]
            print(f"{rank}\tbert_score={probs[idxpos]:.4f}\tfaiss_score={D[0][idxpos]:.4f}\t{passages_list[pid][:200]}")

if __name__ == '__main__':
    main()