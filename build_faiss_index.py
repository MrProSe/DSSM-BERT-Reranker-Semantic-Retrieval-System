# build_faiss_index.py
import argparse, numpy as np, faiss
def parse_args():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--passage_embs', required=True)
    p.add_argument('--index_out', default='faiss_index.ivf')
    p.add_argument('--use_gpu', action='store_true')
    return p.parse_args()

def main():
    args = parse_args()
    data = np.load(args.passage_embs, allow_pickle=True)
    embs = data['embeddings'].astype('float32')
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    if args.use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(embs)
    if args.use_gpu:
        index_cpu = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index_cpu, args.index_out)
    else:
        faiss.write_index(index, args.index_out)
    print("Saved faiss index to", args.index_out)

if __name__ == '__main__':
    main()
