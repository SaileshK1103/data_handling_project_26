
import json
import numpy as np
from pathlib import Path

ART_DIR = Path(__file__).parent / "artifacts"
CHROMA_DIR = ART_DIR / "chroma_db"
COLLECTION_NAME = "labqar_reference_ranges"


class VectorStore:
    def __init__(self, persist_dir: Path = CHROMA_DIR):
        self.persist_dir = persist_dir
        self.backend = None
        self._collection = None
        # Fallback-only state
        self._vectors = None
        self._doc_ids = None
        self._metadatas = None
        self._texts = None
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=str(persist_dir))
            self.backend = "chromadb"
        except ImportError:
            print("WARNING: chromadb not installed -- using an in-memory numpy "
                  "cosine-similarity fallback for offline testing only. In Colab, run:\n"
                  "  pip install chromadb\n"
                  "and use that backend for your actual submission/evaluation.")
            self.backend = "numpy_fallback"

    def index(self, corpus_path: Path, embeddings_path: Path):
        docs = [json.loads(line) for line in open(corpus_path, encoding="utf-8")]
        by_id = {d["doc_id"]: d for d in docs}

        npz = np.load(embeddings_path, allow_pickle=True)
        doc_ids = list(npz["doc_ids"])
        vectors = npz["vectors"]

        texts = [by_id[i]["text"] for i in doc_ids]
        metadatas = []
        for i in doc_ids:
            m = dict(by_id[i]["metadata"])
            for k, v in list(m.items()):
                if v is None:
                    m[k] = ""
            metadatas.append(m)

        if self.backend == "chromadb":
            try:
                self._client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            self._collection = self._client.create_collection(COLLECTION_NAME)
            self._collection.add(
                ids=doc_ids, embeddings=vectors.tolist(),
                documents=texts, metadatas=metadatas,
            )
            print(f"Indexed {len(doc_ids)} documents into Chroma collection "
                  f"'{COLLECTION_NAME}' at {self.persist_dir}")
        else:
            self._vectors = vectors
            self._doc_ids = doc_ids
            self._metadatas = metadatas
            self._texts = texts
            print(f"Indexed {len(doc_ids)} documents into the in-memory fallback store.")

    def query(self, query_vector: np.ndarray, top_k: int = 5, where: dict = None):
        query_vector = np.asarray(query_vector).reshape(1, -1)

        if self.backend == "chromadb":
            kwargs = {"query_embeddings": query_vector.tolist(), "n_results": top_k}
            if where:
                kwargs["where"] = where if len(where) > 1 else where
                if len(where) > 1:
                    kwargs["where"] = {"$and": [{k: v} for k, v in where.items()]}
            res = self._collection.query(**kwargs)
            out = []
            for i in range(len(res["ids"][0])):
                out.append({
                    "doc_id": res["ids"][0][i],
                    "text": res["documents"][0][i],
                    "metadata": res["metadatas"][0][i],
                    "distance": res["distances"][0][i],
                })
            return out
        else:
            mask = np.ones(len(self._doc_ids), dtype=bool)
            if where:
                for k, v in where.items():
                    mask &= np.array([m.get(k) == v for m in self._metadatas])
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                return []
            sub_vectors = self._vectors[idxs]
            sims = sub_vectors @ query_vector.T 
            sims = sims.flatten()
            order = np.argsort(sims)[::-1][:top_k]
            out = []
            for j in order:
                i = idxs[j]
                out.append({
                    "doc_id": self._doc_ids[i],
                    "text": self._texts[i],
                    "metadata": self._metadatas[i],
                    "distance": float(1 - sims[j]),
                })
            return out


def build_vector_store(corpus_path: Path = ART_DIR / "step2_corpus.jsonl",
                        embeddings_path: Path = ART_DIR / "step3b_embeddings.npz",
                        persist_dir: Path = CHROMA_DIR) -> VectorStore:
    store = VectorStore(persist_dir)
    store.index(corpus_path, embeddings_path)
    return store


if __name__ == "__main__":
    from step3b_embeddings import Embedder

    store = build_vector_store()

    embedder = Embedder()
    if embedder.backend == "tfidf_svd_fallback":
        import json as _json
        texts = [_json.loads(l)["text"] for l in open(ART_DIR / "step2_corpus.jsonl")]
        embedder.fit_fallback(texts)

    query = "what is a normal ALT level for an adult"
    qvec = embedder.encode(query)[0]
    results = store.query(qvec, top_k=3)
    print(f"\nQuery: {query!r}")
    for r in results:
        print(f"  [{r['distance']:.3f}] {r['metadata']['parameter']} -- {r['text']}")
