"""
STEP 3c — Vector database.

Indexes the embeddings from Step 3b (with their metadata from Step 2) in a
vector database and exposes similarity search, optionally filtered by
metadata (Chroma's `where` clause) -- so you can still say "semantically
search for ALT, but only rows where gender=Female" instead of a pure
nearest-neighbor search over the whole corpus.

Why ChromaDB
------------
The corpus is 550 short documents. That's small enough that the *choice*
of vector database barely matters for latency or recall -- a hosted,
horizontally-scaled vector DB (Pinecone, Weaviate, Qdrant Cloud, ...) would
be solving a scaling problem this project doesn't have, at the cost of an
external service dependency and, for most of them, a paid tier. Chroma is:
  - embedded / in-process (no server to stand up, works directly in Colab),
  - persistent to disk (`artifacts/chroma_db/`) so you don't re-embed on
    every run,
  - has native metadata filtering, which this corpus actually needs
    (gender/specimen/age_group/category/condition all change what the
    "correct" answer is -- see step2_build_corpus.py's docstring).
For a corpus two or three orders of magnitude larger, or one that needed
to scale across machines, a managed vector DB would be the right call
instead -- the interface below (`VectorStore.query`) is intentionally
narrow so swapping the backend later only touches this one file.

Offline fallback
-----------------
Same reasoning as step3b: if `chromadb` isn't installed, this file falls
back to a plain numpy cosine-similarity search over the same vectors and
supports the same metadata-filter interface, so the rest of the pipeline
can be exercised end-to-end without an internet connection. Use the Chroma
backend for your actual submission -- the fallback is for offline dev only.
"""

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
        # Chroma metadata values must be str/int/float/bool -- None isn't
        # allowed, so normalize missing fields to "" for storage. Callers
        # get "" back for an absent filter value; treat "" as "unset" the
        # same way the rest of the pipeline treats None.
        metadatas = []
        for i in doc_ids:
            m = dict(by_id[i]["metadata"])
            for k, v in list(m.items()):
                if v is None:
                    m[k] = ""
            metadatas.append(m)

        if self.backend == "chromadb":
            # Recreate the collection each time index() is called, so
            # re-running Step 2/3b and re-indexing never leaves stale rows.
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
        """
        Returns a list of dicts: {doc_id, text, metadata, distance}
        (distance: smaller = more similar; both backends use cosine distance).

        where: exact-match metadata filter, e.g. {"parameter_norm": "alt"}
               or {"gender": "Female"}. None = no filter.
        """
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
            sims = sub_vectors @ query_vector.T  # vectors are pre-normalized -> cosine sim
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

    # Demo: a free-text, non-templated query -- this is the case exact-match
    # retrieval in step3_retriever.py cannot handle at all.
    embedder = Embedder()
    if embedder.backend == "tfidf_svd_fallback":
        # the fallback needs to be fit on the same corpus it will search
        import json as _json
        texts = [_json.loads(l)["text"] for l in open(ART_DIR / "step2_corpus.jsonl")]
        embedder.fit_fallback(texts)

    query = "what is a normal ALT level for an adult"
    qvec = embedder.encode(query)[0]
    results = store.query(qvec, top_k=3)
    print(f"\nQuery: {query!r}")
    for r in results:
        print(f"  [{r['distance']:.3f}] {r['metadata']['parameter']} -- {r['text']}")
