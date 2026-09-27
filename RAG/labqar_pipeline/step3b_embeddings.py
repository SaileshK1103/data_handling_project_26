"""
STEP 3b — Embeddings.

Turns each Step-2 corpus document's `text` field into a dense vector.

Why embeddings are needed at all (justification, not just requirement-checking)
---------------------------------------------------------------------------
step3_retriever.py's exact-match stage is the right *primary* strategy for
this corpus (550 rows, fully structured, your fine-tuned extraction model
feeds it clean fields) -- that reasoning doesn't change. But exact match
only works when the caller's `parameter` string already matches LabQAR's
vocabulary closely (or via the alias table). It cannot handle a genuinely
free-text query -- e.g. a clinician typing "what's a normal ALT level for
an adult male" instead of calling retrieve(parameter="ALT", gender="Male").
That is exactly the situation dense embeddings are for: they capture
semantic similarity ("normal ALT level" ~ "reference range for Alanine
aminotransferase") in a way a purely lexical method (character n-gram
TF-IDF, what the old fuzzy fallback used) cannot -- TF-IDF only overlaps on
shared substrings, so it fails on a paraphrase with no shared n-grams (see
step5c_evaluate_rag.py for a measured comparison, not just an assertion).

Model choice
------------
`sentence-transformers/all-MiniLM-L6-v2`:
  - 384-dim, ~80MB, runs on CPU in seconds for 550 short documents --
    no GPU needed, no API key, no per-call cost. For a corpus this size
    and this short (one sentence per doc), a small local model is a better
    fit than a large hosted embedding API: latency and cost from an API
    would dominate for no accuracy benefit at this scale.
  - Widely-used sentence-similarity benchmark performance is good for
    short factual sentences, which is exactly what step2's `text` field is.

If you want to swap in a hosted embedding API (Voyage, OpenAI, etc.) later
for a larger/messier corpus, only this file changes -- everything downstream
(the vector store, the hybrid retriever, the evaluation) talks to `Embedder`
through `.encode()` and doesn't care how the vectors were produced.

Offline fallback
-----------------
If `sentence-transformers` isn't installed (e.g. no internet in this
environment), this file falls back to a TF-IDF + Truncated SVD ("LSA")
embedding -- genuine dense vectors, just weaker semantics, built purely so
the rest of the pipeline (vector store, hybrid retriever, evaluation) can be
smoke-tested end-to-end without downloading model weights. THIS FALLBACK IS
NOT WHAT YOU SHOULD REPORT AS YOUR PROJECT'S EMBEDDING METHOD -- install
sentence-transformers (works fine in Colab, which has internet) and use
that for your actual submission and evaluation numbers.
"""

import json
import numpy as np
from pathlib import Path

ART_DIR = Path(__file__).parent / "artifacts"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.backend = None
        self._model = None
        self._svd = None
        self._tfidf = None
        self.dim = None
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self.backend = "sentence-transformers"
            self.dim = self._model.get_sentence_embedding_dimension()
        except ImportError:
            print("WARNING: sentence-transformers not installed -- using a TF-IDF+SVD "
                  "fallback for offline testing only. In Colab, run:\n"
                  "  pip install sentence-transformers\n"
                  "and re-instantiate Embedder() to use real semantic embeddings.")
            self.backend = "tfidf_svd_fallback"

    def fit_fallback(self, texts):
        """Only used by the offline fallback backend -- fits TF-IDF+SVD on
        the corpus itself. sentence-transformers needs no fitting (it's a
        pretrained model), so this is a no-op when that backend is active."""
        if self.backend != "tfidf_svd_fallback":
            return
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self._tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        X = self._tfidf.fit_transform(texts)
        n_components = min(128, X.shape[0] - 1, X.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(X)
        self.dim = n_components

    def encode(self, texts) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if self.backend == "sentence-transformers":
            return np.asarray(self._model.encode(texts, normalize_embeddings=True))
        else:
            if self._tfidf is None:
                raise RuntimeError("Call fit_fallback(corpus_texts) once before encode() "
                                    "when using the offline TF-IDF+SVD fallback.")
            X = self._tfidf.transform(texts)
            vecs = self._svd.transform(X)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return vecs / norms


def build_embeddings(corpus_path: Path = ART_DIR / "step2_corpus.jsonl",
                      out_path: Path = ART_DIR / "step3b_embeddings.npz",
                      model_name: str = DEFAULT_MODEL) -> Embedder:
    docs = [json.loads(line) for line in open(corpus_path)]
    texts = [d["text"] for d in docs]
    doc_ids = [d["doc_id"] for d in docs]

    embedder = Embedder(model_name)
    embedder.fit_fallback(texts)
    vectors = embedder.encode(texts)

    np.savez_compressed(out_path, doc_ids=np.array(doc_ids), vectors=vectors,
                         backend=embedder.backend, model_name=model_name)
    print(f"Encoded {len(texts)} documents -> {vectors.shape} ({embedder.backend}, "
          f"dim={embedder.dim})")
    print(f"Wrote {out_path}")
    return embedder


if __name__ == "__main__":
    build_embeddings()
