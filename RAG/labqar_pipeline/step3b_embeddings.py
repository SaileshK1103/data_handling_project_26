
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
