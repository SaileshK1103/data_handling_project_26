
import re
from dataclasses import dataclass

from step3_retriever import ReferenceRangeRetriever
from step3b_embeddings import Embedder
from step3c_vector_store import VectorStore, build_vector_store


@dataclass
class HybridResult:
    doc: dict
    method: str 
    score: float


class HybridRetriever:
    def __init__(self, structured: ReferenceRangeRetriever = None,
                 vector_store: VectorStore = None, embedder: Embedder = None):
        self.structured = structured or ReferenceRangeRetriever()
        self.vector_store = vector_store or build_vector_store()
        self.embedder = embedder or Embedder()
        if self.embedder.backend == "tfidf_svd_fallback":
            import json
            from pathlib import Path
            corpus_path = Path(__file__).parent / "artifacts" / "step2_corpus.jsonl"
            texts = [json.loads(l)["text"] for l in open(corpus_path)]
            self.embedder.fit_fallback(texts)

    def retrieve_structured(self, parameter: str, specimen=None, gender="all",
                             age_group="all", category=None, unit=None,
                             condition=None, reference_type=None, top_k=1):
        structured fields -- this is the actual RAG path: embed the query,
        search the vector DB, optionally narrowed by any metadata you do
        know (e.g. gender, specimen)."""
        qvec = self.embedder.encode(query)[0]
        hits = self.vector_store.query(qvec, top_k=top_k, where=metadata_filter)
        return [
            HybridResult(doc={"doc_id": h["doc_id"], "text": h["text"], "metadata": h["metadata"]},
                         method="semantic", score=1 - h["distance"])
            for h in hits
        ]

    def retrieve(self, parameter: str = None, query: str = None, specimen=None,
                 gender="all", age_group="all", category=None, unit=None,
                 condition=None, reference_type=None, top_k=1, metadata_filter=None):
        if parameter:
            structured = self.retrieve_structured(
                parameter, specimen, gender, age_group, category, unit,
                condition, reference_type, top_k,
            )
            if structured and structured[0].method not in ("fuzzy",):
                return structured
            query = query or parameter

        if query:
            return self.retrieve_freetext(query, top_k=top_k, metadata_filter=metadata_filter)

        return []


if __name__ == "__main__":
    hybrid = HybridRetriever()

    print("=== structured path (clean fields) ===")
    for r in hybrid.retrieve(parameter="ALT", gender="all"):
        print(f"  [{r.method}] {r.doc['metadata']['parameter']}")

    print("\n=== free-text path (paraphrased question, no structured fields) ===")
    for r in hybrid.retrieve(query="is 52 U/L a high ALT result for an adult?", top_k=3):
        print(f"  [{r.method}, score={r.score:.3f}] {r.doc['metadata']['parameter']}")

    print("\n=== structured miss -> falls back to semantic ===")
    for r in hybrid.retrieve(parameter="liver enzyme SGPT test", top_k=3):
        print(f"  [{r.method}, score={r.score:.3f}] {r.doc['metadata']['parameter']}")
