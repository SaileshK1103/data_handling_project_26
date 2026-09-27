"""
STEP 3d — Hybrid retriever.

This is the actual RAG retrieval component: it decides, per query, whether
structured exact-match or semantic vector search is the right tool, instead
of picking one strategy for the whole system.

    caller gives clean structured fields (parameter, specimen, gender, ...)
        -> ReferenceRangeRetriever (step3), exact match. Free, deterministic,
           100% precision when the fields are right (see step5_evaluate.py).

    caller gives free text only (a natural-language question, a clinician's
    typed note, an LLM-extracted-but-imperfect parameter name with no exact
    or alias match)
        -> VectorStore (step3c) semantic search over the same corpus,
           optionally still metadata-filtered (e.g. you know the gender/
           specimen from context even if the parameter name is fuzzy).

This mirrors why the assignment asks for embeddings + a vector DB
specifically: they earn their cost precisely in the free-text case, which
the original structured-only design (step3) cannot handle at all, not even
via its TF-IDF fuzzy fallback (lexical overlap only -- see step5c's measured
comparison). For this project's actual query pattern (a fine-tuned
extraction model producing clean fields), structured retrieval should stay
the default; semantic search is the fallback for exactly the queries where
lexical/structured methods have no chance.
"""

import re
from dataclasses import dataclass

from step3_retriever import ReferenceRangeRetriever
from step3b_embeddings import Embedder
from step3c_vector_store import VectorStore, build_vector_store


@dataclass
class HybridResult:
    doc: dict
    method: str  # "exact" | "exact_via_alias" | "exact_fallback_*" | "semantic"
    score: float


class HybridRetriever:
    def __init__(self, structured: ReferenceRangeRetriever = None,
                 vector_store: VectorStore = None, embedder: Embedder = None):
        self.structured = structured or ReferenceRangeRetriever()
        self.vector_store = vector_store or build_vector_store()
        self.embedder = embedder or Embedder()
        if self.embedder.backend == "tfidf_svd_fallback":
            # fit on the same texts the vector store was built from
            import json
            from pathlib import Path
            corpus_path = Path(__file__).parent / "artifacts" / "step2_corpus.jsonl"
            texts = [json.loads(l)["text"] for l in open(corpus_path)]
            self.embedder.fit_fallback(texts)

    def retrieve_structured(self, parameter: str, specimen=None, gender="all",
                             age_group="all", category=None, unit=None,
                             condition=None, reference_type=None, top_k=1):
        """Use when the caller has clean structured fields. Thin passthrough
        to step3's exact-match retriever -- kept here so callers only need
        to import HybridRetriever."""
        results = self.structured.retrieve(
            parameter, specimen, gender, age_group, category, unit,
            condition, reference_type, top_k,
        )
        return [HybridResult(doc=r.doc, method=r.method, score=r.score) for r in results]

    def retrieve_freetext(self, query: str, top_k: int = 3, metadata_filter: dict = None):
        """Use when the caller has a free-text question instead of clean
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
        """
        Single entry point implementing the routing policy described in the
        module docstring: try structured exact match first if `parameter`
        is given; fall back to (or directly use, if only `query` is given)
        semantic search over the vector DB.
        """
        if parameter:
            structured = self.retrieve_structured(
                parameter, specimen, gender, age_group, category, unit,
                condition, reference_type, top_k,
            )
            if structured and structured[0].method not in ("fuzzy",):
                return structured
            # exact match found nothing (or only a weak lexical fuzzy hit) --
            # fall through to semantic search using the parameter name (plus
            # any context) as the query text.
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
