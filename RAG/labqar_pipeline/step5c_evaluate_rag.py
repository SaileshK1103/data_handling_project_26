
import json
import re
from pathlib import Path

from step3_retriever import ReferenceRangeRetriever
from step3d_hybrid_retriever import HybridRetriever
from step4_range_checker_rag import RangeCheckerRAG
from step6_llm_explain import Explainer

ART_DIR = Path(__file__).parent / "artifacts"


# ---------------------------------------------------------------------------
# A. Retrieval evaluation
# ---------------------------------------------------------------------------

def _lexical_baseline_search(structured_retriever: ReferenceRangeRetriever, query: str, top_k: int):
    DIFFERENT status than the one it was given (naive keyword check --
    good enough to catch a generation that ignored its instructions, not
    a substitute for human review)."""
    text = explanation.lower()
    other_statuses = [s for s in STATUS_WORDS if s != status]
    for other in other_statuses:
        for kw in STATUS_WORDS[other]:
            if kw in text:
                return False
    return True


def _numeric_groundedness(explanation: str, result: dict, value: float, tolerance=0.05):
