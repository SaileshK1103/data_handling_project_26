"""
STEP 6b — Classic retrieve-then-generate RAG QA.

Step 4/6 together already form a RAG pipeline for the pipeline's actual
production use case (a fine-tuned extraction model produces clean fields ->
deterministic checker decides -> LLM explains the decision). This file adds
the more literal, general-purpose RAG pattern on top of the same index,
for the case that pipeline doesn't cover: an open-ended free-text question
with no structured extraction step in front of it at all
(e.g. "what's a healthy fasting glucose range for an adult, and what if
mine came back at 110?").

Flow: embed the question -> semantic search the vector DB (Step 3c, via
the hybrid retriever's free-text path) -> pass the top-k retrieved LabQAR
passages to Claude as context -> Claude answers grounded ONLY in that
context, explicitly instructed to say so if the retrieved passages don't
answer the question (rather than falling back on parametric knowledge,
which would defeat the point of retrieval-grounding for a medical-adjacent
use case).

This is deliberately a separate, smaller function from step4/step6 --
merging them would blur the distinction the rest of this pipeline is built
around: Step 4 make a decision from a NUMBER using EXACT retrieval, this
file ANSWERS A QUESTION using SEMANTIC retrieval. Different inputs,
different retrieval strategy, different failure modes -- worth evaluating
separately (see step5c_evaluate_rag.py's retrieval section for the semantic
side of this).
"""

import os

from step3d_hybrid_retriever import HybridRetriever

SYSTEM_PROMPT = """You are a lab-reference-range Q&A assistant. You are
given a user's question and a set of retrieved passages from LabQAR (a
curated reference-range dataset). Answer using ONLY the retrieved passages.

Rules:
- If the passages don't contain enough information to answer, say so
  plainly instead of guessing or using outside knowledge.
- Cite which retrieved parameter/context you're drawing from when relevant.
- Keep the answer to a few sentences.
- Do not give a diagnosis or treatment recommendation.
"""


def _format_context(hits) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] {h.doc['text']}")
    return "\n".join(lines)


class RagQA:
    def __init__(self, retriever: HybridRetriever = None, model: str = "gemini-3.8-flash",
                 api_key: str = None):
        self.retriever = retriever or HybridRetriever()
        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._client = None
        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                print("WARNING: google-genai package not installed -- pip install google-genai")

    def answer(self, question: str, top_k: int = 3, metadata_filter: dict = None) -> dict:
        hits = self.retriever.retrieve_freetext(question, top_k=top_k, metadata_filter=metadata_filter)
        context = _format_context(hits)

        if self._client is None:
            # No API key -- return the retrieved context itself rather than
            # a fabricated answer, so this still demonstrates the retrieval
            # half of the workflow without an LLM call.
            answer_text = ("[No GOOGLE_API_KEY set -- showing retrieved context only, "
                            "no generation step run]\n" + context)
        else:
            user_msg = f"Question: {question}\n\nRetrieved passages:\n{context}\n\nAnswer the question."
            try:
                from google.genai import types
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=user_msg,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                    ),
                )
                answer_text = resp.text.strip()
            except Exception as e:
                answer_text = f"[LLM call failed: {e}]\n" + context

        return {"question": question, "answer": answer_text, "retrieved": hits}


if __name__ == "__main__":
    qa = RagQA()
    questions = [
        "what's a healthy fasting glucose range for an adult?",
        "is a TSH of 6.2 mIU/L normal for an adult?",
        "what does it mean if my LDL cholesterol is elevated?",
    ]
    for q in questions:
        result = qa.answer(q, top_k=3)
        print(f"\nQ: {q}")
        print(f"A: {result['answer']}")
        print("Retrieved:")
        for h in result["retrieved"]:
            print(f"  [{h.score:.3f}] {h.doc['metadata']['parameter']}")
