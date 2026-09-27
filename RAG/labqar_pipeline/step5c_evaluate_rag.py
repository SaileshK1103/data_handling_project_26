"""
STEP 5c — RAG evaluation: retrieval AND generated output, as the brief asks
for explicitly, not just "does the app run".

Two separate things get measured here, because they can fail independently
and a single end-to-end pass/fail number would hide which one broke:

  A. RETRIEVAL evaluation (semantic vs lexical, on free-text queries)
     -----------------------------------------------------------------
     step5_evaluate.py already measures structured (exact-match) retrieval
     on Set_1's templated questions -- that's the right test for that
     component, but it's not a test of the embeddings/vector DB at all
     (see step5b's docstring for why). Here, on the free-text paraphrases
     from step5b, we measure:
       - Accuracy@1 / Accuracy@3 : did the correct LabQAR row appear in the
         top-1 / top-3 semantic search results?
       - MRR (mean reciprocal rank): rewards getting the right row ranked
         near the top even when it's not #1.
     ...and we compare semantic (embeddings) against the OLD lexical
     TF-IDF-char-ngram method from step3_retriever.py on the exact same
     queries, so "embeddings help here" is a measured claim, not an
     assumption. This is also the direct evidence for the "justify your
     technical choices" requirement: the numbers below are the
     justification for using embeddings over pure lexical fuzzy-matching.

  B. GENERATION evaluation (Step 6's LLM explanations)
     ---------------------------------------------------
     There's no human-written gold explanation to compare against, so this
     is NOT a similarity-to-reference metric (BLEU/ROUGE would be
     meaningless here with no reference text). Instead, two automatic,
     rule-based groundedness checks -- appropriate because the generation
     step is deliberately constrained (Step 6's system prompt: explain a
     given verdict, don't re-derive it):
       - verdict_consistency: does the generated text avoid contradicting
         the Low/Normal/High verdict it was given? (naive keyword check for
         the opposite status words)
       - numeric_groundedness: do the numbers the explanation states match
         the retrieved reference bounds (within rounding), rather than
         being invented? Any number in the text that doesn't correspond to
         the retrieved bounds/value is flagged as a potential hallucination.
     If you want a richer generation metric later, the natural next step is
     an LLM-as-judge pass (have a second Claude call score faithfulness
     1-5) -- left out here to keep this evaluation script runnable without
     burning API calls every run; see the comment in `evaluate_generation`
     for where to add it.
"""

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
    """The OLD method (step3's TF-IDF char n-gram fuzzy fallback), run
    directly on free text, as the comparison point for the semantic search."""
    matches = structured_retriever._fuzzy_param_match(query, top_k=top_k)
    return [name for name, _score in matches]


def evaluate_retrieval(top_k: int = 3):
    template_queries = json.loads((ART_DIR / "step5b_eval_queries.json").read_text())
    hard_queries = json.loads((ART_DIR / "step5b_hard_queries.json").read_text())

    structured = ReferenceRangeRetriever()
    hybrid = HybridRetriever(structured=structured)

    def run(queries, label):
        semantic_hits_at_1, semantic_hits_at_k, semantic_rr = 0, 0, 0.0
        lexical_hits_at_1, lexical_hits_at_k, lexical_rr = 0, 0, 0.0
        for q in queries:
            gold = q["gold_parameter_norm"]

            results = hybrid.retrieve_freetext(q["query"], top_k=top_k)
            ranked = [r.doc["metadata"]["parameter_norm"] for r in results]
            if ranked and ranked[0] == gold:
                semantic_hits_at_1 += 1
            if gold in ranked:
                semantic_hits_at_k += 1
                semantic_rr += 1.0 / (ranked.index(gold) + 1)

            lex_ranked = _lexical_baseline_search(structured, q["query"], top_k)
            if lex_ranked and lex_ranked[0] == gold:
                lexical_hits_at_1 += 1
            if gold in lex_ranked:
                lexical_hits_at_k += 1
                lexical_rr += 1.0 / (lex_ranked.index(gold) + 1)

        n = len(queries)
        print(f"\n--- {label} (n={n}) ---")
        print(f"{'method':<12} {'Acc@1':>8} {'Acc@' + str(top_k):>8} {'MRR':>8}")
        print(f"{'semantic':<12} {100*semantic_hits_at_1/n:>7.1f}% {100*semantic_hits_at_k/n:>7.1f}% {semantic_rr/n:>8.3f}")
        print(f"{'lexical':<12} {100*lexical_hits_at_1/n:>7.1f}% {100*lexical_hits_at_k/n:>7.1f}% {lexical_rr/n:>8.3f}")
        return {
            "semantic": {"acc@1": semantic_hits_at_1/n, f"acc@{top_k}": semantic_hits_at_k/n, "mrr": semantic_rr/n},
            "lexical": {"acc@1": lexical_hits_at_1/n, f"acc@{top_k}": lexical_hits_at_k/n, "mrr": lexical_rr/n},
        }

    print("=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)
    template_scores = run(template_queries, "Template queries (parameter name embedded verbatim -- "
                                             "sanity check only, NOT a real semantic test)")
    hard_scores = run(hard_queries, "Hard queries (hand-written synonyms/paraphrases, "
                                     "NO literal parameter-name overlap -- this is the real test)")

    if hybrid.embedder.backend == "tfidf_svd_fallback":
        print("\nNOTE: semantic numbers above used the offline TF-IDF+SVD fallback "
              "embedder (no internet in this environment). Because that fallback is "
              "ITSELF lexical under the hood (built from the same character n-grams as "
              "the 'lexical' baseline, just dimensionality-reduced), it does not show "
              "the real advantage a pretrained sentence-transformers model has on the "
              "hard/paraphrase set -- on the template set it can even lose to raw TF-IDF, "
              "since SVD throws away some of the exact substring signal that raw TF-IDF "
              "exploits directly. Re-run this file in Colab after "
              "`pip install sentence-transformers chromadb` for the numbers that "
              "actually support 'embeddings help with paraphrase' -- expect the gap to "
              "open up specifically on the hard set above, which is the one that matters.")
    return {"template": template_scores, "hard": hard_scores}


# ---------------------------------------------------------------------------
# B. Generation evaluation
# ---------------------------------------------------------------------------

NUM_RE = re.compile(r"-?\d+\.?\d*")
STATUS_WORDS = {
    "High": ["high", "elevated", "above"],
    "Low": ["low", "below", "deficient"],
    "Normal": ["normal", "within", "expected range"],
}


def _verdict_consistency(explanation: str, status: str) -> bool:
    """True if the explanation doesn't use language belonging to a
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
    """Every number mentioned in the explanation should correspond to
    either the measured value or the retrieved lower/upper bound (within a
    small tolerance for rounding). Returns (grounded: bool, unmatched: list)."""
    known = [v for v in (value, result.get("lower_bound"), result.get("upper_bound")) if v is not None]
    found = [float(x) for x in NUM_RE.findall(explanation)]
    unmatched = []
    for f in found:
        if not any(abs(f - k) <= max(tolerance * abs(k), tolerance) for k in known):
            unmatched.append(f)
    return (len(unmatched) == 0), unmatched


def evaluate_generation(n_cases: int = 20, seed: int = 42):
    import random
    rng = random.Random(seed)

    checker = RangeCheckerRAG()
    explainer = Explainer()  # uses GOOGLE_API_KEY if set, else the template fallback

    registry = json.loads((ART_DIR / "step1_reference_registry.json").read_text(encoding="utf-8"))
    two_sided = [r for r in registry if r["lower_bound"] is not None and r["upper_bound"] is not None]
    sample = rng.sample(two_sided, min(n_cases, len(two_sided)))

    consistent, grounded, results_log = 0, 0, []
    for row in sample:
        mid = (row["lower_bound"] + row["upper_bound"]) / 2
        # nudge the value outside the range half the time, so we exercise
        # High/Low explanations too, not only Normal
        value = row["upper_bound"] * 1.3 if rng.random() < 0.5 else mid

        result = checker.evaluate(row["parameter"], value, specimen=row["specimen"],
                                   gender=row["gender"], age_group=row["age_group"])
        explanation = explainer.explain(row["parameter"], value, row["unit"], result)

        ok_verdict = _verdict_consistency(explanation, result["status"])
        ok_numbers, unmatched = _numeric_groundedness(explanation, result, value)
        consistent += ok_verdict
        grounded += ok_numbers
        results_log.append({
            "parameter": row["parameter"], "value": value, "status": result["status"],
            "explanation": explanation, "verdict_consistent": ok_verdict,
            "numerically_grounded": ok_numbers, "unmatched_numbers": unmatched,
        })

    n = len(sample)
    print("\n" + "=" * 70)
    print(f"GENERATION EVALUATION -- Step 6 explanations (n={n}, LLM backend="
          f"{'live API' if explainer._client else 'template fallback (no ANTHROPIC_API_KEY set)'})")
    print("=" * 70)
    print(f"Verdict-consistent:   {consistent}/{n} ({100*consistent/n:.1f}%)")
    print(f"Numerically grounded: {grounded}/{n} ({100*grounded/n:.1f}%)")
    failures = [r for r in results_log if not (r["verdict_consistent"] and r["numerically_grounded"])]
    if failures:
        print("\nFailing cases:")
        for f in failures[:5]:
            print(f"  {f['parameter']} = {f['value']:.2f} -> {f['status']}: {f['explanation']!r}")
            if f["unmatched_numbers"]:
                print(f"    unmatched numbers: {f['unmatched_numbers']}")
    # To go further: add an LLM-as-judge pass here, e.g. a second Explainer
    # call asking Claude to rate 1-5 how faithful `explanation` is to
    # `result`, and report the mean score alongside the two checks above.
    return results_log


if __name__ == "__main__":
    evaluate_retrieval(top_k=3)
    evaluate_generation(n_cases=20)
