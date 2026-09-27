"""
STEP 4 — Deterministic range checker on top of the retriever.

This keeps the separation of concerns from your original design:
    RAG (Step 3)      -> finds the right reference range + context
    RangeChecker (here) -> pure arithmetic decision, no LLM involved
    LLM (Step 6)      -> turns the decision into a human-readable explanation

It is a drop-in replacement for range_checker.py's RangeChecker.evaluate():
same method name/signature-compatible call, but now backed by the LabQAR
registry (with alias resolution + specimen/age context) instead of a static
config/reference_ranges.csv, and without the "fabricate an upper bound"
behavior that produced some of Set_2's bad labels.
"""

from step3_retriever import ReferenceRangeRetriever


class RangeCheckerRAG:
    def __init__(self, retriever: ReferenceRangeRetriever = None):
        self.retriever = retriever or ReferenceRangeRetriever()

    def evaluate(self, parameter: str, value: float, specimen: str = None,
                 gender: str = "all", age_group: str = "all", category=None, unit: str = None,
                 condition=None, reference_type=None) -> dict:
        """
        Returns a dict (not just a string) so the caller keeps the retrieved
        evidence for logging / for the LLM explanation step -- this is the
        auditability the original design was going for.

        NOTE: earlier versions of this method silently dropped `condition`
        (cycle-phase context for hormone tests) instead of forwarding it to
        the retriever, even though step3's retriever already supported it.
        That alone accounted for the majority of Set_2's residual
        classification mismatches -- reproductive-hormone rows with no
        phase specified fell back to whichever row happened to be first in
        the corpus, not the phase-independent one. Fixed here by accepting
        and forwarding both `condition` and `reference_type`.
        """
        results = self.retriever.retrieve(
            parameter=parameter, specimen=specimen, gender=gender,
            age_group=age_group, category=category, unit=unit,
            condition=condition, reference_type=reference_type, top_k=1,
        )

        if not results:
            return {
                "status": "Normal",
                "flag_reason": "no_reference_range_found",
                "matched_parameter": None,
                "retrieval_method": None,
                "lower_bound": None,
                "upper_bound": None,
                "unit": None,
            }

        top = results[0]
        m = top.doc["metadata"]
        lower, upper = m["lower_bound"], m["upper_bound"]

        if lower is not None and value < lower:
            status = "Low"
        elif upper is not None and value > upper:
            status = "High"
        else:
            status = "Normal"

        return {
            "status": status,
            "flag_reason": None,
            "matched_parameter": m["parameter"],
            "retrieval_method": top.method,
            "lower_bound": lower,
            "upper_bound": upper,
            "unit": m["unit"],
        }


if __name__ == "__main__":
    checker = RangeCheckerRAG()
    cases = [
        ("Acetaminophen", 341.62, "Serum, plasma", "all", "all"),  # LabQAR Set_2 ID 1 -> High
        ("Acetoacetic acid", 0.20, "Serum, plasma", "all", "all"),  # ID 2 -> High (upper-only, no fabricated bound)
        ("Alcohol", 65.19, None, "all", "all"),                     # ID 11 -> should be Normal (lower-only)
        ("Leukocytes", 12.0, None, "all", "all"),                   # alias-resolved
        ("Hemoglobin", 11.1, None, "Female", "all"),
    ]
    for parameter, value, specimen, gender, age_group in cases:
        result = checker.evaluate(parameter, value, specimen, gender, age_group)
        print(f"{parameter:20} value={value:<8} -> {result['status']:8} "
              f"(via {result['retrieval_method']}, range=[{result['lower_bound']}, {result['upper_bound']}] {result['unit']})")
