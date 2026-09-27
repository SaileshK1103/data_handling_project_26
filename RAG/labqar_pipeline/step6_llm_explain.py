"""
STEP 6 — LLM explanation layer.

This is the ONLY step in the pipeline where an LLM is allowed to touch the
final answer, and even then it never touches the Low/Normal/High decision
itself -- that was already decided deterministically in Step 4. This script
turns a RangeCheckerRAG.evaluate() result into a short, human-readable
interpretation, the same separation of concerns your original plan called
for:

    RAG (Step 3)        -> finds the reference range + context
    RangeChecker (Step 4) -> pure arithmetic decision, no LLM involved
    LLM (here)           -> turns the decision into a readable explanation

Why this separation matters for evaluation: Step 5 showed the deterministic
checker is right ~98% of the time against corrected LabQAR labels. If you
let an LLM re-decide High/Normal/Low from the number, you lose that
auditability -- you can no longer tell whether a wrong answer came from bad
retrieval, a bad LLM call, or genuine ambiguity in the source data. Keeping
the LLM's job to "explain a verdict I'm given" instead of "decide a verdict"
means a bad LLM completion can produce an unhelpful sentence, never a wrong
clinical flag.

Usage
-----
    from step4_range_checker_rag import RangeCheckerRAG
    from step6_llm_explain import Explainer

    checker = RangeCheckerRAG()
    explainer = Explainer()  # reads GOOGLE_API_KEY from the environment

    result = checker.evaluate("ALT", 52, specimen="Serum")
    summary = explainer.explain(parameter="ALT", value=52, unit="U/L", result=result)
    print(summary)

Runs fine without an API key too -- explain() falls back to a deterministic
templated sentence (no network call) so the rest of the pipeline (Step 7,
dataset_builder.py) doesn't hard-fail in an environment with no key configured,
e.g. when you're just testing the plumbing.
"""

import os

SYSTEM_PROMPT = """You are a clinical lab-report explanation assistant.
You are given a single lab parameter, its measured value, and a reference
range that a separate deterministic system has ALREADY compared it against
-- the Low/Normal/High verdict is not yours to change, only to explain.

Rules:
- Never contradict, hedge on, or restate the verdict as if you were
  independently deciding it. Treat it as given fact.
- One to three sentences. Plain language a patient could understand.
- If context (specimen, gender, age group, condition) narrows the range,
  mention it only if it's relevant to interpreting the number correctly.
- Do not invent a diagnosis, cause, or treatment recommendation -- describe
  what the parameter measures and what the flag means in general terms.
- Do not repeat the raw numbers back verbatim in a robotic "X is between Y
  and Z" way if the caller already displays that separately; focus on
  what it means.
"""

USER_TEMPLATE = """Parameter: {parameter}
Measured value: {value} {unit}
Reference range: {lower} to {upper} {range_unit} (source: LabQAR, matched via {retrieval_method})
Verdict (already decided, do not re-derive): {status}

Write the interpretation_summary."""


def _template_fallback(parameter, value, unit, result) -> str:
    """No-API-key / offline path. Deterministic, not an LLM call."""
    status = result["status"]
    lower, upper = result.get("lower_bound"), result.get("upper_bound")
    if status == "Normal":
        return (f"{parameter} was {value} {unit}, which falls within the "
                f"expected reference range for this test.")
    if status == "High":
        bound = f" (reference upper limit: {upper} {result.get('unit') or unit})" if upper is not None else ""
        return f"{parameter} was {value} {unit}, above the expected reference range{bound}."
    if status == "Low":
        bound = f" (reference lower limit: {lower} {result.get('unit') or unit})" if lower is not None else ""
        return f"{parameter} was {value} {unit}, below the expected reference range{bound}."
    return f"{parameter} was {value} {unit}. No reference range was found to compare it against."


class Explainer:
    def __init__(self, model: str = "gemini-3.1-pro-preview", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._client = None
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(
                    model_name=self.model,
                    system_instruction=SYSTEM_PROMPT,
                )
            except ImportError:
                print("WARNING: google-generativeai package not installed "
                      "(pip install google-generativeai) "
                      "-- falling back to templated explanations. "
                      "Run: pip install google-generativeai")

    def explain(self, parameter: str, value: float, unit: str, result: dict) -> str:
        """
        result is exactly what RangeCheckerRAG.evaluate() returns (Step 4):
        {status, matched_parameter, retrieval_method, lower_bound, upper_bound, unit}
        """
        if self._client is None:
            return _template_fallback(parameter, value, unit, result)

        user_msg = USER_TEMPLATE.format(
            parameter=result.get("matched_parameter") or parameter,
            value=value,
            unit=unit,
            lower=result.get("lower_bound"),
            upper=result.get("upper_bound"),
            range_unit=result.get("unit") or unit,
            retrieval_method=result.get("retrieval_method"),
            status=result["status"],
        )
        try:
            resp = self._client.generate_content(user_msg)
            return resp.text.strip()
        except Exception as e:  # network error, rate limit, bad key, etc.
            print(f"WARNING: LLM call failed ({e}); falling back to templated explanation.")
            return _template_fallback(parameter, value, unit, result)


if __name__ == "__main__":
    from step4_range_checker_rag import RangeCheckerRAG

    checker = RangeCheckerRAG()
    explainer = Explainer()  # will use the template fallback unless GOOGLE_API_KEY is set

    cases = [
        ("ALT", 52.0, "U/L", "Serum", "all"),
        ("Hemoglobin", 111.0, "g/L", None, "Female"),  # note: g/L, not g/dL -- matches the registry's unit
        ("Acetaminophen", 341.62, "μmol/L", "Serum, plasma", "all"),
    ]
    for parameter, value, unit, specimen, gender in cases:
        result = checker.evaluate(parameter, value, specimen=specimen, gender=gender)
        summary = explainer.explain(parameter, value, unit, result)
        print(f"\n{parameter} = {value} {unit} ({gender}) -> {result['status']}")
        print(f"  {summary}")
