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
- Do not repeat the raw numbers back verbatim in a robotic way; focus on
  what it means.
"""

USER_TEMPLATE = """Parameter: {parameter}
Measured value: {value} {unit}
Reference range: {lower} to {upper} {range_unit} (source: LabQAR, matched via {retrieval_method})
Verdict (already decided, do not re-derive): {status}

Write the interpretation_summary."""


def _template_fallback(parameter, value, unit, result) -> str:
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
    def __init__(self, model: str = "gemini-3.8-flash", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._client = None
        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                print("WARNING: google-genai not installed (pip install google-genai)")

    def explain(self, parameter: str, value: float, unit: str, result: dict) -> str:
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
            from google.genai import types
            response = self._client.models.generate_content(
                model=self.model,
                contents=user_msg,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                ),
            )
            return response.text.strip()
        except Exception as e:
            print(f"WARNING: LLM call failed ({e}); falling back to templated explanation.")
            return _template_fallback(parameter, value, unit, result)


if __name__ == "__main__":
    from step4_range_checker_rag import RangeCheckerRAG

    checker = RangeCheckerRAG()
    explainer = Explainer()

    cases = [
        ("ALT", 52.0, "U/L", "Serum", "all"),
        ("Hemoglobin", 111.0, "g/L", None, "Female"),
        ("Acetaminophen", 341.62, "μmol/L", "Serum, plasma", "all"),
    ]
    for parameter, value, unit, specimen, gender in cases:
        result = checker.evaluate(parameter, value, specimen=specimen, gender=gender)
        summary = explainer.explain(parameter, value, unit, result)
        print(f"\n{parameter} = {value} {unit} ({gender}) -> {result['status']}")
        print(f"  {summary}")
