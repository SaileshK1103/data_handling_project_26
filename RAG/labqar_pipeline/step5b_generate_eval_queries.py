"""
STEP 5b — Build a free-text evaluation set for the semantic retriever.

Why this file needs to exist at all
------------------------------------
Set_1's questions are template-generated: "For the lab test 'X' measuring
in 'Y' in Specimen 'Z' for 'G' and 'A' ...". step5_evaluate.py's structured
retrieval eval already scores ~100% on those -- but that's not a fair test
of embeddings/semantic search, because a templated string IS exactly what
exact-match string parsing is built for. Evaluating the vector DB on the
same templated text would just show "yes, embeddings can also solve the
easy case", which tells you nothing you didn't already know from step5.

The whole point of adding embeddings is the free-text case: a clinician's
paraphrase, or an extraction step that produced an imperfect parameter
name. So this script generates several paraphrased, non-templated queries
per LabQAR row (rule-based -- no LLM call needed, deterministic, reviewable)
and step5c evaluates semantic retrieval against THAT, which is the
realistic use case.
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
ART_DIR = Path(__file__).parent / "artifacts"

TEMPLATES = [
    "what is a normal {param} level{ctx}?",
    "is my {param} result within the reference range{ctx}?",
    "typical {param} reference interval{ctx}",
    "{param} normal range{ctx}",
    "how do I know if {param} is too high or too low{ctx}?",
]

# A small, hand-written "hard" set: genuine synonyms/informal phrasing with
# NO literal substring overlap with the LabQAR parameter string. The
# TEMPLATES above always embed the exact parameter name, so a lexical
# character-n-gram method can "win" on them for the wrong reason (it's
# matching the literal substring, not doing anything semantic) -- that's
# visible in step5c's results: the lexical baseline actually beats the
# TF-IDF+SVD fallback embedder on the templated set. This set is what
# actually tests whether retrieval understands meaning rather than string
# overlap, which is the case embeddings exist to handle.
HARD_QUERIES = [
    ("liver enzyme SGPT test result", "alanine aminotransferase (alt, sgpt)"),
    ("is my liver function enzyme okay", "alanine aminotransferase (alt, sgpt)"),
    ("blood sugar level test", "glucose"),
    ("fasting blood glucose normal range", "glucose"),
    ("Hgb level in blood", "hemoglobin"),
    ("Na+ electrolyte panel result", "sodium"),
    ("K+ electrolyte level", "potassium"),
    ("thyroid stimulating hormone test", "thyrotropin (thyroid--stimulating hormone, tsh)"),
    ("red blood cell count normal range", "red blood cell count"),
    ("white blood cell count normal range", "white blood cell count"),
    ("clotting cell count in blood", "platelet count"),
    ("iron storage protein blood test", "ferritin"),
    ("average blood sugar over three months", "hemoglobin a1c"),
    ("good cholesterol level (HDL)", "cholesterol, high-density lipoproteins (hdl)"),
    ("bad cholesterol level (LDL)", "cholesterol, low--density lipoproteins (ldl)"),
    ("kidney function creatinine test", "creatinine"),
]


def context_phrase(specimen, gender, age_group):
    bits = []
    if gender and gender not in ("any gender", "all"):
        bits.append(f"for a {gender.lower()}")
    if age_group and age_group not in ("any age group", "all"):
        bits.append(f"({age_group.lower()})")
    if specimen:
        bits.append(f"in {specimen.lower()}")
    return " " + " ".join(bits) if bits else ""


def build_eval_queries(n_per_row: int = 2, seed: int = 42, sample_size: int = None):
    """
    Returns a list of {ref_id, query, gold_parameter, gold metadata...} --
    one gold LabQAR row per query (the row the paraphrase was generated
    from), so retrieval@k can be scored against it.
    """
    registry = json.loads((ART_DIR / "step1_reference_registry.json").read_text(encoding="utf-8"))
    rng = random.Random(seed)

    rows = registry
    if sample_size:
        rows = rng.sample(registry, min(sample_size, len(registry)))

    queries = []
    for row in rows:
        templates = rng.sample(TEMPLATES, min(n_per_row, len(TEMPLATES)))
        ctx = context_phrase(row["specimen"], row["gender"], row["age_group"])
        for t in templates:
            q = t.format(param=row["parameter"], ctx=ctx)
            queries.append({
                "ref_id": row["ref_id"],
                "query": q,
                "gold_parameter": row["parameter"],
                "gold_parameter_norm": row["parameter"].strip().lower(),
                "gold_specimen": row["specimen"],
                "gold_gender": row["gender"],
                "gold_age_group": row["age_group"],
            })
    return queries


if __name__ == "__main__":
    queries = build_eval_queries(n_per_row=2, sample_size=100)
    out_path = ART_DIR / "step5b_eval_queries.json"
    out_path.write_text(json.dumps(queries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated {len(queries)} template-based eval queries from a sample of "
          f"{len(set(q['ref_id'] for q in queries))} LabQAR rows -> {out_path}")
    print("(These still contain the literal parameter name -- good for a sanity check, "
          "not a real test of semantic understanding. See HARD_QUERIES / step5b_hard_queries.json.)")

    hard = [{"query": q, "gold_parameter_norm": g} for q, g in HARD_QUERIES]
    hard_path = ART_DIR / "step5b_hard_queries.json"
    hard_path.write_text(json.dumps(hard, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(hard)} hand-written synonym/paraphrase queries (no literal "
          f"parameter-name overlap) -> {hard_path}")

    print("\nSample template queries:")
    for q in queries[:4]:
        print(f"  [{q['ref_id']:>3}] {q['query']!r}  (gold: {q['gold_parameter']})")
    print("\nSample hard queries:")
    for q, g in HARD_QUERIES[:4]:
        print(f"  {q!r}  (gold: {g})")
