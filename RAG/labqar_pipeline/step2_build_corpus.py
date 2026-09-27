"""
STEP 2 — Turn the registry into RAG documents.

A RAG "document" here is one retrievable unit: a single test's reference
range in a single context (specimen + gender + age_group + category +
condition), plus a natural-language rendering of that same information for
lexical/semantic matching. Keeping structured fields *alongside* the text
(rather than text-only) is what lets Step 3's retriever do exact filtering
first and fall back to fuzzy text search only when needed -- important here
because LabQAR's contexts (gender, specimen, age) change what the "correct"
answer even is, so a text-only nearest-neighbor match could silently return
the wrong context.
"""

import json
from pathlib import Path

ART_DIR = Path(__file__).parent / "artifacts"


def bound_phrase(lower, upper, unit):
    if lower is not None and upper is not None:
        return f"between {lower} and {upper} {unit}"
    if lower is not None:
        return f"greater than {lower} {unit}"
    if upper is not None:
        return f"less than {upper} {unit}"
    return "undefined"


def make_doc(row: dict) -> dict:
    context_bits = [f"specimen: {row['specimen']}", f"gender: {row['gender']}", f"age group: {row['age_group']}"]
    if row["category"]:
        context_bits.append(f"category: {row['category']}")
    if row["condition"]:
        context_bits.append(f"condition: {row['condition']}")
    if row.get("reference_type"):
        context_bits.append(f"reference type: {row['reference_type']}")
    context_str = "; ".join(context_bits)

    text = (
        f"Reference range for {row['parameter']} ({context_str}): "
        f"{bound_phrase(row['lower_bound'], row['upper_bound'], row['unit'])}."
    )

    return {
        "doc_id": f"labqar_{row['ref_id']}",
        "text": text,
        "metadata": {
            "parameter": row["parameter"],
            "parameter_norm": row["parameter"].strip().lower(),
            "category": row["category"],
            "condition": row["condition"],
            "reference_type": row.get("reference_type"),
            "specimen": row["specimen"],
            "gender": row["gender"],
            "age_group": row["age_group"],
            "unit": row["unit"],
            "lower_bound": row["lower_bound"],
            "upper_bound": row["upper_bound"],
            "range_type": row["range_type"],
            "source": row["source"],
        },
    }


def main():
    registry = json.loads((ART_DIR / "step1_reference_registry.json").read_text(encoding="utf-8"))
    corpus = [make_doc(row) for row in registry]

    out_path = ART_DIR / "step2_corpus.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Wrote {len(corpus)} documents to {out_path}")
    print("\nSample document:")
    print(json.dumps(corpus[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
