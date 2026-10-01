"""
STEP 1 — Build the master reference-range registry from LabQAR Set_1 / Set_2.

Why this step exists
---------------------
Set_1.json and Set_2.json are QA *pairs* (Question/Answer), not a clean
knowledge base. Before anything can be embedded or retrieved, we need one
row per test/specimen/gender/age/category/condition combination with typed,
numeric fields. That row is what the RAG corpus will be built from in Step 2.

Key findings baked into this script (verified directly against your files,
not assumed):
1. Set_1['Answer'] is a *string* ("70-200", "<0.1", "21.7"). Set_2['reference_range']
   already stores the same bounds as numeric lower_bound/upper_bound. We use
   Set_2's numeric field as the source of truth for numbers, and use the
   shared Question text (identical structure in both files) as the source of
   truth for metadata (parameter, specimen, gender, age_group, category,
   condition, unit).
2. Set_2['Answer'] (the High/Normal/Low letter) is NOT always consistent with
   Set_2['reference_range']. Recomputing the label from the bounds and the
   value stated in the question finds 20/550 (3.6%) mismatches — this script
   reproduces that check and writes out the corrected labels
   (see step1_set2_corrected.json), which Step 5 uses instead of the raw
   'Answer' field as evaluation ground truth.
3. One-sided ranges ("<0.1" / lower_bound is null, or ">17.4" / upper_bound is
   null) are kept as genuinely open-ended. We do NOT invent a synthetic
   opposite bound (e.g. upper = lower * 2) the way the original generation
   code did — that fabrication is exactly what produces some of the 20
   mismatches above. An open bound means "no upper/lower limit is defined";
   the checker in Step 4 treats it as such.
"""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "artifacts"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Parse the structured fields out of the (identical) Question text shared
#    by Set_1 and Set_2. Independent single-quote-delimited regexes are used
#    instead of one long pattern, because the age/category/condition clauses
#    appear in varying order and are otherwise easy to mis-capture.
# ---------------------------------------------------------------------------

RE_PARAM = re.compile(r"lab test '([^']+)'")
RE_UNIT = re.compile(r"measuring in '([^']+)'")
RE_SPECIMEN = re.compile(r"in Specimen '([^']+)'")
RE_GENDER = re.compile(r"for '([^']+)' and")
RE_AGE = re.compile(r"and '([^']+)'")
RE_CATEGORY = re.compile(r"in the category '([^']+)'")
RE_CONDITION = re.compile(r"with the condition '([^']+)'")
# LabQAR renders this clause with *curly* quotes ('reference type '..'')
# even though every other clause uses straight quotes -- a separate regex
# is needed or this silently never matches. Currently only the "Cholesterol"
# / Total-category rows (Desirable / Borderline high / High) use this field,
# but the pattern is written generally in case future rows add more.
RE_REFTYPE = re.compile(r"reference type [\u2018']([^\u2019']+)[\u2019']")


def parse_question(q: str) -> dict:
    def grab(pattern):
        m = pattern.search(q)
        return m.group(1) if m else None

    return {
        "parameter": grab(RE_PARAM),
        "unit": grab(RE_UNIT),
        "specimen": grab(RE_SPECIMEN),
        "gender": grab(RE_GENDER),
        "age_group": grab(RE_AGE),
        "category": grab(RE_CATEGORY),
        "condition": grab(RE_CONDITION),
        "reference_type": grab(RE_REFTYPE),
    }


def range_type(lower, upper):
    if lower is not None and upper is not None:
        return "two_sided"
    if lower is not None and upper is None:
        return "lower_only"  # e.g. ">17.4" -- only a floor is defined
    if lower is None and upper is not None:
        return "upper_only"  # e.g. "<0.1"  -- only a ceiling is defined
    return "undefined"


def classify(value, lower, upper):
    """Ground-truth classifier used both to build the registry and to
    recompute Set_2 labels. No fabricated bounds -- an undefined side of
    the range simply can never trigger that flag."""
    if lower is not None and value < lower:
        return "Low"
    if upper is not None and value > upper:
        return "High"
    return "Normal"


def main():
    set1 = json.loads((DATA_DIR / "Set_1.json").read_text(encoding="utf-8"))
    set2 = {r["ID"]: r for r in json.loads((DATA_DIR / "Set_2.json").read_text(encoding="utf-8"))}

    registry_rows = []
    parse_failures = 0

    for r1 in set1:
        rid = r1["ID"]
        r2 = set2.get(rid)
        if r2 is None:
            continue

        meta = parse_question(r1["Question"])
        if any(meta[k] is None for k in ("parameter", "unit", "specimen", "gender", "age_group")):
            parse_failures += 1
            continue

        rr = r2["reference_range"]
        lower = rr.get("lower_bound")
        upper = rr.get("upper_bound")
        unit = rr.get("unit", meta["unit"])

        registry_rows.append({
            "ref_id": rid,
            "parameter": meta["parameter"],
            "category": meta["category"],
            "condition": meta["condition"],
            "reference_type": meta["reference_type"],
            "specimen": meta["specimen"],
            "gender": "all" if meta["gender"] == "any gender" else meta["gender"],
            "age_group": "all" if meta["age_group"] == "any age group" else meta["age_group"],
            "unit": unit,
            "lower_bound": lower,
            "upper_bound": upper,
            "range_type": range_type(lower, upper),
            "source": "LabQAR_Set1_Set2",
        })

    print(f"Registry rows built: {len(registry_rows)} (parse failures: {parse_failures})")

    with open(OUT_DIR / "step1_reference_registry.json", "w", encoding="utf-8") as f:
        json.dump(registry_rows, f, indent=2, ensure_ascii=False)

    # Also a flat CSV, for parity with your existing reference_ranges.csv shape
    import csv
    with open(OUT_DIR / "step1_reference_registry.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(registry_rows[0].keys()))
        writer.writeheader()
        writer.writerows(registry_rows)

    # -----------------------------------------------------------------
    # 2. Recompute Set_2 gold labels from reference_range + the value
    #    stated in the question, and diff against the raw Answer letter.
    # -----------------------------------------------------------------
    val_pattern = re.compile(r"is (?:'([0-9.]+)'|([0-9.]+))\.")
    corrected, mismatches = [], []

    for r2 in set2.values():
        m = val_pattern.search(r2["Question"])
        if not m:
            continue
        value = float(m.group(1) or m.group(2))
        rr = r2["reference_range"]
        true_label = classify(value, rr.get("lower_bound"), rr.get("upper_bound"))

        choices = {}
        for line in r2["Choices"].strip().split("\n"):
            letter, label = line.split(":", 1)
            choices[letter.strip()] = label.strip()
        raw_label = choices.get(r2["Answer"])

        row = {
            "ID": r2["ID"],
            "value": value,
            "lower_bound": rr.get("lower_bound"),
            "upper_bound": rr.get("upper_bound"),
            "raw_label": raw_label,
            "corrected_label": true_label,
            "was_mismatch": raw_label != true_label,
        }
        corrected.append(row)
        if row["was_mismatch"]:
            mismatches.append(row)

    with open(OUT_DIR / "step1_set2_corrected_labels.json", "w", encoding="utf-8") as f:
        json.dump(corrected, f, indent=2)

    print(f"\nSet_2 label audit: {len(mismatches)}/{len(corrected)} "
          f"({100*len(mismatches)/len(corrected):.1f}%) raw 'Answer' labels "
          f"disagree with reference_range-derived ground truth.")
    for row in mismatches[:5]:
        print(f"  ID {row['ID']}: value={row['value']} range=[{row['lower_bound']},{row['upper_bound']}] "
              f"raw='{row['raw_label']}' corrected='{row['corrected_label']}'")
    print(f"  ... full list in step1_set2_corrected_labels.json")

    print(f"\nWrote:")
    print(f"  artifacts/step1_reference_registry.json/.csv  ({len(registry_rows)} rows)")
    print(f"  artifacts/step1_set2_corrected_labels.json     ({len(corrected)} rows)")


if __name__ == "__main__":
    main()
