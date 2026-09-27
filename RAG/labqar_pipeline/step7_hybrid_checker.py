"""
STEP 7 — Wire the LabQAR RAG pipeline into your existing range_checker.py /
dataset_builder.py, WITHOUT replacing either.

This follows section 8 of your plan exactly:

    Primary truth layer      -> your reference_ranges.csv (range_checker.py, unchanged)
    Enrichment / secondary   -> LabQAR (steps 1-4 here)
    Evaluation               -> LabQAR (step 5)
    LLM                      -> explanation only (step 6)

Why not just replace RangeChecker with RangeCheckerRAG?
---------------------------------------------------------
Your CSV is the one you control, validated for your own 20-parameter
extraction vocabulary (dataset_builder.py's CODE_TO_PARAM), and -- per the
plan -- more trustworthy than LabQAR for the ~10 parameters LabQAR itself
can't disambiguate from text alone (see step3_retriever.py's docstring).
Replacing it would trade a smaller, known-good source for a larger one with
its own blind spots.

What HybridRangeChecker actually does
--------------------------------------
1. Always evaluates against your CSV first (range_checker.RangeChecker) --
   that result is what gets used for downstream Low/Normal/High flags, same
   as today. dataset_builder.py's behavior does not change.
2. In parallel, evaluates against LabQAR (RangeCheckerRAG) purely for
   logging/audit -- did the two sources agree? This is exactly the
   "corroborating source" framing from section 2A of the plan, just
   implemented as a live check instead of a static merged JSON.
3. Disagreements are collected, not surfaced as errors -- a disagreement
   might mean your CSV needs a look, or it might be one of LabQAR's own
   known-ambiguous parameters (again, see step3's docstring) and your CSV
   is right to override it. This script doesn't decide which; it just
   makes the discrepancy visible instead of silent.

Drop-in for dataset_builder.py
-------------------------------
dataset_builder.py currently does:

    from range_checker import RangeChecker
    checker = RangeChecker()
    ...
    status = checker.evaluate(param, val, gender=gender)

To get audit logging with NO change to the resulting dataset, swap only the
import and constructor:

    from step7_hybrid_checker import HybridRangeChecker
    checker = HybridRangeChecker()
    ...
    status = checker.evaluate(param, val, gender=gender)   # unchanged call site

Then, after build_datasets() finishes, call checker.print_audit_report() to
see where LabQAR disagreed with your CSV -- a free correctness signal on
reference_ranges.csv you didn't have before, at zero cost to the existing
pipeline behavior.
"""

from range_checker import RangeChecker

try:
    from step4_range_checker_rag import RangeCheckerRAG
    _LABQAR_AVAILABLE = True
except Exception as e:  # missing artifacts/step2_corpus.jsonl, sklearn not installed, etc.
    _LABQAR_AVAILABLE = False
    _LABQAR_IMPORT_ERROR = e


class HybridRangeChecker:
    def __init__(self, ref_file: str = "config/reference_ranges.csv", enable_audit: bool = True):
        self.primary = RangeChecker(ref_file=ref_file)
        self.enable_audit = enable_audit and _LABQAR_AVAILABLE
        self.secondary = RangeCheckerRAG() if self.enable_audit else None
        if enable_audit and not _LABQAR_AVAILABLE:
            print(f"NOTE: LabQAR audit layer disabled (couldn't load it: {_LABQAR_IMPORT_ERROR}). "
                  f"Falling back to CSV-only, same as range_checker.py alone. "
                  f"Run steps 1-2 to build artifacts/step2_corpus.jsonl if you want the audit trail.")
        self.disagreements = []
        self.n_checked = 0
        self.n_labqar_no_match = 0

    def evaluate(self, parameter: str, value: float, gender: str = "all", age: int = 40) -> str:
        """Same signature as range_checker.RangeChecker.evaluate() -- this
        IS the primary decision, unchanged. The LabQAR call below never
        affects the return value."""
        primary_status = self.primary.evaluate(parameter, value, gender=gender, age=age)
        self.n_checked += 1

        if self.enable_audit:
            secondary = self.secondary.evaluate(parameter, value, gender=gender)
            if secondary["retrieval_method"] is None:
                self.n_labqar_no_match += 1
            elif secondary["status"] != primary_status:
                self.disagreements.append({
                    "parameter": parameter,
                    "value": value,
                    "gender": gender,
                    "csv_status": primary_status,
                    "labqar_status": secondary["status"],
                    "labqar_range": (secondary["lower_bound"], secondary["upper_bound"], secondary["unit"]),
                    "labqar_matched_as": secondary["matched_parameter"],
                })

        return primary_status

    def print_audit_report(self):
        print("=" * 70)
        print("HYBRID CHECKER AUDIT REPORT (CSV=primary/authoritative vs LabQAR=secondary)")
        print("=" * 70)
        print(f"Total evaluations: {self.n_checked}")
        if not self.enable_audit:
            print("(LabQAR audit layer was disabled for this run.)")
            return
        print(f"LabQAR had no match at all: {self.n_labqar_no_match}")
        print(f"Disagreements: {len(self.disagreements)}")
        if self.disagreements:
            print("\nSample disagreements (check whether reference_ranges.csv needs a look, "
                  "or whether this is one of LabQAR's known-ambiguous parameters -- "
                  "see step3_retriever.py docstring):")
            for d in self.disagreements[:15]:
                print(f"  {d['parameter']:20} value={d['value']:<8} gender={d['gender']:8} "
                      f"CSV={d['csv_status']:8} LabQAR={d['labqar_status']:8} "
                      f"(LabQAR range={d['labqar_range']}, matched as '{d['labqar_matched_as']}')")


if __name__ == "__main__":
    # Minimal smoke test using this repo's own artifacts. Requires
    # config/reference_ranges.csv to exist -- point ref_file at yours.
    import os
    if not os.path.exists("config/reference_ranges.csv"):
        print("No config/reference_ranges.csv found in this environment -- "
              "this is a dry-run showing the LabQAR side only.")
        if _LABQAR_AVAILABLE:
            rag = RangeCheckerRAG()
            for p, v in [("ALT", 52.0), ("Hemoglobin", 111.0)]:
                r = rag.evaluate(p, v, gender="Female")
                print(f"  {p} = {v} -> {r['status']} (LabQAR only, no CSV to compare against)")
    else:
        checker = HybridRangeChecker()
        tests = [
            ("Glucose", 168.0, "all"),
            ("Hemoglobin", 11.1, "F"),
            ("ALT", 52.0, "all"),
        ]
        for param, value, gender in tests:
            status = checker.evaluate(param, value, gender=gender)
            print(f"{param:12} = {value:<8} ({gender}) -> {status}")
        checker.print_audit_report()
