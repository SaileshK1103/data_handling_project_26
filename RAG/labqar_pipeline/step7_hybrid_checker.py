from range_checker import RangeChecker

try:
    from step4_range_checker_rag import RangeCheckerRAG
    _LABQAR_AVAILABLE = True
except Exception as e: 
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
