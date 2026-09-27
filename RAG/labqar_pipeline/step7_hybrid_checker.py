
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
