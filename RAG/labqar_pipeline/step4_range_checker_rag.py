
from step3_retriever import ReferenceRangeRetriever


class RangeCheckerRAG:
    def __init__(self, retriever: ReferenceRangeRetriever = None):
        self.retriever = retriever or ReferenceRangeRetriever()

    def evaluate(self, parameter: str, value: float, specimen: str = None,
                 gender: str = "all", age_group: str = "all", category=None, unit: str = None,
                 condition=None, reference_type=None) -> dict:
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
        ("Acetaminophen", 341.62, "Serum, plasma", "all", "all"), 
        ("Acetoacetic acid", 0.20, "Serum, plasma", "all", "all"), 
        ("Alcohol", 65.19, None, "all", "all"),                    
        ("Leukocytes", 12.0, None, "all", "all"),                   # alias-resolved
        ("Hemoglobin", 11.1, None, "Female", "all"),
    ]
    for parameter, value, specimen, gender, age_group in cases:
        result = checker.evaluate(parameter, value, specimen, gender, age_group)
        print(f"{parameter:20} value={value:<8} -> {result['status']:8} "
              f"(via {result['retrieval_method']}, range=[{result['lower_bound']}, {result['upper_bound']}] {result['unit']})")
