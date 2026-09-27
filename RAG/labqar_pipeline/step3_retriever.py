
import json
import re
from pathlib import Path
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ART_DIR = Path(__file__).parent / "artifacts"

PARAMETER_ALIASES = {
    "leukocytes": "white blood cell count",
    "platelets": "platelet count",
    "erythrocytes": "red blood cell count",
    "hba1c": "hemoglobin a1c",
    "alt": "alanine aminotransferase (alt, sgpt)",
    "urea nitrogen": "urea nitrogen (bun)",
    "carbon dioxide": "carbon dioxide",
    "tsh": "thyrotropin (thyroid--stimulating hormone, tsh)",
}


@dataclass
class RetrievalResult:
    doc: dict
    method: str 
    score: float = 1.0


class ReferenceRangeRetriever:
    def __init__(self, corpus_path: Path = ART_DIR / "step2_corpus.jsonl"):
        self.docs = [json.loads(line) for line in open(corpus_path, encoding="utf-8")]

        self.by_key = {}
        for doc in self.docs:
            m = doc["metadata"]
            key = (m["parameter_norm"], m["specimen"], m["gender"], m["age_group"])
            self.by_key.setdefault(key, []).append(doc)

        self.by_param = {}
        for doc in self.docs:
            self.by_param.setdefault(doc["metadata"]["parameter_norm"], []).append(doc)

        self.param_names = sorted(self.by_param.keys())
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self._param_matrix = self._vectorizer.fit_transform(self.param_names)

    @staticmethod
    def _norm(s):
        return re.sub(r"\s+", " ", s.strip().lower()) if s else None

    def _fuzzy_param_match(self, parameter: str, top_k: int = 1):
        query_vec = self._vectorizer.transform([self._norm(parameter)])
        sims = cosine_similarity(query_vec, self._param_matrix)[0]
        ranked = sims.argsort()[::-1][:top_k]
        return [(self.param_names[i], float(sims[i])) for i in ranked]

    def retrieve(self, parameter: str, specimen: str = None, gender: str = "all",
                 age_group: str = "all", category=None, unit: str = None,
                 condition=None, reference_type=None, top_k: int = 1) -> list:
        p_norm = self._norm(parameter)
        gender = gender if gender in ("Male", "Female") else "all"
        age_group = age_group if age_group in ("Adult", "Child", "Infant") else "all"
        method = "exact"

        if p_norm not in self.by_param and p_norm in PARAMETER_ALIASES:
            p_norm = PARAMETER_ALIASES[p_norm]
            method = "exact_via_alias"

        candidates = self.by_param.get(p_norm)
        if not candidates:
            fuzzy_matches = self._fuzzy_param_match(parameter, top_k=top_k)
            results = []
            for name, score in fuzzy_matches:
                for doc in self.by_param[name][:1]:
                    results.append(RetrievalResult(doc=doc, method="fuzzy", score=score))
            return results

        def filter_by(cands, **kwargs):
            return [d for d in cands if all(d["metadata"][k] == v for k, v in kwargs.items())]

        pool = candidates
        if specimen:
            by_specimen = filter_by(pool, specimen=specimen)
            if by_specimen:
                pool = by_specimen

        by_category = filter_by(pool, category=category)
        if by_category:
            pool = by_category

        if unit:
            by_unit = filter_by(pool, unit=unit)
            if by_unit:
                pool = by_unit

        by_condition = filter_by(pool, condition=condition)
        if by_condition:
            pool = by_condition

        by_reftype = filter_by(pool, reference_type=reference_type)
        if by_reftype:
            pool = by_reftype

        by_gender = filter_by(pool, gender=gender)
        if by_gender:
            pool = by_gender
        else:
            method = "exact_fallback_gender" if method == "exact" else method
            fallback = filter_by(pool, gender="all")
            pool = fallback if fallback else pool

        by_age = filter_by(pool, age_group=age_group)
        if by_age:
            pool = by_age
        else:
            method = "exact_fallback_age" if method == "exact" else method
            fallback = filter_by(pool, age_group="all")
            pool = fallback if fallback else pool

        return [RetrievalResult(doc=d, method=method, score=1.0) for d in pool[:top_k]]


if __name__ == "__main__":
    retriever = ReferenceRangeRetriever()

    tests = [
        ("Acetaminophen", "Serum, plasma", "all", "all"),
        ("ALT", None, "all", "all"),            # resolved via alias table
        ("Leukocytes", None, "all", "all"),     # resolved via alias table
        ("Hemoglobin", None, "Female", "all"),
        ("Erythrocytes", None, "Male", "all"),  # resolved via alias table
    ]
    for parameter, specimen, gender, age_group in tests:
        results = retriever.retrieve(parameter, specimen, gender, age_group)
        print(f"\nQuery: parameter={parameter!r} specimen={specimen!r} gender={gender!r} age_group={age_group!r}")
        for r in results:
            m = r.doc["metadata"]
            print(f"  [{r.method}, score={r.score:.2f}] {m['parameter']} "
                  f"[{m['lower_bound']}, {m['upper_bound']}] {m['unit']} (gender={m['gender']}, age={m['age_group']})")
