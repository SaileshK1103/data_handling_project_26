"""
STEP 3 — Retriever.

Design decision: this corpus is small (550 rows) and highly structured
(every row already has clean parameter/specimen/gender/age_group fields).
For that situation, a semantic embedding model is *not* the right default
retrieval mechanism -- it would add latency, a dependency on downloading
model weights, and a real risk of confidently returning the wrong gender-
or specimen-specific row for a near-duplicate test name. Instead:

  1. EXACT match on (parameter_norm, specimen, gender, age_group[, category,
     condition]) when the caller has all of that metadata -- this is what
     the fine-tuned extraction model in your pipeline is meant to produce.
  2. Graceful fallback: if there's no row for this exact gender/age, fall
     back to gender="all" / age_group="all", the same "any gender / any age
     group" rows LabQAR itself uses as defaults.
  3. Fuzzy fallback (TF-IDF character n-grams + cosine similarity) ONLY
     when step 1/2 return nothing -- e.g. the extractor said "ALT" but the
     registry has "Alanine aminotransferase (ALT, SGPT)". This uses
     scikit-learn, already in your environment, no model download needed.

If you later swap in a real embedding model (sentence-transformers, Voyage,
OpenAI, etc.) for a larger/messier corpus, keep the exact-match path first --
it's strictly more reliable whenever structured metadata is available, and
free.
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ART_DIR = Path(__file__).parent / "artifacts"

# LabQAR sometimes stores the SAME analyte under more than one parameter
# name (e.g. as a row of the "Complete blood count (CBC)" panel AND as its
# own standalone parameter, or split by differential "category" such as
# Neutrophils/Lymphocytes under a WBC-count parameter). Your pipeline's
# extraction vocabulary (dataset_builder.py's CODE_TO_PARAM) uses short
# clinical abbreviations that don't always literally match LabQAR's
# parameter string. This alias table was built by inspecting the registry
# (see step3b_explore_aliases.py) -- extend it as you add more parameters.
PARAMETER_ALIASES = {
    "leukocytes": "white blood cell count",
    "platelets": "platelet count",
    "erythrocytes": "red blood cell count",
    # MCV is not present in LabQAR (only MCH/MCHC exist); no alias needed.
    "hba1c": "hemoglobin a1c",
    "alt": "alanine aminotransferase (alt, sgpt)",
    "urea nitrogen": "urea nitrogen (bun)",
    "carbon dioxide": "carbon dioxide",
    "tsh": "thyrotropin (thyroid--stimulating hormone, tsh)",
}


@dataclass
class RetrievalResult:
    doc: dict
    method: str  # "exact" | "exact_fallback_gender" | "exact_fallback_age" | "exact_fallback_all" | "fuzzy"
    score: float = 1.0


class ReferenceRangeRetriever:
    def __init__(self, corpus_path: Path = ART_DIR / "step2_corpus.jsonl"):
        self.docs = [json.loads(line) for line in open(corpus_path, encoding="utf-8")]

        # Index by (parameter_norm, specimen, gender, age_group) for O(1) exact lookup
        self.by_key = {}
        for doc in self.docs:
            m = doc["metadata"]
            key = (m["parameter_norm"], m["specimen"], m["gender"], m["age_group"])
            self.by_key.setdefault(key, []).append(doc)

        # Also index by parameter name alone, for gender/age fallback scans
        self.by_param = {}
        for doc in self.docs:
            self.by_param.setdefault(doc["metadata"]["parameter_norm"], []).append(doc)

        # TF-IDF over parameter names (character n-grams handle abbreviations
        # and partial matches, e.g. "ALT" inside "Alanine aminotransferase (ALT, SGPT)")
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
        """
        Retrieve the best-matching reference-range document(s) for a query.

        parameter : test name as produced by your extraction step
        specimen  : e.g. "Serum, plasma" (optional -- if omitted, any specimen matches)
        gender    : "all" | "Male" | "Female"
        age_group : "all" | "Adult" | "Child" | "Infant"
        category  : sub-analyte within a panel (e.g. "Neutrophils" under a
                    WBC-count parameter). Defaults to None -- the top-level
                    single-value reading, not a differential/panel sub-row.
        unit      : disambiguates SI-vs-conventional unit pairs for the same
                    analyte (e.g. Vitamin D reported in both nmol/L and
                    pmol/L with different numeric bounds). Pass the unit
                    your extraction step read off the lab report.
        condition : disambiguates cycle-phase/state-dependent hormone rows
                    (e.g. "Follicular phase", "Postmenopausal", "Luteal
                    phase"). Defaults to None -- the phase-independent row,
                    when one exists. This is what fixed most of the ~20
                    reproductive-hormone mismatches in Set_2 (FSH, LH,
                    Estradiol, Estrone, Progesterone, Inhibin A, 17-OHP,
                    Pregnanediol, C-telopeptide): the question text already
                    states the phase, step5 just wasn't parsing/forwarding it.
        reference_type : disambiguates clinical risk-tier rows for the same
                    test (currently only "Cholesterol"/Total: "Desirable" /
                    "Borderline high" / "High"). Defaults to None.

        KNOWN LIMITATION (see step5_evaluate.py output / README): after
        specimen+category+unit+condition+reference_type filtering, ~10 of
        LabQAR's 550 rows *still* resolve to more than one candidate --
        HDL/LDL risk bands (LabQAR gives them no reference_type field, only
        Cholesterol Total has one), Arsenic toxicity tiers, smoker-status
        pairs (Carboxyhemoglobin, CEA), Cortisol AM/PM, and Methotrexate
        post-dose timing. None of these are recoverable from the question
        text LabQAR ships -- there's no field left to parse. For those
        specific tests, prefer your own reference_ranges.csv over this
        corpus, or extend LabQAR's own schema with the missing context
        before trusting it.
        """
        p_norm = self._norm(parameter)
        gender = gender if gender in ("Male", "Female") else "all"
        age_group = age_group if age_group in ("Adult", "Child", "Infant") else "all"
        method = "exact"

        # 0. Alias resolution: your pipeline's short name -> LabQAR's parameter string
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

        # Prefer the plain top-level reading (category=None) unless a
        # sub-category was explicitly requested -- avoids accidentally
        # returning a CBC differential row (Neutrophils, Bands, ...) when
        # the caller just asked for the panel's headline parameter.
        by_category = filter_by(pool, category=category)
        if by_category:
            pool = by_category

        if unit:
            by_unit = filter_by(pool, unit=unit)
            if by_unit:
                pool = by_unit

        # Reproductive-hormone tests (FSH, LH, Estradiol, Progesterone, ...)
        # have menstrual-cycle-phase-dependent ranges; default to the
        # phase-independent row (condition=None) unless a phase is given.
        by_condition = filter_by(pool, condition=condition)
        if by_condition:
            pool = by_condition

        # Risk-tier rows (currently just Cholesterol Total: Desirable /
        # Borderline high / High). Same pattern as condition above.
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
