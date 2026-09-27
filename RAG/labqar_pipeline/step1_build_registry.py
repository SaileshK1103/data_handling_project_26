
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "artifacts"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

RE_PARAM = re.compile(r"lab test '([^']+)'")
RE_UNIT = re.compile(r"measuring in '([^']+)'")
RE_SPECIMEN = re.compile(r"in Specimen '([^']+)'")
RE_GENDER = re.compile(r"for '([^']+)' and")
RE_AGE = re.compile(r"and '([^']+)'")
RE_CATEGORY = re.compile(r"in the category '([^']+)'")
RE_CONDITION = re.compile(r"with the condition '([^']+)'")
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
        return "lower_only" 
    if lower is None and upper is not None:
        return "upper_only" 
    return "undefined"


def classify(value, lower, upper):
