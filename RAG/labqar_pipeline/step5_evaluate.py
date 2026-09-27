
import json
import re
from pathlib import Path

from step3_retriever import ReferenceRangeRetriever
from step4_range_checker_rag import RangeCheckerRAG

DATA_DIR = Path(__file__).parent / "data"
ART_DIR = Path(__file__).parent / "artifacts"

RE_PARAM = re.compile(r"lab test '([^']+)'")
RE_SPECIMEN = re.compile(r"in Specimen '([^']+)'")
RE_GENDER = re.compile(r"for '([^']+)' and")
RE_AGE = re.compile(r"and '([^']+)'")
RE_CATEGORY = re.compile(r"in the category '([^']+)'")
RE_UNIT = re.compile(r"measuring in '([^']+)'")
RE_CONDITION = re.compile(r"with the condition '([^']+)'")
RE_REFTYPE = re.compile(r"reference type [\u2018']([^\u2019']+)[\u2019']")
VAL_PATTERN = re.compile(r"is (?:'([0-9.]+)'|([0-9.]+))\.")


def parse_query_fields(question: str):
