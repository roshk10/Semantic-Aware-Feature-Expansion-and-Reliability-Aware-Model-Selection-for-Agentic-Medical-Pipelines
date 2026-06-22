import json
import os
import re

DICTIONARY_PATH = "data/models/clinical_abbreviations.json"

KNOWN_STANDARD_SHORT = {
    "age",
    "sex",
    "bmi",
    "glucose",
    "insulin",
    "outcome",
    "target",
    "slope",
    "ca",
    "thal"
}


def load_dictionary():

    if not os.path.exists(DICTIONARY_PATH):
        print("[Agent 2] Dictionary not found.")
        return {}

    with open(DICTIONARY_PATH, "r") as f:
        return json.load(f)


def classify_header(header):

    header = header.strip()
    lower = header.lower()

    if lower in KNOWN_STANDARD_SHORT:
        return "STANDARD"

    if len(header) <= 6:
        return "ABBREVIATED"

    if "_" in header:
        return "ABBREVIATED"

    if re.match(r'^[A-Za-z][a-zA-Z]+$', header):
        return "STANDARD"

    return "AMBIGUOUS"


def expand_header(header, dictionary):

    key = header.lower().strip()

    key_clean = (
        key.replace("_", "")
           .replace("-", "")
           .replace(" ", "")
    )

    if key in dictionary:
        return {
            "expanded": dictionary[key],
            "confidence": 1.0,
            "method": "dictionary"
        }

    if key_clean in dictionary:
        return {
            "expanded": dictionary[key_clean],
            "confidence": 1.0,
            "method": "dictionary"
        }

    return {
        "expanded": header,
        "confidence": 0.0,
        "method": "unresolved"
    }


def assess_header_style(details):

    total = len(details)

    if total == 0:
        return "UNKNOWN"

    count = sum(
        1
        for d in details
        if d["classification"] == "ABBREVIATED"
    )

    ratio = count / total

    if ratio >= 0.6:
        return "ABBREVIATED"

    elif ratio <= 0.2:
        return "STANDARD"

    return "MIXED"


def run(intake_output):

    print("\n[Agent 2] Starting Semantic Feature Agent")

    if intake_output["status"] != "ok":

        return {
            "status": "error",
            "error_message": "Agent 1 failed"
        }

    headers = intake_output["columns"]

    df = intake_output["dataframe"]

    dictionary = load_dictionary()

    expanded_headers = []

    expansion_confidence = []

    expansion_method = []

    per_header_detail = []

    for header in headers:

        classification = classify_header(header)

        result = expand_header(
            header,
            dictionary
        )

        expanded_headers.append(
            result["expanded"]
        )

        expansion_confidence.append(
            result["confidence"]
        )

        expansion_method.append(
            result["method"]
        )

        per_header_detail.append({

            "raw": header,

            "expanded": result["expanded"],

            "classification": classification,

            "confidence": result["confidence"],

            "method": result["method"]

        })

    header_style = assess_header_style(
        per_header_detail
    )

    return {

        "status": "ok",

        "raw_headers": headers,

        "expanded_headers": expanded_headers,

        "expansion_confidence":
            expansion_confidence,

        "expansion_method":
            expansion_method,

        "header_style":
            header_style,

        "per_header_detail":
            per_header_detail,

        "dataframe":
            df,

        "error_message":
            None
    }