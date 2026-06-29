import json
import os
import re

from utils.sapbert import (
    get_embedding,
    cosine_similarity
)


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
MODEL_DATABASE_PATH = (
    "data/models/model_database.json"
)

SAPBERT_EXPANSION_THRESHOLD = 0.65

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
def expand_with_sapbert(
    header,
    model_db
):

    all_features = []

    for model_info in model_db.values():

        all_features.extend(
            model_info.get(
                "canonical_features",
                []
            )
        )

    if len(all_features) == 0:

        return {
            "expanded": header,
            "confidence": 0.0,
            "method": "unresolved"
        }

    header_embedding = (
        get_embedding(header)
    )

    best_feature = None

    best_score = -1.0

    for feature in all_features:

        feature_embedding = (
            get_embedding(feature)
        )

        score = cosine_similarity(
            header_embedding,
            feature_embedding
        )

        if score > best_score:

            best_score = score

            best_feature = feature

    if best_score >= SAPBERT_EXPANSION_THRESHOLD:

        return {

            "expanded":
                best_feature,

            "confidence":
                round(best_score, 4),

            "method":
                "sapbert"
        }

    return {

        "expanded":
            header,

        "confidence":
            round(best_score, 4),

        "method":
            "unresolved"
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
    model_db = {}

    if os.path.exists(
        MODEL_DATABASE_PATH
    ):

       with open(
           MODEL_DATABASE_PATH,
           "r"
        ) as f:

            model_db = json.load(f)

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

        if result["method"] == "unresolved":

            print(
            f"[Agent 2] "
            f"'{header}' "
            f"not found in dictionary."
            )

            result = expand_with_sapbert(
                header,
                model_db
             )
        print(
            f"[Agent 2] "
            f"{header}"
            f" -> "
            f"{result['expanded']} "
            f"[{result['method']}, "
            f"{result['confidence']}]"
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