import json
import os

MODEL_DATABASE_PATH = "data/models/model_database.json"

CONFIDENCE_THRESHOLD = 0.60


def load_model_database():

    if not os.path.exists(MODEL_DATABASE_PATH):

        raise FileNotFoundError(
            f"Model database not found: {MODEL_DATABASE_PATH}"
        )

    with open(MODEL_DATABASE_PATH, "r") as f:

        return json.load(f)


def normalize(text):

    return (
        str(text)
        .lower()
        .strip()
        .replace("_", " ")
        .replace("-", " ")
    )


def score_model(
        expanded_headers,
        canonical_features
):

    expanded_set = {
        normalize(h)
        for h in expanded_headers
    }

    canonical_set = {
        normalize(f)
        for f in canonical_features
    }

    matches = expanded_set.intersection(
        canonical_set
    )

    # Safety: avoid division by zero
    if len(expanded_set) == 0:
        return 0.0, []

    score = (
        len(matches)
        /
        len(expanded_set)
    )

    return round(score, 4), list(matches)


def run(features_output):

    print(
        "\n[Agent 3] Starting Model Matching Agent"
    )

    if features_output["status"] != "ok":

        return {

            "status": "error",

            "error_message":
                "Agent 2 failed"
        }

    expanded_headers = (
        features_output["expanded_headers"]
    )

    df = (
        features_output["dataframe"]
    )

    try:

        model_db = load_model_database()

    except Exception as e:

        return {

            "status": "error",

            "error_message":
                str(e)
        }

    ranking = []

    match_detail = {}

    for model_name, model_info in model_db.items():

        score, matches = score_model(

            expanded_headers,

            model_info["canonical_features"]
        )

        ranking.append({

            "model": model_name,

            "score": score
        })

        match_detail[model_name] = {

            "matched_features": matches,

            "match_count": len(matches)
        }

        print(
            f"[Agent 3] {model_name}"
            f" → {score}"
        )

    # Safety: no models in database
    if len(ranking) == 0:

        return {

            "status": "error",

            "error_message":
                "No models found in model database."
        }

    ranking.sort(

        key=lambda x: x["score"],

        reverse=True
    )

    best = ranking[0]

    selected_model = best["model"]

    match_score = best["score"]

    gate_passed = (

        match_score
        >=
        CONFIDENCE_THRESHOLD
    )

    if gate_passed:

        status = "ok"

    else:

        status = "low_confidence"

    exact_matches = len(

        match_detail[selected_model]
        ["matched_features"]
    )

    unresolved_features = (

        len(expanded_headers)
        -
        exact_matches
    )

    return {

        "status": status,

        "selected_model":
            selected_model,

        "match_score":
            match_score,

        "gate_passed":
            gate_passed,

        "ranking":
            ranking,

        "match_detail":
            match_detail,

        "exact_matches":
            exact_matches,

        "unresolved_features":
            unresolved_features,

        "expanded_headers":
            expanded_headers,

        "dataframe":
            df,

        "error_message":
            None
    }