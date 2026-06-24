import json
import joblib
import shap
import numpy as np


def load_model(model_path):

    return joblib.load(model_path)

# ==========================================
# V2.4 GLOBAL SHAP
# ==========================================

def _run_global_shap(model, X):

    try:

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = (
            explainer.shap_values(X)
        )

        mean_abs = (
            np.abs(shap_values)
            .mean(axis=0)
        )

        shap_summary = {

            feature: round(
                float(score),
                6
            )

            for feature, score
            in zip(
                X.columns,
                mean_abs
            )
        }

        shap_summary = dict(

            sorted(

                shap_summary.items(),

                key=lambda item:
                    item[1],

                reverse=True
            )
        )

        return shap_summary

    except Exception as e:

        print(
            f"[Agent 4] "
            f"SHAP failed: {e}"
        )

        return {}
def _get_top_features(
    shap_summary,
    n=5
):

    return list(
        shap_summary.keys()
    )[:n]

def run(matcher_output):

    print(
        "\n[Agent 4] Starting Inference Agent"
    )
    if matcher_output.get(
    "status"
    ) == "conflict":

       print(
        "[Agent 4] "
        "Inference blocked "
        "due to domain conflict."
    )

       return {

        "status":
            "skipped",

        "reason":
            "domain_conflict",

        "selected_model":
            None,

        "predictions":
            [],

        "probabilities":
            [],

        "row_count":
            0,

        "error_message":
            "Domain conflict detected.",
        "shap_summary": {},

        "top_features": [],
    }

    if matcher_output["status"] != "ok":

        return {

            "status": "error",

            "error_message":
                "Agent 3 failed",
            "shap_summary": {},

            "top_features": [],
        }
    if not matcher_output["gate_passed"]:

        return {

            "status": "skipped",

            "selected_model":
                matcher_output["selected_model"],

            "error_message":
                "Confidence gate failed",
            "shap_summary": {},

            "top_features": [],
        }

    selected_model = (
        matcher_output["selected_model"]
    )

    with open(
        "data/models/model_database.json",
        "r"
    ) as f:

        model_db = json.load(f)

    model_path = (
        model_db[selected_model]
        ["model_file"]
    )

    model = load_model(
        model_path
    )

    df = matcher_output["dataframe"]

    if selected_model == "heart_disease":

        X = df.drop(
            "target",
            axis=1
        )

    elif selected_model == "diabetes":

        X = df.drop(
            "Outcome",
            axis=1
        )

    else:

        return {

            "status": "error",

            "error_message":
                f"Unsupported model: {selected_model}",
            "shap_summary": {},

            "top_features": [],
        }

    predictions = model.predict(X)

    probabilities = (
        model.predict_proba(X)
    )
    print(
    "[Agent 4] "
    "Running SHAP..."
    )

    shap_summary = (
        _run_global_shap(
            model,
            X
        )
    )

    top_features = (
        _get_top_features(
            shap_summary,
            n=5
        )
    )

    if shap_summary:

        print(
            "\n[Agent 4] "
            "Top SHAP Features:"
        )

        for feature in top_features:

            print(
                f"   {feature}"
                f" -> "
                f"{shap_summary[feature]}"
            )

    return {

        "status": "ok",

        "selected_model":
            selected_model,

        "predictions":
            predictions.tolist(),

        "probabilities":
            probabilities.tolist(),

        "total_predictions":
            len(predictions),

        "dataframe":
            df,

        "error_message":
            None,
        "row_count": len(df),
        "shap_summary":
            shap_summary,

        "top_features":
            top_features,
    }

