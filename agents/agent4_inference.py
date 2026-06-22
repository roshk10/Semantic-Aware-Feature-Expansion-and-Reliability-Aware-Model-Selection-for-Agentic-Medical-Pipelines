import json
import joblib


def load_model(model_path):

    return joblib.load(model_path)


def run(matcher_output):

    print(
        "\n[Agent 4] Starting Inference Agent"
    )

    if matcher_output["status"] != "ok":

        return {

            "status": "error",

            "error_message":
                "Agent 3 failed"
        }
    if not matcher_output["gate_passed"]:

        return {

            "status": "skipped",

            "selected_model":
                matcher_output["selected_model"],

            "error_message":
                "Confidence gate failed"
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
                f"Unsupported model: {selected_model}"
        }

    predictions = model.predict(X)

    probabilities = (
        model.predict_proba(X)
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
    }