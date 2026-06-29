import json
import os
from datetime import datetime

RUN_LOG_PATH = "logs/reliability_memory.json"

CONFIDENCE_THRESHOLD = 0.60


# --------------------------------------------------
# Run Log
# --------------------------------------------------

def load_run_log():

    if not os.path.exists(
        RUN_LOG_PATH
    ):
        return []

    with open(
        RUN_LOG_PATH,
        "r"
    ) as f:

        try:
            return json.load(f)

        except json.JSONDecodeError:

            return []


def save_run_log(log):

    os.makedirs(
        "logs",
        exist_ok=True
    )

    with open(
        RUN_LOG_PATH,
        "w"
    ) as f:

        json.dump(
            log,
            f,
            indent=2
        )


# --------------------------------------------------
# Agent 1 Score
# --------------------------------------------------

def score_agent1(
        intake_output
):

    warnings = []

    if intake_output["status"] != "ok":

        return 0.0, [
            "Agent 1 failed"
        ]

    if intake_output.get(
        "pii_flag"
    ):

        warnings.append(
            "PII columns detected"
        )

    return 1.0, warnings


# --------------------------------------------------
# Agent 2 Score
# --------------------------------------------------

def score_agent2(
        features_output
):

    warnings = []

    if features_output[
        "status"
    ] != "ok":

        return 0.0, [
            "Agent 2 failed"
        ]

    confidences = (
        features_output.get(
            "expansion_confidence",
            []
        )
    )

    if len(confidences) == 0:

        return 0.0, [
            "No confidence scores"
        ]

    score = (
        sum(confidences)
        /
        len(confidences)
    )

    unresolved = [

        d["raw"]

        for d in features_output[
            "per_header_detail"
        ]

        if d["method"]
        ==
        "unresolved"
    ]

    if unresolved:

        warnings.append(

            f"Unresolved headers: "
            f"{unresolved}"
        )

    return round(
        score,
        4
    ), warnings


# --------------------------------------------------
# Agent 3 Score
# --------------------------------------------------

def score_agent3(
    matcher_output
):
    warnings = []
    if matcher_output.get(
    "conflict_detected"
    ):
    
       warnings.append(
        "DOMAIN_CONFLICT"
       )

       return 0.0, warnings

    

    if matcher_output[
        "status"
    ] == "error":

        return 0.0, [
            "Agent 3 failed"
        ]

    score = matcher_output.get(
    "match_score"
    )

    if score is None:
       score = 0.0

    if not matcher_output.get(
        "gate_passed"
    ):

        warnings.append(

            f"Gate failed "
            f"({score})"
        )

    ranking = matcher_output.get(
        "ranking",
        []
    )

    if len(ranking) >= 2:

        gap = (
            ranking[0]["score"]
            -
            ranking[1]["score"]
        )

        if gap < 0.1:

            warnings.append(

                "Ambiguous model "
                "selection"
            )

    return round(
        score,
        4
    ), warnings


# --------------------------------------------------
# Agent 4 Score
# --------------------------------------------------

def score_agent4(
        inference_output
):

    warnings = []

    if inference_output[
        "status"
    ] == "error":

        return 0.0, [
            "Agent 4 failed"
        ]

    probabilities = (
        inference_output.get(
            "probabilities",
            []
        )
    )

    if len(probabilities) == 0:

        return 0.0, [
            "No probabilities"
        ]

    if isinstance(
        probabilities[0],
        list
    ):

        positive_probs = [

            row[1]

            for row
            in probabilities
        ]

    else:

        positive_probs = (
            probabilities
        )

    mean_decisiveness = (

        sum(

            abs(
                p - 0.5
            )

            for p
            in positive_probs

        )

        /

        len(
            positive_probs
        )
    )

    score = round(

        min(
            mean_decisiveness * 2,
            1.0
        ),

        4
    )

    if score < 0.3:

        warnings.append(

            "Low prediction "
            "confidence"
        )

    return score, warnings


# --------------------------------------------------
# Failure Detection
# --------------------------------------------------

def detect_failures(

        intake_output,

        features_output,

        matcher_output,

        inference_output
):
    failures = []
    if matcher_output.get(
    "conflict_detected"
):

     failures.append(
        "DOMAIN_CONFLICT"
    )

    
    if inference_output.get(
    "status"
    ) == "skipped":

      failures.append(
        "INFERENCE_BLOCKED"
    )

    if intake_output[
        "status"
    ] != "ok":

        failures.append(
            "INTAKE_FAILURE"
        )

    if features_output[
        "status"
    ] != "ok":

        failures.append(
            "EXPANSION_FAILURE"
        )

    else:

        unresolved = [

            d

            for d
            in features_output[
                "per_header_detail"
            ]

            if d["method"]
            ==
            "unresolved"
        ]

        if unresolved:

            failures.append(
                "PARTIAL_EXPANSION"
            )

    if matcher_output[
    "status"
    ] == "error":

        failures.append(
            "MATCHING_FAILURE"
        )

    elif matcher_output.get(
        "conflict_detected"
    ):

        pass

    elif not matcher_output[
        "gate_passed"
    ]:

        failures.append(
            "LOW_CONFIDENCE_MATCH"
        )
    return failures


# --------------------------------------------------
# Pipeline Health
# --------------------------------------------------

def compute_pipeline_health(

        agent_scores,

        failures
):

    weights = {

        "agent1": 0.10,

        "agent2": 0.25,

        "agent3": 0.40,

        "agent4": 0.25
    }

    weighted_score = sum(

        agent_scores[a]
        *
        weights[a]

        for a
        in weights
    )

    penalty = (
        len(failures)
        *
        0.10
    )

    health = max(

        0.0,

        weighted_score
        -
        penalty
    )

    return round(
        health,
        4
    )


# --------------------------------------------------
# Main Entry
# --------------------------------------------------

def run(

        intake_output,

        features_output,

        matcher_output,

        inference_output,

        dataset_name="unknown"
):

    print(
        "\n[Agent 5] "
        "Starting Reliability "
        "Intelligence Agent"
    )

    a1_score, a1_warn = (
        score_agent1(
            intake_output
        )
    )

    a2_score, a2_warn = (
        score_agent2(
            features_output
        )
    )

    a3_score, a3_warn = (
        score_agent3(
            matcher_output
        )
    )

    a4_score, a4_warn = (
        score_agent4(
            inference_output
        )
    )

    agent_scores = {

        "agent1":
            a1_score,

        "agent2":
            a2_score,

        "agent3":
            a3_score,

        "agent4":
            a4_score
    }

    warnings = (

        a1_warn
        +
        a2_warn
        +
        a3_warn
        +
        a4_warn
    )

    failures = detect_failures(

        intake_output,

        features_output,

        matcher_output,

        inference_output
    )

    pipeline_health = (
        compute_pipeline_health(

            agent_scores,

            failures
        )
    )

    run_entry = {

        "timestamp":
            datetime.now().isoformat(),

        "dataset_name":
            dataset_name,

        "selected_model":
            matcher_output.get(
                "selected_model"
            ),

        "match_score":
            matcher_output.get(
                "match_score"
            ),

        "pipeline_health":
            pipeline_health,

        "failures":
            failures
    }

    run_log = load_run_log()

    run_log.append(
        run_entry
    )

    save_run_log(
        run_log
    )

    print(
        "\nPipeline Health:",
        pipeline_health
    )

    print(
        "Failures:",
        failures
        if failures
        else "None"
    )

    return {

        "status":
            "ok",

        "pipeline_health":
            pipeline_health,

        "agent_scores":
            agent_scores,

        "failures":
            failures,

        "warnings":
            warnings,

        "run_log_entry":
            run_entry,

        "error_message":
            None
    }