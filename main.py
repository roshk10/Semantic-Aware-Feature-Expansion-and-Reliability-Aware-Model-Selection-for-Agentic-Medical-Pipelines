import argparse
import os

from agents.agent1_intake import run as agent1
from agents.agent2_features import run as agent2
from agents.agent3_matcher import run as agent3
from agents.agent4_inference import run as agent4
from agents.agent5_evaluation import run as agent5


def parse_args():

    parser = argparse.ArgumentParser(
        description="SAFE-MAP Medical Agent Pipeline"
    )

    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to dataset"
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional run name"
    )

    return parser.parse_args()


def print_final_report(
        a1_out,
        a2_out,
        a3_out,
        a4_out,
        a5_out,
        dataset_name
):

    print("\n")
    print("=" * 60)
    print("SAFE-MAP FINAL REPORT")
    print("=" * 60)

    print(
        f"\nDataset: {dataset_name}"
    )

    print(
        f"Format: "
        f"{a1_out.get('detected_format')}"
    )

    print(
        f"Rows: "
        f"{a1_out.get('row_count')}"
    )

    print(
        f"Columns: "
        f"{a1_out.get('col_count')}"
    )

    print(
        f"PII Flag: "
        f"{a1_out.get('pii_flag')}"
    )

    print("\nMODEL MATCHING")

    print(
        f"Selected Model: "
        f"{a3_out.get('selected_model')}"
    )

    print(
        f"Match Score: "
        f"{a3_out.get('match_score')}"
    )

    print(
        f"Gate Passed: "
        f"{a3_out.get('gate_passed')}"
    )

    print("\nMODEL RANKING")

    for item in a3_out.get(
        "ranking",
        []
    ):

        print(
            f"{item['model']} "
            f"-> "
            f"{item['score']}"
        )

    print("\nINFERENCE")

    if a4_out["status"] == "skipped":

        print(
            "Inference Skipped"
        )

    elif a4_out["status"] == "error":

        print(
            f"Error: "
            f"{a4_out['error_message']}"
        )

    else:

        print(
            f"Rows Scored: "
            f"{a4_out['row_count']}"
        )

        print(
            "\nFirst 5 Predictions:"
        )

        print(
            a4_out["predictions"][:5]
        )

    print("\nRELIABILITY")

    print(
        f"Pipeline Health: "
        f"{a5_out['pipeline_health']}"
    )

    print(
        f"Failures: "
        f"{a5_out['failures']}"
    )

    print("\nAgent Scores")

    for agent_name, score in (
        a5_out["agent_scores"]
        .items()
    ):

        print(
            f"{agent_name}: "
            f"{score}"
        )

    print("\n")
    print("=" * 60)
    print("RUN COMPLETE")
    print("=" * 60)


def run_pipeline(
        file_path,
        dataset_name
):

    print("\n")
    print("=" * 60)
    print("SAFE-MAP PIPELINE STARTING")
    print("=" * 60)

    print(
        f"\nInput File: "
        f"{file_path}"
    )

    # ------------------
    # Agent 1
    # ------------------

    a1_out = agent1(
        file_path
    )

    if a1_out["status"] != "ok":

        print(
            a1_out[
                "error_message"
            ]
        )

        return

    # ------------------
    # Agent 2
    # ------------------

    a2_out = agent2(
        a1_out
    )

    if a2_out["status"] != "ok":

        print(
            a2_out[
                "error_message"
            ]
        )

        return

    # ------------------
    # Agent 3
    # ------------------

    a3_out = agent3(
        a2_out
    )

    if a3_out["status"] == "error":

        print(
            a3_out[
                "error_message"
            ]
        )

        return

    # ------------------
    # Agent 4
    # ------------------

    a4_out = agent4(
        a3_out
    )

    # ------------------
    # Agent 5
    # ------------------

    a5_out = agent5(

        a1_out,

        a2_out,

        a3_out,

        a4_out,

        dataset_name=
        dataset_name
    )

    # ------------------
    # Final Report
    # ------------------

    print_final_report(

        a1_out,

        a2_out,

        a3_out,

        a4_out,

        a5_out,

        dataset_name
    )


if __name__ == "__main__":

    args = parse_args()

    if args.name:

        dataset_name = (
            args.name
        )

    else:

        dataset_name = (
            os.path.splitext(
                os.path.basename(
                    args.file
                )
            )[0]
        )

    run_pipeline(

        args.file,

        dataset_name
    )