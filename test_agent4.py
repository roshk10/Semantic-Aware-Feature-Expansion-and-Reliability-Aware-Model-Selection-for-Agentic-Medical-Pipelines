from agents.agent1_intake import run as agent1_run
from agents.agent2_features import run as agent2_run
from agents.agent3_matcher import run as agent3_run
from agents.agent4_inference import run as agent4_run


print("\n===================================")
print("TESTING AGENT 1 → 2 → 3 → 4")
print("===================================\n")


dataset_path = "data/datasets/heart_disease.csv"


# ------------------
# Agent 1
# ------------------
agent1_output = agent1_run(
    dataset_path
)

if agent1_output["status"] != "ok":

    print("\nAgent 1 Failed")
    print(
        agent1_output["error_message"]
    )

    exit()


# ------------------
# Agent 2
# ------------------
agent2_output = agent2_run(
    agent1_output
)

if agent2_output["status"] != "ok":

    print("\nAgent 2 Failed")
    print(
        agent2_output["error_message"]
    )

    exit()


# ------------------
# Agent 3
# ------------------
agent3_output = agent3_run(
    agent2_output
)

if agent3_output["status"] == "error":

    print("\nAgent 3 Failed")
    print(
        agent3_output["error_message"]
    )

    exit()


# ------------------
# Agent 4
# ------------------
agent4_output = agent4_run(
    agent3_output
)

if agent4_output["status"] != "ok":

    print("\nAgent 4 Failed")
    print(
        agent4_output["error_message"]
    )

    exit()


print("\n===================================")
print("AGENT 4 RESULTS")
print("===================================")

print(
    f"\nSelected Model: "
    f"{agent4_output['selected_model']}"
)

print(
    f"\nTotal Predictions: "
    f"{agent4_output['total_predictions']}"
)

print(
    "\nFirst 10 Predictions:"
)

print(
    agent4_output["predictions"][:10]
)

print(
    "\nFirst 3 Probability Rows:"
)

for row in (
    agent4_output["probabilities"][:3]
):
    print(row)

print("\n===================================")
print("TEST COMPLETE")
print("===================================")