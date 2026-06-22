from agents.agent1_intake import run as agent1_run
from agents.agent2_features import run as agent2_run
from agents.agent3_matcher import run as agent3_run


print("\n===================================")
print("TESTING AGENT 1 → AGENT 2 → AGENT 3")
print("===================================\n")


# Change filename if needed
dataset_path = "data/datasets/diabetes.csv"


# ------------------------
# Agent 1
# ------------------------
agent1_output = agent1_run(dataset_path)

if agent1_output["status"] != "ok":

    print("\nAgent 1 Failed")
    print(agent1_output["error_message"])

    exit()


# ------------------------
# Agent 2
# ------------------------
agent2_output = agent2_run(agent1_output)

if agent2_output["status"] != "ok":

    print("\nAgent 2 Failed")
    print(agent2_output["error_message"])

    exit()


# ------------------------
# Agent 3
# ------------------------
agent3_output = agent3_run(agent2_output)

if agent3_output["status"] == "error":

    print("\nAgent 3 Failed")
    print(agent3_output["error_message"])

    exit()


# ------------------------
# Results
# ------------------------
print("\n===================================")
print("AGENT 3 RESULTS")
print("===================================")

print(
    f"\nSelected Model: "
    f"{agent3_output['selected_model']}"
)

print(
    f"Match Score: "
    f"{agent3_output['match_score']}"
)

print(
    f"Gate Passed: "
    f"{agent3_output['gate_passed']}"
)

print(
    f"Exact Matches: "
    f"{agent3_output['exact_matches']}"
)

print(
    f"Unresolved Features: "
    f"{agent3_output['unresolved_features']}"
)

print("\nModel Ranking:")

for item in agent3_output["ranking"]:

    print(
        f"  {item['model']} "
        f"→ {item['score']}"
    )

print("\n===================================")
print("TEST COMPLETE")
print("===================================")