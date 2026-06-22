from agents.agent1_intake import run as agent1_run
from agents.agent2_features import run as agent2_run


print("\n===================================")
print("TESTING AGENT 1 → AGENT 2")
print("===================================\n")


# Change filename if needed
dataset_path = "data/datasets/heart_disease.csv"


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
# Results
# ------------------------
print("\n===================================")
print("AGENT 2 RESULTS")
print("===================================")

print("\nRaw Headers:")

for h in agent2_output["raw_headers"]:
    print("  ", h)

print("\nExpanded Headers:")

for h in agent2_output["expanded_headers"]:
    print("  ", h)

print("\nExpansion Confidence:")

for c in agent2_output["expansion_confidence"]:
    print("  ", c)

print("\nExpansion Method:")

for m in agent2_output["expansion_method"]:
    print("  ", m)

print(
    f"\nHeader Style: "
    f"{agent2_output['header_style']}"
)

print("\nPer Header Detail:")

for item in agent2_output["per_header_detail"]:

    print(
        f"{item['raw']} "
        f"→ {item['expanded']} "
        f"[{item['method']}, "
        f"{item['confidence']}]"
    )

print("\n===================================")
print("TEST COMPLETE")
print("===================================")