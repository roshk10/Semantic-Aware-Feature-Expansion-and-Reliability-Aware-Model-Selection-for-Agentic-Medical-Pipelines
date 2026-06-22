from agents.agent1_intake import run as intake
from agents.agent2_features import run as features
from agents.agent3_matcher import run as matcher
from agents.agent4_inference import run as inference
from agents.agent5_evaluation import run as evaluate


print("\n===================================")
print("TESTING AGENT 1 → 2 → 3 → 4 → 5")
print("===================================\n")

a1 = intake(
    "data/datasets/heart_disease.csv"
)

a2 = features(a1)

a3 = matcher(a2)

a4 = inference(a3)

a5 = evaluate(

    a1,

    a2,

    a3,

    a4,

    dataset_name=
    "heart_test_01"
)

print("\n===================================")
print("AGENT 5 RESULTS")
print("===================================")

print(
    "\nPipeline Health:",
    a5["pipeline_health"]
)

print(
    "\nAgent Scores:"
)

for k, v in a5[
    "agent_scores"
].items():

    print(
        f"{k}: {v}"
    )

print(
    "\nFailures:",
    a5["failures"]
)

print(
    "\nWarnings:",
    a5["warnings"]
)

print("\n===================================")
print("TEST COMPLETE")
print("===================================")