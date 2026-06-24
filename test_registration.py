from agents.agent1_intake import run as intake
from agents.agent2_features import run as features
from agents.agent3_matcher import run as matcher

print("\n=== Testing External Model Registration ===")

a1 = intake("data/datasets/heart_disease.csv")
a2 = features(a1)
a3 = matcher(a2)

print("\nModels available to Agent 3:\n")

for r in a3["ranking"]:
    print(
        f"{r['model']} -> {r['score']}"
    )