import pandas as pd

merged = {

    # HEART
    "cp": [1],
    "trestbps": [130],
    "chol": [250],
    "fbs": [0],
    "restecg": [1],
    "thalach": [150],

    # DIABETES
    "Pregnancies": [2],
    "Glucose": [120],
    "BloodPressure": [70],
    "Insulin": [80],
    "BMI": [28.5],
    "DiabetesPedigreeFunction": [0.5]
}

pd.DataFrame(
    merged
).to_csv(

    "data/datasets/merged_test.csv",

    index=False
)
from agents.agent1_intake import run as intake
from agents.agent2_features import run as features
from agents.agent3_matcher import run as matcher
from agents.agent4_inference import run as inference
from agents.agent5_evaluation import run as evaluate

a1 = intake(
    "data/datasets/merged_test.csv"
)

a2 = features(a1)

a3 = matcher(a2)

a4 = inference(a3)

a5 = evaluate(
    a1,
    a2,
    a3,
    a4,
    dataset_name="conflict_test_01"
)

a1 = intake(
    "data/datasets/merged_test.csv"
)

a2 = features(a1)

a3 = matcher(a2)

a4 = inference(a3)

a5 = evaluate(
    a1,
    a2,
    a3,
    a4,
    dataset_name="conflict_test_01"
)