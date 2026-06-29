# SAFE-MAP

### Semantic-Aware Feature Expansion and Reliability-Aware Model Selection for Agentic Medical Pipelines

SAFE-MAP is a modular multi-agent framework that automatically analyzes structured medical datasets, expands non-standard clinical feature names, intelligently selects the most suitable prediction model, performs disease inference, generates explainable predictions using SHAP, and evaluates overall pipeline reliability.

The system is designed to improve interoperability across heterogeneous clinical datasets while reducing unsafe model selection through semantic understanding and reliability-aware decision making.

---

# Features

* Multi-Agent Medical AI Pipeline (5-Agent Architecture)
* Intelligent Clinical Feature Expansion
* Dictionary-based Medical Abbreviation Resolution
* SapBERT Semantic Fallback for Unknown Biomedical Terms
* Automatic Dataset-to-Model Matching
* Domain Conflict Detection for Safe Model Selection
* External Model Registration without modifying pipeline code
* XGBoost-based Clinical Prediction
* SHAP Global Feature Importance
* SHAP Local Prediction Explanation
* Reliability Intelligence Agent with Pipeline Health Score
* Confidence-aware Clinical Decision Support

---

# System Architecture

```text
Medical Dataset
      │
      ▼
Agent 1
Data Intake & Validation
      │
      ▼
Agent 2
Semantic Feature Expansion
(Dictionary + SapBERT)
      │
      ▼
Agent 3
Model Matching
+
Confidence Gate
+
Conflict Detection
      │
      ▼
Agent 4
Prediction
+
SHAP Explainability
      │
      ▼
Agent 5
Reliability Intelligence
      │
      ▼
Final Prediction Report
```

---

# Repository Structure

```text
SAFE-MAP/
│
├── agents/
│   ├── agent1_intake.py
│   ├── agent2_features.py
│   ├── agent3_matcher.py
│   ├── agent4_inference.py
│   └── agent5_evaluation.py
│
├── data/
│   ├── datasets/
│   ├── models/
│   └── reliability/
│
├── utils/
│   └── sapbert.py
│
├── main.py
├── register_model.py
├── train_models.py
├── requirements.txt
└── README.md
```

---

# Agent Overview

### Agent 1 — Data Intake

* Dataset validation
* File format detection
* Metadata extraction
* PII column detection

---

### Agent 2 — Semantic Feature Expansion

* Header classification
* Clinical abbreviation expansion
* Dictionary lookup
* SapBERT semantic fallback for unseen biomedical terms

---

### Agent 3 — Intelligent Model Matching

* Canonical feature matching
* Confidence scoring
* Model ranking
* Confidence gate
* Domain conflict detection
* External model support

---

### Agent 4 — Prediction & Explainability

* Dataset preprocessing
* XGBoost inference
* SHAP global feature importance
* SHAP local prediction explanation

---

### Agent 5 — Reliability Intelligence

* Agent-level scoring
* Failure detection
* Pipeline health computation
* Reliability reporting

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Hugging Face Transformers
* SapBERT
* PyTorch
* Joblib

---

# Supported Models

Current registered prediction models include:

* Heart Disease Prediction
* Diabetes Prediction
* Sepsis Prediction (Externally Registered)
* Palliative Care (Placeholder)

The architecture supports dynamic registration of additional prediction models without changing the core pipeline.

---

# Installation

Clone the repository

```bash
git clone <repository-url>
cd Semantic-Aware-Feature-Expansion-and-Reliability-Aware-Model-Selection-for-Agentic-Medical-Pipelines
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Pipeline

Heart Disease Dataset

```bash
python main.py --file data/datasets/heart_disease.csv
```

Diabetes Dataset

```bash
python main.py --file data/datasets/diabetes.csv
```

Register a New Prediction Model

```bash
python register_model.py --config model_config.json
```

---

# Example Output

The pipeline produces:

* Dataset metadata
* Expanded clinical features
* Model ranking
* Selected prediction model
* Prediction results
* SHAP explanations
* Reliability evaluation
* Pipeline health score

---

# Research Contributions

The proposed SAFE-MAP framework introduces:

* Hybrid semantic feature expansion using a clinical abbreviation dictionary with SapBERT fallback.
* Domain-aware intelligent model selection with automatic conflict detection.
* Plug-and-play external model registration architecture.
* Reliability-aware explainability using SHAP and agent-level evaluation.

---

# Future Work

* Support additional medical prediction models
* FHIR and Electronic Health Record integration
* Clinical dashboard for deployment
* Human-in-the-loop validation
* Retrieval-Augmented Generation (RAG) support
* Multi-modal medical data support

---

# Authors

**Roshan Raguraman**

Summer Internship Project

Department of Computer Science and Engineering

---

# License

This project is intended for academic and research purposes.
