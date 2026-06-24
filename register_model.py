"""
register_model.py
------------------
Registers an external model into the SAFE-MAP model database.

Usage:
    python register_model.py --config path/to/config.json

What it does:
    1. Reads the config file
    2. Validates all required fields
    3. Checks for duplicate model names
    4. Appends to model_database.json
    5. Confirms registration

What it does NOT do (V3 scope):
    - Does not download or fetch the model file
    - Does not infer features from the model object itself
    - Does not validate that the .pkl file works

V2 scope:
    Models with a declared feature list only.
    The user provides canonical_features explicitly.
    Covers GitHub models with a README or paper
    that lists expected input features.
"""

import argparse
import json
import os
import sys
from datetime import datetime

MODEL_DATABASE_PATH = "data/models/model_database.json"

REQUIRED_FIELDS = [
    "model_name",
    "description",
    "domain",
    "canonical_features",
    "model_file",
    "target_column"
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_config(config: dict) -> list:
    """
    Checks the config for required fields and correctness.

    Returns a list of error strings.
    Empty list = config is valid.
    """
    errors = []

    # Check all required fields exist
    for field in REQUIRED_FIELDS:
        if field not in config:
            errors.append(f"Missing required field: '{field}'")

    # Stop here if fields are missing
    # Remaining checks assume fields exist
    if errors:
        return errors

    # model_name must be non-empty string
    name = config["model_name"]
    if not isinstance(name, str) or not name.strip():
        errors.append("'model_name' must be a non-empty string.")

    # model_name must use only letters, numbers, underscores
    # Must match key format used in model_database.json
    elif not all(c.isalnum() or c == "_" for c in name.strip()):
        errors.append(
            f"'model_name' must contain only letters, numbers, "
            f"and underscores. Got: '{name}'"
        )

    # canonical_features must be non-empty list of strings
    features = config.get("canonical_features", [])
    if not isinstance(features, list) or len(features) == 0:
        errors.append(
            "'canonical_features' must be a non-empty list of strings."
        )
    else:
        non_strings = [f for f in features if not isinstance(f, str)]
        if non_strings:
            errors.append(
                f"All items in 'canonical_features' must be strings. "
                f"Non-strings found: {non_strings}"
            )

    # domain must be non-empty string
    domain = config.get("domain", "")
    if not isinstance(domain, str) or not domain.strip():
        errors.append("'domain' must be a non-empty string.")

    # description must be non-empty string
    desc = config.get("description", "")
    if not isinstance(desc, str) or not desc.strip():
        errors.append("'description' must be a non-empty string.")

    # target_column must be non-empty string
    target = config.get("target_column", "")
    if not isinstance(target, str) or not target.strip():
        errors.append("'target_column' must be a non-empty string.")

    return errors


def _check_duplicate(model_name: str, model_db: dict) -> bool:
    """
    Returns True if model_name already exists in the database.
    """
    return model_name.strip() in model_db


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------

def _load_model_database() -> dict:
    """
    Loads existing model_database.json.
    Returns empty dict if file does not exist yet.
    """
    if not os.path.exists(MODEL_DATABASE_PATH):
        print(f"[Register] No database found at {MODEL_DATABASE_PATH}.")
        print(f"[Register] A new database will be created.")
        return {}

    with open(MODEL_DATABASE_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            print(f"[Register] ERROR: Database file is corrupted: {e}")
            sys.exit(1)


def _save_model_database(model_db: dict) -> None:
    """
    Saves the updated model database back to disk.
    Creates the directory if it does not exist.
    """
    os.makedirs(
        os.path.dirname(MODEL_DATABASE_PATH),
        exist_ok=True
    )
    with open(MODEL_DATABASE_PATH, "w") as f:
        json.dump(model_db, f, indent=2)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(config_path: str) -> bool:
    """
    Main registration function.

    Parameters
    ----------
    config_path : str
        Path to the JSON config file.

    Returns
    -------
    bool
        True if registration succeeded.
        False if validation or duplicate check failed.
    """

    print(f"\n[Register] Reading config: {config_path}")

    # Check config file exists
    if not os.path.exists(config_path):
        print(f"[Register] ERROR: Config file not found: {config_path}")
        return False

    # Load config JSON
    with open(config_path, "r") as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            print(f"[Register] ERROR: Config file is not valid JSON: {e}")
            return False

    # Validate fields
    errors = _validate_config(config)
    if errors:
        print(f"[Register] ERROR: Config validation failed:")
        for err in errors:
            print(f"  ✗ {err}")
        return False

    print(f"[Register] Config valid.")

    model_name = config["model_name"].strip()

    # Load existing database
    model_db = _load_model_database()

    # Check for duplicate
    if _check_duplicate(model_name, model_db):
        print(
            f"[Register] ERROR: Model '{model_name}' already exists "
            f"in the database."
        )
        print(
            f"[Register] To update an existing model, manually edit "
            f"{MODEL_DATABASE_PATH} and change the relevant entry."
        )
        return False

    # Build the database entry
    # "registration_type": "external" distinguishes
    # registered models from your own trained models
    entry = {
        "description": config["description"].strip(),
        "domain": config["domain"].strip(),
        "source": config.get("source", "user_registered").strip(),
        "canonical_features": [
            f.strip().lower()
            for f in config["canonical_features"]
        ],
        "model_file": config["model_file"].strip(),
        "target_column": config["target_column"].strip(),
        "registered_at": datetime.now().isoformat(),
        "registration_type": "external"
    }

    # Write to database
    model_db[model_name] = entry
    _save_model_database(model_db)

    # Confirm
    print(f"\n[Register] ✓ '{model_name}' registered successfully.")
    print(f"  Description : {entry['description']}")
    print(f"  Domain      : {entry['domain']}")
    print(f"  Source      : {entry['source']}")
    print(f"  Features    : {entry['canonical_features']}")
    print(f"  Model file  : {entry['model_file']}")
    print(f"  Target col  : {entry['target_column']}")
    print(f"  Registered  : {entry['registered_at']}")
    print(
        f"\n[Register] Agent 3 will now include '{model_name}' "
        f"in all future runs automatically."
    )

    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Register an external model into SAFE-MAP"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model registration JSON config file"
    )

    args = parser.parse_args()
    success = register(args.config)

    if not success:
        print("\n[Register] Registration failed. No changes made.")
        sys.exit(1)

    print("\n[Register] Done.")
    sys.exit(0)