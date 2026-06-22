"""
Agent 1: Data Intake Agent
--------------------------
Responsibilities:
  - Validate that the file exists and is non-empty
  - Detect its format using python-magic (content-based, not extension-based)
  - Load it into a pandas DataFrame
  - Run a basic PII flag scan (regex-based, lightweight)
  - Return a structured output dict for Agent 2

Output schema:
  {
    "status": "ok" | "error",
    "file_path": str,
    "detected_format": "csv" | "excel" | "json" | "unknown",
    "row_count": int,
    "col_count": int,
    "columns": [str, ...],
    "pii_flag": bool,
    "pii_columns": [str, ...],   # columns that may contain PII
    "dataframe": pd.DataFrame,   # passed in memory to next agent
    "error_message": str | None
  }
"""

import os
import re
import pandas as pd

# python-magic detects file type from content bytes, not the filename extension.
# This matters because someone could rename a .xlsx file to .csv and fool an
# extension-based check. Magic reads the actual file signature (first N bytes).
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    print("[Agent 1 WARNING] python-magic not installed. Falling back to extension-based detection.")


# --- PII detection patterns ---
# These are intentionally simple regex patterns.
# Full Presidio integration is listed as future work.
# For the demo, we flag columns whose NAME matches known PII patterns
# (e.g., a column called "ssn" or "patient_email" is suspicious).
PII_COLUMN_PATTERNS = [
    r"\bname\b",
    r"\bemail\b",
    r"\bphone\b",
    r"\bssn\b",
    r"\bsocial.?security\b",
    r"\baddress\b",
    r"\bzip\b",
    r"\bdob\b",
    r"\bdate.?of.?birth\b",
    r"\bpatient.?id\b",
    r"\bmedical.?record\b",
    r"\bip.?address\b",
]

def _flag_pii_columns(columns: list[str]) -> list[str]:
    """
    Checks each column name against known PII regex patterns.
    Returns a list of column names that match.

    Why column names only?
      Because scanning every cell value is slow and Presidio handles that.
      For Agent 1, we just want a quick structural warning.
    """
    flagged = []
    for col in columns:
        col_lower = col.lower().strip()
        for pattern in PII_COLUMN_PATTERNS:
            if re.search(pattern, col_lower):
                flagged.append(col)
                break  # no need to check more patterns for this column
    return flagged


def _detect_format_magic(file_path: str) -> str:
    """
    Uses python-magic to read the file's MIME type from its content.
    Maps the MIME type to one of our four known formats.
    """
    mime = magic.from_file(file_path, mime=True)

    # MIME type → our format label
    mime_map = {
        "text/csv": "csv",
        "text/plain": "csv",            # some CSVs appear as plain text
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "excel",
        "application/vnd.ms-excel": "excel",
        "application/json": "json",
    }
    return mime_map.get(mime, "unknown")


def _detect_format_extension(file_path: str) -> str:
    """
    Fallback: detect format from file extension.
    Less reliable than content-based detection, but works without python-magic.
    """
    ext = os.path.splitext(file_path)[1].lower()
    ext_map = {
        ".csv": "csv",
        ".xlsx": "excel",
        ".xls": "excel",
        ".json": "json",
    }
    return ext_map.get(ext, "unknown")


def _load_dataframe(file_path: str, fmt: str) -> pd.DataFrame:
    """
    Loads the file into a pandas DataFrame based on detected format.
    Each format needs a different pandas reader.
    """
    if fmt == "csv":
        return pd.read_csv(file_path)
    elif fmt == "excel":
        return pd.read_excel(file_path)
    elif fmt == "json":
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Cannot load unknown format: {fmt}")


def run(file_path: str) -> dict:
    """
    Main entry point for Agent 1.

    Parameters
    ----------
    file_path : str
        Path to the uploaded medical data file.

    Returns
    -------
    dict
        Structured output for Agent 2 to consume.
    """

    print(f"\n[Agent 1] Starting Data Intake for: {file_path}")

    # --- Step 1: File existence and size check ---
    # If the file doesn't exist or is empty, there's nothing to do.
    # We return early with an error status so the pipeline can handle it cleanly.
    if not os.path.exists(file_path):
        return {
            "status": "error",
            "file_path": file_path,
            "error_message": f"File not found: {file_path}",
            "dataframe": None,
        }

    file_size_bytes = os.path.getsize(file_path)
    if file_size_bytes == 0:
        return {
            "status": "error",
            "file_path": file_path,
            "error_message": "File is empty (0 bytes).",
            "dataframe": None,
        }

    print(f"[Agent 1] File found. Size: {file_size_bytes} bytes.")

    # --- Step 2: Format detection ---
    # Prefer content-based detection (magic) if available.
    if MAGIC_AVAILABLE:
        detected_format = _detect_format_magic(file_path)
        print(f"[Agent 1] Format detected via python-magic: {detected_format}")
    else:
        detected_format = _detect_format_extension(file_path)
        print(f"[Agent 1] Format detected via extension fallback: {detected_format}")

    if detected_format == "unknown":
        return {
            "status": "error",
            "file_path": file_path,
            "detected_format": "unknown",
            "error_message": "File format not supported. Expected CSV, Excel, or JSON.",
            "dataframe": None,
        }

    # --- Step 3: Load into DataFrame ---
    # This is where actual parsing happens. We wrap it in try/except because
    # a file can have the right extension but be malformed inside.
    try:
        df = _load_dataframe(file_path, detected_format)
        print(f"[Agent 1] Loaded successfully. Shape: {df.shape}")
    except Exception as e:
        return {
            "status": "error",
            "file_path": file_path,
            "detected_format": detected_format,
            "error_message": f"Failed to parse file: {str(e)}",
            "dataframe": None,
        }

    # --- Step 4: Basic PII column scan ---
    # We check column names against known PII patterns.
    # This is a warning, not a block — the pipeline continues either way.
    columns = list(df.columns)
    pii_columns = _flag_pii_columns(columns)
    pii_flag = len(pii_columns) > 0

    if pii_flag:
        print(f"[Agent 1] ⚠ PII WARNING — Suspicious columns detected: {pii_columns}")
    else:
        print(f"[Agent 1] No PII column names detected.")

    # --- Step 5: Build and return output dict ---
    output = {
        "status": "ok",
        "file_path": file_path,
        "detected_format": detected_format,
        "row_count": len(df),
        "col_count": len(df.columns),
        "columns": columns,
        "pii_flag": pii_flag,
        "pii_columns": pii_columns,
        "dataframe": df,           # passed in memory — Agent 2 will use this directly
        "error_message": None,
    }

    print(f"[Agent 1] ✓ Intake complete. Status: ok")
    return output