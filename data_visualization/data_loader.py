import io
import zipfile
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class ExperimentData:
    """
    Holds all raw data for a single experiment (one C2 protein),
    fully loaded into memory — no filesystem access needed after this point.
    """
    protein_name: str
    color: str
    df_colocalization: pd.DataFrame
    # Maps filename → bytes, reconstructed from the uploaded zip
    intensity_files: dict = field(default_factory=dict)
    # Populated after validation
    colocalize_ids: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    
@dataclass 
class LoadResult:
    """
    Result of attempting to load one experiment.
    Separates success/failure cleanly so the UI can report errors per-experiment.
    """
    success: bool
    experiment: Optional[ExperimentData] = None
    error: Optional[str] = None

# ─────────────────────────────────────────────
# ZIP reconstruction
# ─────────────────────────────────────────────

def extract_zip_to_dict(zip_bytes: bytes) -> dict[str, bytes]:
    """
    Extracts a zip file (provided as raw bytes) into a flat dict of
    {filename: file_bytes}, ignoring directory structure and hidden files.
    
    Your existing code references files by name only (not subdirectories),
    so flattening is correct here.
    """
    file_dict = {}
    
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for zip_info in zf.infolist():
                # Skip directories and hidden/system files
                if zip_info.is_dir():
                    continue
                filename = zip_info.filename.split("/")[-1]  # strip any folder prefix
                if filename.startswith(".") or filename.startswith("__"):
                    continue
                if not filename.endswith(".csv"):
                    continue
                with zf.open(zip_info) as f:
                    file_dict[filename] = f.read()
                    
    except zipfile.BadZipFile:
        raise ValueError("The uploaded file is not a valid zip archive.")
    except Exception as e:
        raise ValueError(f"Failed to extract zip: {e}")
    
    return file_dict

# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────

def validate_colocalization_csv(df: pd.DataFrame) -> list[str]:
    """
    Checks that the colocalization CSV has the expected columns.
    Returns a list of error strings (empty = valid).
    """
    required_columns = {
        "COLOCALIZE_ID", "CHANNEL", "TRACK_START", "TRACK_STOP","TRACK_DURATION", "FILE_ID",
    }
    missing = required_columns - set(df.columns)
    if missing:
        return [f"Colocalization CSV is missing columns: {', '.join(sorted(missing))}"]
    return []


def validate_intensity_files(
    df_colocalization: pd.DataFrame,
    intensity_files: dict[str, bytes]
) -> tuple[list[str], list[str]]:
    """
    Checks that intensity profile CSVs exist for all colocalize IDs.
    Returns (errors, warnings) — errors are blocking, warnings are informational.
    """
    errors, warnings = [], []
    colocalize_ids = df_colocalization["COLOCALIZE_ID"].unique()
    
    missing_files = []
    for cid in colocalize_ids:
        for channel in [1, 2]:
            expected = f"Colocalized_ID_{cid}_C{channel}.csv"
            if expected not in intensity_files:
                missing_files.append(expected)
    
    if missing_files:
        # Warn rather than hard-error — your existing code already skips missing files
        if len(missing_files) <= 10:
            warnings.append(f"Missing intensity files: {', '.join(missing_files)}")
        else:
            warnings.append(
                f"{len(missing_files)} intensity files not found in zip. "
                f"First few: {', '.join(missing_files[:5])}, ..."
            )
    
    return errors, warnings

# ─────────────────────────────────────────────
# File reader
# ─────────────────────────────────────────────

def load_intensity_from_memory(
    intensity_files: dict[str, bytes],
    channel: int,
    colocalize_id: int
) -> Optional[pd.DataFrame]:
    """
	Load intensity files.
    """
    filename = f"Colocalized_ID_{colocalize_id}_C{channel}.csv"
    
    if filename not in intensity_files:
        logging.warning(f"File not found in memory: {filename}")
        return None
    
    try:
        return pd.read_csv(io.BytesIO(intensity_files[filename]))
    except Exception as e:
        logging.error(f"Failed to parse {filename}: {e}")
        return None

# ─────────────────────────────────────────────
# Main loading function — called from Streamlit
# ─────────────────────────────────────────────

def load_experiment(
    protein_name: str,
    color: str,
    colocalization_bytes: bytes,
    intensity_zip_bytes: bytes,
) -> LoadResult:
    """
    Loads and validates a single experiment from raw uploaded bytes.
    Returns a LoadResult with either a populated ExperimentData or an error message.
    
    This is the only function Streamlit needs to call per experiment.
    """
    
    # 1. Parse colocalization CSV
    try:
        df_colocalization = pd.read_csv(io.BytesIO(colocalization_bytes))
    except Exception as e:
        return LoadResult(success=False, error=f"Could not read colocalization CSV: {e}")
    
    # 2. Validate colocalization CSV structure
    csv_errors = validate_colocalization_csv(df_colocalization)
    if csv_errors:
        return LoadResult(success=False, error="\n".join(csv_errors))
    
    # 3. Extract intensity profiles from zip
    try:
        intensity_files = extract_zip_to_dict(intensity_zip_bytes)
    except ValueError as e:
        return LoadResult(success=False, error=str(e))
    
    # 4. Validate intensity files against colocalization IDs
    file_errors, file_warnings = validate_intensity_files(df_colocalization, intensity_files)
    if file_errors:
        return LoadResult(success=False, error="\n".join(file_errors))
    
    # 5. Build ExperimentData
    experiment = ExperimentData(
        protein_name=protein_name,
        color=color,
        df_colocalization=df_colocalization,
        intensity_files=intensity_files,
        colocalize_ids=list(df_colocalization["COLOCALIZE_ID"].unique()),
        warnings=file_warnings,
    )
    
    return LoadResult(success=True, experiment=experiment)


def load_all_experiments(experiment_inputs: list[dict]) -> tuple[list[ExperimentData], list[str]]:
    """
    Loads multiple experiments. Returns (successful_experiments, error_messages).
    Partial success is allowed — if 2 of 3 experiments load, you get those 2.
    
    Each dict in experiment_inputs should have:
        protein_name: str
        color: str
        colocalization_bytes: bytes
        intensity_zip_bytes: bytes
    """
    loaded = []
    errors = []
    
    for entry in experiment_inputs:
        result = load_experiment(
            protein_name=entry["protein_name"],
            color=entry["color"],
            colocalization_bytes=entry["colocalization_bytes"],
            intensity_zip_bytes=entry["intensity_zip_bytes"],
        )
        if result.success:
            loaded.append(result.experiment)
            if result.experiment.warnings:
                # Surface warnings but don't block
                for w in result.experiment.warnings:
                    errors.append(f"⚠️ {entry['protein_name']}: {w}")
        else:
            errors.append(f"❌ {entry['protein_name']}: {result.error}")
    
    return loaded, errors
