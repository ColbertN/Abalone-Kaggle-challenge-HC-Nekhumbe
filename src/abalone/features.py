"""Input validation and leakage-safe feature construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import RAW_DATA_PATH

AGE_TARGET = "age_years"
RING_COLUMN = "Rings"
BASE_FEATURES = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
]
NUMERIC_BASE_FEATURES = [column for column in BASE_FEATURES if column != "Sex"]
ENGINEERED_FEATURES = [
    "length_to_diameter",
    "whole_weight_per_length",
    "meat_yield",
    "shell_weight_ratio",
    "condition_index",
]
NUMERIC_FEATURES = NUMERIC_BASE_FEATURES + ENGINEERED_FEATURES
CATEGORICAL_FEATURES = ["Sex"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
VALID_SEX_VALUES = {"M", "F", "I"}


def read_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read and validate a data set with the eight measurement features.

    The target is deliberately not required here so the same function works for
    inference data. The original Kaggle names, including spaces, are retained to
    make the data provenance obvious.
    """
    frame = pd.read_csv(path)
    missing = sorted(set(BASE_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    frame = frame.copy()
    frame["Sex"] = frame["Sex"].astype("string").str.strip().str.upper()
    for column in NUMERIC_BASE_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_training_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load training data and express the competition target in years."""
    frame = read_raw_data(path)
    if RING_COLUMN not in frame.columns:
        raise ValueError(f"Training data must contain '{RING_COLUMN}'.")
    frame[RING_COLUMN] = pd.to_numeric(frame[RING_COLUMN], errors="coerce")
    if frame[RING_COLUMN].isna().any():
        raise ValueError("Rings contains missing or non-numeric values.")
    frame[AGE_TARGET] = frame[RING_COLUMN] + 1.5
    return frame


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create biologically interpretable ratios without using the target.

    Ratios capture shell/body allocation and condition while retaining the raw
    measures. Division by zero becomes missing and is handled explicitly by the
    model's input check rather than silently creating infinite values.
    """
    features = frame.loc[:, BASE_FEATURES].copy()
    safe_length = features["Length"].replace(0, np.nan)
    safe_diameter = features["Diameter"].replace(0, np.nan)
    safe_whole_weight = features["Whole weight"].replace(0, np.nan)

    features["length_to_diameter"] = features["Length"] / safe_diameter
    features["whole_weight_per_length"] = features["Whole weight"] / safe_length
    features["meat_yield"] = features["Shucked weight"] / safe_whole_weight
    features["shell_weight_ratio"] = features["Shell weight"] / safe_whole_weight
    features["condition_index"] = (
        features["Whole weight"] / (safe_length * safe_diameter * features["Height"].replace(0, np.nan))
    )
    return features.loc[:, MODEL_FEATURES]


def input_quality_summary(frame: pd.DataFrame) -> dict[str, object]:
    """Return concise, machine-readable checks for raw model inputs."""
    numerical = frame[NUMERIC_BASE_FEATURES]
    invalid_sex = (~frame["Sex"].isin(VALID_SEX_VALUES)).sum()
    return {
        "records": int(len(frame)),
        "duplicate_rate": float(frame.duplicated().mean()),
        "missing_rate": {column: float(frame[column].isna().mean()) for column in BASE_FEATURES},
        "non_positive_measurements": {
            column: int((numerical[column] <= 0).sum()) for column in NUMERIC_BASE_FEATURES
        },
        "unexpected_sex_values": int(invalid_sex),
    }


def build_preprocessor() -> ColumnTransformer:
    """Encode categoricals inside the modelling pipeline to prevent leakage."""
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )
