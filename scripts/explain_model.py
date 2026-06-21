"""Create SHAP global explanations for the selected model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from abalone.config import FIGURES_DIR, MODEL_PATH, RAW_DATA_PATH, REPORTS_DIR, ensure_output_directories
from abalone.features import load_training_data, make_features


def create_shap_report(data_path: str | Path, sample_size: int = 500) -> None:
    ensure_output_directories()
    data = load_training_data(data_path)
    features = make_features(data)
    sample = features.sample(min(sample_size, len(features)), random_state=42)
    pipeline = joblib.load(MODEL_PATH)
    transformed = pipeline.named_steps["preprocess"].transform(sample)
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    estimator = pipeline.named_steps["model"]

    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(transformed)
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 0]

    importance = pd.DataFrame(
        {
            "feature": names,
            "mean_absolute_shap": np.abs(values).mean(axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    importance.to_csv(REPORTS_DIR / "shap_feature_importance.csv", index=False)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(values, transformed, feature_names=names, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()

    top_feature_index = int(np.argmax(np.abs(values).mean(axis=0)))
    plt.figure(figsize=(7, 5))
    shap.dependence_plot(top_feature_index, values, transformed, feature_names=names, interaction_index=None, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_dependence_top_feature.png", dpi=160, bbox_inches="tight")
    plt.close()

    top_features = importance.head(10)
    report = [
        "# SHAP explanation",
        "",
        f"Global explanation calculated from a reproducible sample of {len(sample):,} records.",
        "Mean absolute SHAP values quantify average contribution magnitude in predicted years; they do not establish causation.",
        "",
        "## Most influential transformed features",
        "",
    ]
    report.extend(f"- {row.feature}: {row.mean_absolute_shap:.4f} years" for row in top_features.itertuples())
    (REPORTS_DIR / "shap_explanation.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(RAW_DATA_PATH))
    parser.add_argument("--sample-size", type=int, default=500)
    args = parser.parse_args()
    create_shap_report(args.data, sample_size=args.sample_size)
