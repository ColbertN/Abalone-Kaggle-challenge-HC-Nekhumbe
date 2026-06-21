"""Generate meaningful EDA reports and figures for the abalone data set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from abalone.config import FIGURES_DIR, RAW_DATA_PATH, REPORTS_DIR, ensure_output_directories
from abalone.features import AGE_TARGET, NUMERIC_BASE_FEATURES, input_quality_summary, load_training_data


def _save_figure(figure: plt.Figure, name: str) -> None:
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(figure)


def create_eda(data_path: str | Path) -> None:
    ensure_output_directories()
    data = load_training_data(data_path)
    quality = input_quality_summary(data)
    numeric = data[NUMERIC_BASE_FEATURES + [AGE_TARGET]]
    summary = numeric.describe().T
    summary["missing_count"] = numeric.isna().sum()
    summary.to_csv(REPORTS_DIR / "data_summary.csv")

    correlations = numeric.corr(numeric_only=True)[AGE_TARGET].sort_values(key=abs, ascending=False)
    correlations.rename("correlation_with_age_years").to_csv(REPORTS_DIR / "age_correlations.csv")

    figure, axis = plt.subplots(figsize=(8, 4.5))
    sns.histplot(data=data, x=AGE_TARGET, bins=26, kde=True, ax=axis, color="#2a9d8f")
    axis.set(title="Age distribution", xlabel="Age (years)", ylabel="Count")
    _save_figure(figure, "age_distribution.png")

    figure, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(numeric.corr(numeric_only=True), cmap="vlag", center=0, annot=True, fmt=".2f", ax=axis)
    axis.set_title("Measurement and age correlations")
    _save_figure(figure, "correlation_heatmap.png")

    figure, axis = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=data, x="Sex", y=AGE_TARGET, hue="Sex", legend=False, order=["I", "M", "F"], ax=axis)
    axis.set(title="Age by sex category", xlabel="Sex category", ylabel="Age (years)")
    _save_figure(figure, "age_by_sex.png")

    top_features = [feature for feature in correlations.index if feature != AGE_TARGET][:6]
    figure, axes = plt.subplots(2, 3, figsize=(13, 7))
    for axis, feature in zip(axes.ravel(), top_features, strict=True):
        sns.regplot(data=data, x=feature, y=AGE_TARGET, scatter_kws={"alpha": 0.25, "s": 10}, line_kws={"color": "#e76f51"}, ax=axis)
        axis.set_title(f"{feature} vs age")
    _save_figure(figure, "top_measurement_relationships.png")

    sex_counts = data["Sex"].value_counts(dropna=False).rename_axis("Sex").rename("records")
    lines = [
        "# Abalone data profile",
        "",
        f"- Records: {len(data):,}",
        f"- Columns in source data: {len(data.columns) - 1}",
        "- Prediction target: age in years, calculated as `Rings + 1.5`.",
        f"- Duplicate-row rate: {quality['duplicate_rate']:.2%}",
        f"- Unexpected sex values: {quality['unexpected_sex_values']}",
        "",
        "## Target and cohort",
        "",
        f"- Age range: {data[AGE_TARGET].min():.1f} to {data[AGE_TARGET].max():.1f} years.",
        f"- Median age: {data[AGE_TARGET].median():.1f} years.",
        "- Sex counts: " + ", ".join(f"{index}={value}" for index, value in sex_counts.items()),
        "",
        "## Data-quality findings",
        "",
        "- Missing values: " + ", ".join(f"{column}={rate:.2%}" for column, rate in quality["missing_rate"].items()),
        "- Non-positive measurements: " + ", ".join(f"{column}={count}" for column, count in quality["non_positive_measurements"].items()),
        "",
        "## Strongest linear associations with age",
        "",
    ]
    lines.extend(f"- {feature}: {value:.3f}" for feature, value in correlations.drop(AGE_TARGET).head(5).items())
    lines.extend(
        [
            "",
            "## Modelling implications",
            "",
            "- Age is right-skewed, so MAE and RMSE in years are reported alongside R².",
            "- Measurements are highly correlated, so the workflow uses tree ensembles and avoids target-informed feature filtering.",
            "- No observations are dropped as outliers before the train/test split; extreme animals are part of the prediction problem.",
        ]
    )
    (REPORTS_DIR / "data_profile.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(RAW_DATA_PATH))
    args = parser.parse_args()
    create_eda(args.data)
