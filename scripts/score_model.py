"""Score new abalone measurements with the selected model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np

from abalone.config import METADATA_PATH, MODEL_PATH, REPORTS_DIR, ensure_output_directories
from abalone.evaluation import regression_metrics
from abalone.features import AGE_TARGET, RING_COLUMN, make_features, read_raw_data


def score(input_path: str | Path, output_path: str | Path) -> None:
    ensure_output_directories()
    raw = read_raw_data(input_path)
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    predictions = model.predict(make_features(raw))
    interval = metadata["approximate_95pct_interval_half_width_years"]
    output = raw.copy()
    output["predicted_age_years"] = predictions
    output["prediction_interval_lower_years"] = np.maximum(0, predictions - interval)
    output["prediction_interval_upper_years"] = predictions + interval
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "records_scored": int(len(output)),
        "prediction_interval": "Approximate 95% interval from held-out residual standard deviation.",
    }
    if RING_COLUMN in raw.columns:
        actual = raw[RING_COLUMN] + 1.5
        metrics = regression_metrics(actual, predictions)
        summary["labelled_batch_metrics"] = metrics
        print(json.dumps(metrics, indent=2))
    else:
        summary["labelled_batch_metrics"] = "Unavailable: input has no Rings column."
    (REPORTS_DIR / "scoring_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(output):,} predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV containing the eight raw measurement columns.")
    parser.add_argument("--output", default=str(ROOT / "submissions" / "abalone_predictions.csv"))
    args = parser.parse_args()
    score(args.input, args.output)
