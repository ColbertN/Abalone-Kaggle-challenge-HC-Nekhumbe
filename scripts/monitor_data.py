"""Monitor a scoring batch for input quality, drift, and optional performance."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib

from abalone.config import MODEL_PATH, MONITORING_DIR, REFERENCE_PROFILE_PATH, ensure_output_directories
from abalone.evaluation import regression_metrics
from abalone.features import RING_COLUMN, input_quality_summary, make_features, read_raw_data
from abalone.monitoring import assess_drift


def monitor(input_path: str | Path, output_path: str | Path | None = None) -> Path:
    ensure_output_directories()
    raw = read_raw_data(input_path)
    reference = json.loads(REFERENCE_PROFILE_PATH.read_text(encoding="utf-8"))
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_path": str(input_path),
        "input_quality": input_quality_summary(raw),
        "drift": assess_drift(make_features(raw), reference),
    }
    if RING_COLUMN in raw.columns:
        model = joblib.load(MODEL_PATH)
        actual = raw[RING_COLUMN] + 1.5
        report["performance"] = regression_metrics(actual, model.predict(make_features(raw)))
    else:
        report["performance"] = "Unavailable: input has no Rings column. Join delayed labels to monitor accuracy."

    path = Path(output_path) if output_path else MONITORING_DIR / "latest_monitoring_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Monitoring status: {report['drift']['overall_status']}")
    print(f"Wrote {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV batch to validate and compare with training data.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    monitor(args.input, args.output)
