"""Tune candidate models, select the best one, and save reproducible artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abalone.config import RAW_DATA_PATH
from abalone.modeling import train_and_compare


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(RAW_DATA_PATH))
    parser.add_argument("--iterations", type=int, default=4, help="Random-search draws per tuned model.")
    args = parser.parse_args()
    summary = train_and_compare(args.data, n_iter=args.iterations)
    print(f"Best model: {summary['best_model']}")
    print(summary["best_test_metrics"])
