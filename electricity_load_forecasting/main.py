from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a quick electricity load forecasting baseline and emit a 24h forecast."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional path to a CSV with columns: timestamp, load_mw, temperature_c. Defaults to a synthetic sample.",
    )
    parser.add_argument(
        "--test-hours",
        type=int,
        default=72,
        help="Number of trailing hours to reserve for evaluation.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Forecast horizon in hours.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "artifacts"),
        help="Where to write forecast and history sample CSVs.",
    )
    return parser.parse_args()


def render_summary(metrics: dict, forecast_sample: pd.DataFrame) -> None:
    print("\nModel evaluation on hold-out window")
    print("-----------------------------------")
    print(f"MAE : {metrics['mae']:.2f} MW")
    print(f"RMSE: {metrics['rmse']:.2f} MW")
    print(f"MAPE: {metrics['mape'] * 100:.2f}%")
    print(f"R^2 : {metrics['r2']:.3f}")

    print("\nNext horizon forecast (first 5 rows)")
    print("------------------------------------")
    print(forecast_sample.head())


def main(args: Optional[argparse.Namespace] = None) -> None:
    parsed = args or parse_args()
    data_path = Path(parsed.data) if parsed.data else None
    output_dir = Path(parsed.output_dir)

    results = run_pipeline(
        data_path=data_path,
        output_dir=output_dir,
        test_hours=parsed.test_hours,
        horizon=parsed.horizon,
        seed=parsed.seed,
    )

    render_summary(results["metrics"], pd.read_csv(results["forecast_path"]))
    print(f"\nSaved forecast to: {results['forecast_path']}")
    print(f"Saved history sample to: {results['history_path']}")


if __name__ == "__main__":
    main()
