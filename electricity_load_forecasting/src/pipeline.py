from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURE_COLUMNS = [
    "temperature_c",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
]
REQUIRED_COLUMNS = {"timestamp", "load_mw", "temperature_c"}
MINIMUM_TEMPERATURE_SAMPLES = 24


def generate_synthetic_dataset(hours: int = 24 * 90, seed: int = 42) -> pd.DataFrame:
    """
    Create a lightweight synthetic hourly load dataset with weather and calendar effects.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=hours, freq="h")

    annual_temp = 18 + 10 * np.sin(2 * np.pi * timestamps.dayofyear / 365)
    diurnal_temp = 6 * np.sin(2 * np.pi * timestamps.hour / 24)
    temperature_c = annual_temp + diurnal_temp + rng.normal(0, 1.5, hours)

    weekend = (timestamps.dayofweek >= 5).astype(int)
    base_load = 420 + 55 * np.sin(2 * np.pi * timestamps.hour / 24 - 0.5)
    weather_effect = 2.2 * (temperature_c - 18)
    weekend_effect = weekend * -35
    noise = rng.normal(0, 12, hours)
    load_mw = base_load + weather_effect + weekend_effect + noise + 60

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature_c": temperature_c.round(2),
            "load_mw": load_mw.round(2),
        }
    )


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_dataset(data_path: Optional[Path]) -> pd.DataFrame:
    """
    Load a CSV dataset or fall back to a synthetic sample.
    Expected columns: timestamp, load_mw, temperature_c
    """
    if data_path and data_path.exists():
        df = pd.read_csv(data_path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Dataset is missing required columns: {missing_str}")
        df = _ensure_datetime(df)
        return df.sort_values("timestamp").reset_index(drop=True)

    return generate_synthetic_dataset()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    stamped = _ensure_datetime(df)
    stamped["hour"] = stamped["timestamp"].dt.hour
    stamped["day_of_week"] = stamped["timestamp"].dt.dayofweek
    stamped["month"] = stamped["timestamp"].dt.month
    stamped["is_weekend"] = (stamped["day_of_week"] >= 5).astype(int)
    stamped["hour_sin"] = np.sin(2 * np.pi * stamped["hour"] / 24)
    stamped["hour_cos"] = np.cos(2 * np.pi * stamped["hour"] / 24)
    return stamped


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    enriched = add_time_features(df)
    X = enriched[FEATURE_COLUMNS].copy()
    y = enriched["load_mw"].copy()
    return X, y


def chronological_train_test_split(
    X: pd.DataFrame, y: pd.Series, test_hours: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if test_hours <= 0 or test_hours >= len(y):
        raise ValueError("test_hours must be positive and smaller than the dataset size.")
    split_idx = len(y) - test_hours
    return (
        X.iloc[:split_idx],
        X.iloc[split_idx:],
        y.iloc[:split_idx],
        y.iloc[split_idx:],
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series, seed: int) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(math.sqrt(mean_squared_error(y_test, preds)))

    y_true = np.asarray(y_test)
    non_zero_mask = np.abs(y_true) > 0
    if non_zero_mask.any():
        actual_masked = y_true[non_zero_mask]
        preds_masked = preds[non_zero_mask]
        mape = float(np.mean(np.abs(actual_masked - preds_masked) / np.abs(actual_masked)))
    else:
        mape = float("nan")

    r2 = float(model.score(X_test, y_test))
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def build_future_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    last_timestamp = pd.to_datetime(df_sorted["timestamp"]).iloc[-1]

    base_temperatures = (
        df_sorted["temperature_c"]
        .tail(max(horizon, MINIMUM_TEMPERATURE_SAMPLES))
        .to_numpy()
    )  # ensure at least one full day of temperature pattern to tile across the forecast horizon
    if base_temperatures.size == 0:
        base_temperatures = np.array([18.0])

    repeats = math.ceil(horizon / len(base_temperatures))
    repeated_temps = np.tile(base_temperatures, repeats)[:horizon]

    future_timestamps = pd.date_range(last_timestamp + pd.Timedelta(hours=1), periods=horizon, freq="h")
    future_df = pd.DataFrame(
        {
            "timestamp": future_timestamps,
            "temperature_c": repeated_temps,
        }
    )
    enriched = add_time_features(future_df)
    return enriched


def run_pipeline(
    data_path: Optional[Path],
    output_dir: Path,
    test_hours: int,
    horizon: int,
    seed: int,
) -> dict:
    df = load_dataset(data_path)

    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = chronological_train_test_split(X, y, test_hours)
    model = train_model(X_train, y_train, seed)
    metrics = evaluate(model, X_test, y_test)

    future_enriched = build_future_features(df, horizon)
    forecast = pd.DataFrame(
        {
            "timestamp": future_enriched["timestamp"],
            "temperature_c": future_enriched["temperature_c"],
            "predicted_load_mw": model.predict(future_enriched[FEATURE_COLUMNS]),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = output_dir / "forecast.csv"
    history_path = output_dir / "history_sample.csv"

    forecast.to_csv(forecast_path, index=False)
    df.tail(test_hours).to_csv(history_path, index=False)

    return {
        "metrics": metrics,
        "forecast_path": forecast_path,
        "history_path": history_path,
        "model": model,
        "history_tail": df.tail(test_hours),
    }
