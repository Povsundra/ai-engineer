"""
Encoder–decoder LSTM baseline for electricity load forecasting with 13 feature inputs.

The script can be run as a CLI:
    python eletricity_load_forecasting/model.py --epochs 3
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_synthetic_load(
    rows: int = 3000, feature_count: int = 13, target_col: str = "load"
) -> pd.DataFrame:
    """
    Create a lightweight synthetic dataset to smoke-test the pipeline.
    The target series is a mixture of daily/weekly/annual seasonality with noise.
    """
    rng = np.random.default_rng(seed=42)
    time = np.arange(rows)
    daily = 0.6 * np.sin(2 * np.pi * time / 24)
    weekly = 0.3 * np.sin(2 * np.pi * time / (24 * 7))
    annual = 0.1 * np.cos(2 * np.pi * time / (24 * 365))
    weather = 0.2 * np.sin(2 * np.pi * time / (24 * 30))
    trend = 0.0002 * time

    base_load = 1.0 + daily + weekly + annual + weather + trend
    noise = rng.normal(0.0, 0.05, size=rows)
    load = base_load + noise

    columns: List[str] = []
    feature_matrix = []
    # Create (feature_count - 1) auxiliary signals.
    for idx in range(feature_count - 1):
        phase = rng.uniform(0, 2 * np.pi)
        feature_matrix.append(
            0.5 * np.sin(2 * np.pi * time / (24 * (idx + 2)) + phase)
            + 0.05 * rng.normal(size=rows)
        )
        columns.append(f"f{idx + 1}")

    df = pd.DataFrame(np.stack(feature_matrix, axis=1), columns=columns)
    df[target_col] = load
    return df


def _ensure_feature_list(
    df: pd.DataFrame, feature_cols: Optional[Sequence[str]], target_col: str
) -> List[str]:
    """
    Returns exactly 13 feature column names, ensuring the target column is included.
    Raises ValueError if the constraints are not satisfied.
    """
    if feature_cols is None or len(feature_cols) == 0:
        feature_cols = list(df.columns[:13])

    feature_cols = list(dict.fromkeys(feature_cols))  # preserve order, drop duplicates

    if target_col not in feature_cols:
        raise ValueError(
            f"Target column '{target_col}' must be included among the 13 features."
        )
    if len(feature_cols) != 13:
        raise ValueError(
            f"Exactly 13 feature columns are required (got {len(feature_cols)})."
        )
    return feature_cols


def _split_indices(length: int, train_frac: float = 0.7, val_frac: float = 0.15) -> Tuple[int, int]:
    train_end = int(length * train_frac)
    val_end = int(length * (train_frac + val_frac))
    return train_end, val_end


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    target_scaler: MinMaxScaler


class ElectricityLoadDataset(Dataset):
    """
    Converts a multivariate time series into encoder/decoder windows.
    """

    def __init__(
        self,
        feature_data: np.ndarray,
        target_data: np.ndarray,
        input_window: int,
        output_window: int,
        target_feature_idx: int,
    ) -> None:
        if feature_data.shape[1] <= target_feature_idx:
            raise ValueError("target_feature_idx is out of bounds for feature_data.")
        max_start = feature_data.shape[0] - input_window - output_window
        if max_start < 0:
            raise ValueError(
                "Not enough data to build sequences. "
                f"Need at least {input_window + output_window} rows."
            )

        samples: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        starts: List[np.ndarray] = []

        for idx in range(max_start + 1):
            x = feature_data[idx : idx + input_window]
            y = target_data[idx + input_window : idx + input_window + output_window]
            start_token = x[-1, target_feature_idx]
            samples.append(x)
            targets.append(y)
            starts.append(start_token)

        self.features = torch.tensor(np.stack(samples), dtype=torch.float32)
        self.labels = torch.tensor(np.stack(targets), dtype=torch.float32)
        self.starts = torch.tensor(np.stack(starts), dtype=torch.float32)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx], self.starts[idx]


class EncoderDecoderLSTM(nn.Module):
    """
    Encoder–decoder with shared hidden state for sequence forecasting.
    Decoder operates autoregressively with optional teacher forcing.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_window: int,
    ) -> None:
        super().__init__()
        decoder_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=decoder_dropout,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=decoder_dropout,
            batch_first=True,
        )
        self.output_window = output_window
        self.projection = nn.Linear(hidden_size, 1)

    def forward(
        self,
        encoder_inputs: torch.Tensor,
        decoder_targets: Optional[torch.Tensor],
        start_tokens: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        batch_size = encoder_inputs.size(0)
        _, (hidden, cell) = self.encoder(encoder_inputs)

        decoder_input = start_tokens.view(batch_size, 1, 1)
        outputs = []
        for step in range(self.output_window):
            decoder_out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            step_pred = self.projection(decoder_out)  # (batch, 1, 1)
            outputs.append(step_pred)

            use_teacher = (
                decoder_targets is not None
                and torch.rand(1).item() < teacher_forcing_ratio
                and step < decoder_targets.shape[1]
            )
            next_token = (
                decoder_targets[:, step] if use_teacher else step_pred.squeeze(1).squeeze(1)
            )
            decoder_input = next_token.view(batch_size, 1, 1)

        return torch.cat(outputs, dim=1).squeeze(-1)


def prepare_dataloaders(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    input_window: int,
    output_window: int,
    batch_size: int,
) -> DataBundle:
    feature_cols = _ensure_feature_list(df, feature_cols, target_col)
    target_feature_idx = feature_cols.index(target_col)

    train_end, val_end = _split_indices(len(df))
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    train_features = df.iloc[:train_end][feature_cols]
    feature_scaler.fit(train_features)

    train_target = df.iloc[:train_end][[target_col]]
    target_scaler.fit(train_target)

    scaled_features = feature_scaler.transform(df[feature_cols])
    scaled_target = target_scaler.transform(df[[target_col]])[:, 0]

    def _make_loader(feature_slice: np.ndarray, target_slice: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = ElectricityLoadDataset(
            feature_data=feature_slice,
            target_data=target_slice,
            input_window=input_window,
            output_window=output_window,
            target_feature_idx=target_feature_idx,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    train_loader = _make_loader(scaled_features[:train_end], scaled_target[:train_end], shuffle=True)
    val_loader = _make_loader(
        scaled_features[train_end:val_end],
        scaled_target[train_end:val_end],
        shuffle=False,
    )
    test_loader = _make_loader(scaled_features[val_end:], scaled_target[val_end:], shuffle=False)

    return DataBundle(train=train_loader, val=val_loader, test=test_loader, target_scaler=target_scaler)


def train_one_epoch(
    model: EncoderDecoderLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    teacher_forcing_ratio: float,
) -> float:
    model.train()
    running = 0.0
    total = 0
    for batch in loader:
        features, targets, starts = batch
        features = features.to(device)
        targets = targets.to(device)
        starts = starts.to(device)

        optimizer.zero_grad()
        preds = model(
            encoder_inputs=features,
            decoder_targets=targets,
            start_tokens=starts,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        loss = criterion(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = features.size(0)
        running += loss.item() * batch_size
        total += batch_size
    return running / max(total, 1)


def evaluate_loss(
    model: EncoderDecoderLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    running = 0.0
    total = 0
    with torch.no_grad():
        for features, targets, starts in loader:
            features = features.to(device)
            targets = targets.to(device)
            starts = starts.to(device)
            preds = model(
                encoder_inputs=features,
                decoder_targets=None,
                start_tokens=starts,
                teacher_forcing_ratio=0.0,
            )
            loss = criterion(preds, targets)
            batch_size = features.size(0)
            running += loss.item() * batch_size
            total += batch_size
    return running / max(total, 1)


def evaluate_metrics(
    model: EncoderDecoderLSTM,
    loader: DataLoader,
    device: torch.device,
    target_scaler: MinMaxScaler,
) -> Dict[str, float]:
    model.eval()
    preds_list: List[np.ndarray] = []
    trues_list: List[np.ndarray] = []
    with torch.no_grad():
        for features, targets, starts in loader:
            features = features.to(device)
            starts = starts.to(device)
            pred = model(
                encoder_inputs=features,
                decoder_targets=None,
                start_tokens=starts,
                teacher_forcing_ratio=0.0,
            )
            preds_list.append(pred.cpu().numpy())
            trues_list.append(targets.numpy())

    preds_scaled = np.concatenate(preds_list, axis=0)
    trues_scaled = np.concatenate(trues_list, axis=0)

    preds_flat = preds_scaled.reshape(-1, 1)
    trues_flat = trues_scaled.reshape(-1, 1)

    preds_unscaled = target_scaler.inverse_transform(preds_flat).reshape(preds_scaled.shape)
    trues_unscaled = target_scaler.inverse_transform(trues_flat).reshape(trues_scaled.shape)

    mae = mean_absolute_error(trues_unscaled, preds_unscaled)
    mse = mean_squared_error(trues_unscaled, preds_unscaled)
    rmse = float(np.sqrt(mse))
    mape = float(
        np.mean(
            np.abs((trues_unscaled - preds_unscaled) / np.clip(np.abs(trues_unscaled), 1e-3, None))
        )
        * 100.0
    )
    return {"mae": mae, "rmse": rmse, "mape": mape}


def run_training(args: argparse.Namespace) -> None:
    device = _default_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.data_path:
        df = pd.read_csv(args.data_path)
    else:
        df = generate_synthetic_load(feature_count=13, rows=3000, target_col=args.target_col)

    features = _ensure_feature_list(df, args.feature_cols, args.target_col)
    dataloaders = prepare_dataloaders(
        df=df,
        feature_cols=features,
        target_col=args.target_col,
        input_window=args.input_window,
        output_window=args.output_window,
        batch_size=args.batch_size,
    )

    model = EncoderDecoderLSTM(
        input_size=len(features),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_window=args.output_window,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=dataloaders.train,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            teacher_forcing_ratio=args.teacher_forcing,
        )
        val_loss = evaluate_loss(
            model=model,
            loader=dataloaders.val,
            criterion=criterion,
            device=device,
        )
        print(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

    test_metrics = evaluate_metrics(
        model=model,
        loader=dataloaders.test,
        device=device,
        target_scaler=dataloaders.target_scaler,
    )
    print(
        f"Test MAE: {test_metrics['mae']:.4f} | "
        f"RMSE: {test_metrics['rmse']:.4f} | "
        f"MAPE: {test_metrics['mape']:.2f}%"
    )

    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_window": args.input_window,
                "output_window": args.output_window,
                "feature_cols": features,
            },
            args.save_path,
        )
        print(f"Saved model checkpoint to {args.save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encoder–decoder LSTM for electricity load forecasting (13 features)."
    )
    parser.add_argument("--data_path", type=str, default=None, help="CSV dataset path.")
    parser.add_argument(
        "--feature_cols",
        nargs="*",
        help="Space-separated list of exactly 13 feature columns (must include target column).",
    )
    parser.add_argument("--target_col", type=str, default="load", help="Target column name.")
    parser.add_argument("--input_window", type=int, default=48, help="Encoder lookback length.")
    parser.add_argument("--output_window", type=int, default=12, help="Forecast horizon length.")
    parser.add_argument("--hidden_size", type=int, default=128, help="LSTM hidden size.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout between LSTM layers.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument(
        "--teacher_forcing",
        type=float,
        default=0.5,
        help="Probability of teacher forcing during decoding.",
    )
    parser.add_argument("--save_path", type=str, default=None, help="Optional checkpoint path.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
