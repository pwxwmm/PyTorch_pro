# Author / 作者: mmwei3 (韦蒙蒙)
# Date / 日期: 2025-09-13
# Weather / 天气: Sunny / 晴
# Project: Metric Forecast Demo (Training)

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import LSTMForecast


@dataclass
class TrainConfig:
    seq_len: int = 30
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    train_split: float = 0.8
    data_csv: str = os.path.join("data", "metrics.csv")
    save_dir: str = os.path.join("saved_model")
    seed: int = 42


class MetricDataset(Dataset):
    def __init__(self, values: np.ndarray, seq_len: int) -> None:
        self.values = values.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.values) - self.seq_len)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.values[idx : idx + self.seq_len]
        y = self.values[idx + self.seq_len]
        # (T, 1), scalar target
        return torch.from_numpy(x).unsqueeze(-1), torch.tensor([y], dtype=torch.float32)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(cfg: TrainConfig) -> None:
    os.makedirs(os.path.dirname(cfg.data_csv), exist_ok=True)
    os.makedirs(cfg.save_dir, exist_ok=True)


def maybe_generate_synthetic_data(cfg: TrainConfig, num_points: int = 2000) -> None:
    if os.path.exists(cfg.data_csv):
        return
    x = np.linspace(0, 60, num_points, dtype=np.float32)
    # Trend + seasonality + noise
    y = (
        0.02 * x
        + np.sin(0.6 * x)
        + 0.5 * np.sin(0.05 * x)
        + np.random.normal(0.0, 0.1, size=num_points)
    )
    df = pd.DataFrame(
        {"timestamp": np.arange(num_points), "value": y.astype(np.float32)}
    )
    df.to_csv(cfg.data_csv, index=False)


def load_series(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    # Accept common column names
    if "value" in df.columns:
        values = df["value"].to_numpy(dtype=np.float32)
    elif "y" in df.columns:
        values = df["y"].to_numpy(dtype=np.float32)
    else:
        # Fallback to the last column
        values = df.iloc[:, -1].to_numpy(dtype=np.float32)
    return values


def zscore_normalize(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mean = float(values.mean())
    std = float(values.std() + 1e-8)
    return (values - mean) / std, mean, std


def inverse_zscore(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return x * std + mean


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / max(1, len(loader.dataset))


def eval_one_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / max(1, len(loader.dataset))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=TrainConfig.seq_len)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--hidden_size", type=int, default=TrainConfig.hidden_size)
    parser.add_argument("--num_layers", type=int, default=TrainConfig.num_layers)
    parser.add_argument("--dropout", type=float, default=TrainConfig.dropout)
    parser.add_argument("--data_csv", type=str, default=TrainConfig.data_csv)
    parser.add_argument("--save_dir", type=str, default=TrainConfig.save_dir)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = parser.parse_args()

    cfg = TrainConfig(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        data_csv=args.data_csv,
        save_dir=args.save_dir,
        seed=args.seed,
    )

    ensure_dirs(cfg)
    set_seed(cfg.seed)
    maybe_generate_synthetic_data(cfg)

    values = load_series(cfg.data_csv)
    values_norm, mean, std = zscore_normalize(values)

    split_idx = int(len(values_norm) * cfg.train_split)
    train_values = values_norm[:split_idx]
    val_values = values_norm[split_idx - cfg.seq_len :]

    train_ds = MetricDataset(train_values, seq_len=cfg.seq_len)
    val_ds = MetricDataset(val_values, seq_len=cfg.seq_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecast(
        input_size=1,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        output_size=1,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val = math.inf
    best_path = os.path.join(cfg.save_dir, "forecast.pt")
    meta_path = os.path.join(cfg.save_dir, "metadata.json")

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = eval_one_epoch(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict()}, best_path)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "seq_len": cfg.seq_len,
                        "mean": mean,
                        "std": std,
                        "model": {
                            "hidden_size": cfg.hidden_size,
                            "num_layers": cfg.num_layers,
                            "dropout": cfg.dropout,
                        },
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    print(f"Saved best model to {best_path}")


if __name__ == "__main__":
    main()
