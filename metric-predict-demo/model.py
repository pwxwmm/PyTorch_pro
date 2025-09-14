# Author / 作者: mmwei3 (韦蒙蒙)
# Date / 日期: 2025-09-13
# Weather / 天气: Sunny / 晴
# Project: Metric Forecast Demo (PyTorch LSTM)

import torch
import torch.nn as nn


class LSTMForecast(nn.Module):
    """Simple LSTM forecaster for univariate time series.

    Inputs shape: (batch_size, seq_len, 1)
    Outputs shape: (batch_size, 1)
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1)
        output, _ = self.lstm(x)
        # Take the last time step
        last_hidden = output[:, -1, :]
        y_hat = self.readout(last_hidden)
        return y_hat
