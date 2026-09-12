from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LossConfig:
    loss_name: str = "mse"
    alpha: float = 0.7
    beta: float = 0.3
    peak_lambda: float = 4.0
    peak_gamma: float = 2.0


class MSEMAELoss(nn.Module):
    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((pred - target) ** 2)
        mae = torch.mean(torch.abs(pred - target))
        return self.alpha * mse + self.beta * mae


class PeakWeightedMSELoss(nn.Module):
    def __init__(self, peak_lambda: float = 4.0, peak_gamma: float = 2.0):
        super().__init__()
        self.peak_lambda = peak_lambda
        self.peak_gamma = peak_gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = 1.0 + self.peak_lambda * torch.pow(target, self.peak_gamma)
        return torch.mean(weights * (pred - target) ** 2)


class PeakWeightedMSEMAELoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        peak_lambda: float = 4.0,
        peak_gamma: float = 2.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.peak_lambda = peak_lambda
        self.peak_gamma = peak_gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = 1.0 + self.peak_lambda * torch.pow(target, self.peak_gamma)
        weighted_mse = torch.mean(weights * (pred - target) ** 2)
        weighted_mae = torch.mean(weights * torch.abs(pred - target))
        return self.alpha * weighted_mse + self.beta * weighted_mae


def build_loss(
    loss_name: str = "mse",
    alpha: float = 0.7,
    beta: float = 0.3,
    peak_lambda: float = 4.0,
    peak_gamma: float = 2.0,
) -> nn.Module:
    loss_name = loss_name.lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mse_mae":
        return MSEMAELoss(alpha=alpha, beta=beta)
    if loss_name == "peak_weighted_mse":
        return PeakWeightedMSELoss(peak_lambda=peak_lambda, peak_gamma=peak_gamma)
    if loss_name == "peak_weighted_mse_mae":
        return PeakWeightedMSEMAELoss(
            alpha=alpha,
            beta=beta,
            peak_lambda=peak_lambda,
            peak_gamma=peak_gamma,
        )
    raise ValueError(f"Unsupported loss_name: {loss_name}")
