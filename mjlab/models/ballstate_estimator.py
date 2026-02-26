"""Temporal depth-to-ballstate estimator."""

from __future__ import annotations

import torch
from torch import nn


class BallStateEstimator(nn.Module):
  """Small CNN over temporal depth stacks predicting 6D ball state."""

  def __init__(
    self,
    history: int = 4,
    input_h: int = 32,
    input_w: int = 32,
    hidden_dim: int = 128,
  ):
    super().__init__()
    self.history = history
    self.input_h = input_h
    self.input_w = input_w
    self.encoder = nn.Sequential(
      nn.Conv2d(history, 16, kernel_size=3, stride=2, padding=1),
      nn.ELU(),
      nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
      nn.ELU(),
      nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
      nn.ELU(),
      nn.Flatten(),
    )
    with torch.no_grad():
      dummy = torch.zeros(1, history, input_h, input_w)
      feat_dim = int(self.encoder(dummy).shape[-1])
    self.head = nn.Sequential(
      nn.Linear(feat_dim, hidden_dim),
      nn.ELU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ELU(),
      nn.Linear(hidden_dim, 6),
    )

  def forward(self, depth_stack: torch.Tensor) -> torch.Tensor:
    """Predict [x,y,z,vx,vy,vz] from [B,history,H,W] depth stack."""
    features = self.encoder(depth_stack)
    return self.head(features)

