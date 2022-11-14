from typing import Callable, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import nn


@torch.no_grad()
def init_fc_snn(layer):
    if not isinstance(layer, nn.Linear) or isinstance(layer, nn.LazyLinear):
        return
    nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
    if layer.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(layer.bias, -bound, bound)


class WeightedLosses(nn.Module):
    def __init__(self, losses: Mapping[str, Tuple[float, nn.Module]], reduction: str | Callable = "mean"):
        super().__init__()
        self.losses = nn.ModuleDict({k: m for k, (m, _) in losses.items()})
        self.weights = {k: w for k, (_, w) in losses.items()}
        self.reduction = torch.mean if reduction == "mean" else reduction

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        wloss = {}
        for k, loss in self.losses.items():
            out = loss(preds, target)
            out = out if out.numel() == 1 else self.reduction(out)
            wloss[k] = out * self.weights[k]

        return wloss
