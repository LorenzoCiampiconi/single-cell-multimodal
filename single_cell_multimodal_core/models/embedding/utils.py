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
