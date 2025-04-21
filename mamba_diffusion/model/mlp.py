import torch
import torch.nn as nn

import torch.nn.functional as F
from torch import Tensor

class MLP (nn.Module):
    def __init__(self, 
                fan_in,
                linear_bias:bool = False,
                gated_linear_unit: bool = True,
                is_expert: bool = False,
                layer_idx=None,
                device=None,
    ):
        super().__init__()

        self.layer = layer_idx
        fan_h1 = 4 * fan_in
        fan_h2 = 4 * fan_in

        # if gated linear unit, then double the fan to get scale shift
        if gated_linear_unit:
            fan_h1 *= 2
        
        self.fc1 = nn.Linear (fan_in, fan_h1, bias=linear_bias, device=device)
        self.fc1.is_expert = is_expert

        if gated_linear_unit:

            def glu(x):
                x1, x2 = x.chunk(2, dim=-1)
                # devaiation from previous
                return F.silu (x1) * x2
            self.activation_func = glu
        else:
            self.activation_func = F.silu
        
        self.fc2 = nn.Linear (fan_h2, fan_in, bias=linear_bias, device=device)
    
    def forward (self, x, inference_params=None):
        x = self.fc1(x)
        x = self.activation_func(x)
        x = self.fc2(x)
        return x

class GatedMLP(nn.Module):
    def __init__(
        self,
        fan_in: int,
        fan_h: int = None,
        fan_out: int = None,
        act_layer=F.silu,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        fan_out = fan_out if fan_out is not None else fan_in
        fan_h = fan_h if fan_h is not None else fan_in
        self.fc1 = nn.Linear(fan_in, 2 * fan_h, bias=bias)
        self.fc2 = nn.Linear(fan_h, fan_out, bias=bias)
        self.act_layer = act_layer()

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=-1)
        hidden = self.act_layer(x1) * x2
        return self.fc2(hidden)

