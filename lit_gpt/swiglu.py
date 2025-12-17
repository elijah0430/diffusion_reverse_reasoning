from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from xformers.ops import SwiGLU as SwiGLU  # type: ignore
except ModuleNotFoundError:  # pragma: no cover

    class SwiGLU(nn.Module):
        def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: Optional[int] = None,
            bias: bool = True,
            _pack_weights: bool = False,
        ) -> None:
            super().__init__()
            out_features = out_features if out_features is not None else in_features

            self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
            self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
            self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.w3(F.silu(self.w1(x)) * self.w2(x))
