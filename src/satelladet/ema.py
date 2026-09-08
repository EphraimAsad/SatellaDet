from __future__ import annotations

from copy import deepcopy
import math
import torch


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9998, tau: float = 2000.0):
        self.ema = deepcopy(model).eval()
        self.decay = float(decay)
        self.tau = float(tau)
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def current_decay(self) -> float:
        return self.decay * (1.0 - math.exp(-self.updates / self.tau))

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.updates += 1
        d = self.current_decay()
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()
        for k, v in ema_state.items():
            src = model_state[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src, alpha=1.0 - d)
            else:
                v.copy_(src)
