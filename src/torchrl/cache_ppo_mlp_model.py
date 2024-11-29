import os
import sys

from typing import  Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.backbone import CacheBackbone


class CachePPOMlpModel(nn.Module):
    def __init__(self,
                 latency_dim: int,
                 victim_acc_dim: int,
                 action_dim: int,
                 step_dim: int,
                 window_size: int,
                 action_embed_dim: int,
                 step_embed_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 1) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.backbone = CacheBackbone(latency_dim, victim_acc_dim, action_dim,
                                      step_dim, window_size, action_embed_dim,
                                      step_embed_dim, hidden_dim, num_layers)

        self.action_head = nn.Linear(self.hidden_dim, self.output_dim)
        self.value_head = nn.Linear(self.hidden_dim, 1)

        self._device = None

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        p = self.action_head(h)
        logpi = F.log_softmax(p, dim=-1)
        v = self.value_head(h)
        return logpi, v

    # @remote.remote_method(batch_size=128)
    def act(
        self, obs: torch.Tensor, deterministic_policy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._device is None:
            self._device = next(self.parameters()).device

        with torch.no_grad():
            logpi, v = self.forward(obs)
            greedy_action = logpi.argmax(-1, keepdim=True)
            sample_action = logpi.exp().multinomial(1, replacement=True)
            action = torch.where(deterministic_policy, greedy_action,
                                 sample_action)
            logpi = logpi.gather(dim=-1, index=action)

        return action, logpi, v

    def get_backbone(self, in_key="observation"):
        out = Seq(
            Mod(self.backbone, in_keys=[in_key], out_keys=["h"]),
        )
        return out

    def get_actor_head(self):
        out = Seq(
            Mod(self.action_head, in_keys=["h"], out_keys=["logits"]),
            Mod(lambda p: F.log_softmax(p, dim=-1), in_keys=["logits"], out_keys=["logits"])
        )
        return out

    def get_value_head(self):
        out = Mod(self.value_head, in_keys=["h"], out_keys=["state_value"])
        return out
