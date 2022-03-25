from typing import Dict, List, Tuple

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from dnn import DNNEncoder


class DNNModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str, **kwargs) -> None:
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        if len(kwargs) > 0:
            custom_model_config = kwargs
        else:
            custom_model_config = model_config["custom_model_config"]

        self.input_dim = custom_model_config["input_dim"]
        self.embed_dim = custom_model_config["embed_dim"]
        self.hidden_dim = custom_model_config["hidden_dim"]
        self.action_dim = num_outputs
        self.num_blocks = custom_model_config.get("num_blocks", 1)

        self.step_embed = nn.Embedding(
            custom_model_config["max_episode_steps"], self.embed_dim)
        self.backbone = DNNEncoder(self.input_dim - 1 + self.embed_dim,
                                   self.hidden_dim, self.hidden_dim,
                                   self.num_blocks)
        self.linear_a = nn.Linear(self.hidden_dim, self.action_dim)
        self.linear_v = nn.Linear(self.hidden_dim, 1)

        self._device = None
        self._features = None

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str,
                                       TensorType], state: List[TensorType],
                seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        if self._device is None:
            self._device = next(self.parameters()).device

        obs = input_dict["obs"].to(self._device)
        batch_size = obs.size(0)

        cur_step = obs[:, :, -1].to(torch.int64)
        cur_step_embed = self.step_embed(cur_step)
        obs = torch.cat((obs, cur_step_embed), dim=-1)
        obs = obs.view(batch_size, -1)

        h = self.backbone(obs)
        a = self.linear_a(h)
        self._features = h

        return a, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None
        v = self.linear_v(self._features)
        return v.squeeze(1)


ModelCatalog.register_custom_model("dnn_model", DNNModel)
