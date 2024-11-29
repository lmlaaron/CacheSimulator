
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.feature_extraction import get_graph_node_names, \
    create_feature_extractor
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq, TensorDictModuleBase

from torchrl.modules import LSTMModule
from tensordict.prototype import symbolic_trace

class CachePPOLstmModel(nn.Module):
    def __init__(self,
                 latency_dim: int,
                 victim_acc_dim: int,
                 action_dim: int,
                 step_dim: int,
                 action_embed_dim: int,
                 step_embed_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 1) -> None:
        super().__init__()

        self.latency_dim = latency_dim
        self.victim_acc_dim = victim_acc_dim
        self.action_dim = action_dim
        self.step_dim = step_dim
        # self.window_size = window_size

        self.action_embed_dim = action_embed_dim
        # self.step_embed_dim = step_embed_dim
        # self.input_dim = (self.latency_dim + self.victim_acc_dim +
        #                   self.action_embed_dim + self.step_embed_dim)
        self.input_dim = (self.latency_dim + self.victim_acc_dim +
                          self.action_embed_dim)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.action_embed = nn.Embedding(self.action_dim,
                                         self.action_embed_dim)
        # self.step_embed = nn.Embedding(self.step_dim, self.step_embed_dim)

        self.linear_i = nn.Linear(self.input_dim, self.hidden_dim)

        self.encoder = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            self.num_layers,
            bias=False,  # Disable bias for pre-padding sequence
            bidirectional=False,
            batch_first=True
        )

        self.action_head = nn.Linear(2 * self.hidden_dim, self.output_dim)
        self.value_head = nn.Linear(2 * self.hidden_dim, 1)

        self._device = None

    def make_one_hot(self, src: torch.Tensor,
                     num_classes: int) -> torch.Tensor:
        mask = (src == -1)
        src = src.masked_fill(mask, 0)
        ret = F.one_hot(src, num_classes)
        return ret.masked_fill(mask.unsqueeze(-1), 0.0)

    def make_embedding(self, src: torch.Tensor,
                       embed: nn.Embedding) -> torch.Tensor:
        mask = (src == -1)
        src = src.masked_fill(mask, 0)
        ret = embed(src)
        return ret.masked_fill(mask.unsqueeze(-1), 0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = obs.to(torch.int64)
        assert obs.dim() == 3

        # batch_size = obs.size(0)
        obs = torch.flip(obs, dims=(1, ))  # Reverse input to pre-padding

        l, v, act, _ = torch.unbind(obs, dim=-1)
        l = self.make_one_hot(l, self.latency_dim)
        v = self.make_one_hot(v, self.victim_acc_dim)
        act = self.make_embedding(act, self.action_embed)
        # stp = self.make_embedding(stp, self.step_embed)

        x = torch.cat((l, v, act), dim=-1)
        x = self.linear_i(x)
        x = x.transpose(0, 1).contiguous()

        _, (h, c) = self.encoder(x)
        h = h.mean(dim=0)
        c = c.mean(dim=0)
        h = torch.cat((h, c), dim=-1)

        p = self.action_head(h)
        logpi = F.log_softmax(p, dim=-1)
        v = self.value_head(h)

        return logpi, v

    # @remote.remote_method(batch_size=128)
    def act(
        self, obs: torch.Tensor, deterministic_policy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logpi, v = self.forward(obs)
            greedy_action = logpi.argmax(-1, keepdim=True)
            sample_action = logpi.exp().multinomial(1, replacement=True)
            action = torch.where(deterministic_policy, greedy_action,
                                 sample_action)
            logpi = logpi.gather(dim=-1, index=action)

        return action, logpi, v

    def common(self, in_key="observation"):
        nodes = get_graph_node_names(self)
        print(nodes)
        nodes = create_feature_extractor(self, return_nodes=nodes[:-3])
        return Mod(nodes, in_keys=[in_key], out_keys=["hidden"])

    def get_backbone(self, in_key="observation"):
        lstm = LSTMModule(lstm=self.encoder, in_keys=["x", "h", "c"], out_keys=["_", "h", "c"])
        lstm = lstm.set_recurrent_mode(True)
        class PrintTD(TensorDictModuleBase):
            in_keys = []
            out_keys = []
            def forward(self, td):
                print(td)
                return td
        out = Seq(
            Mod(
                lambda obs: obs.to(torch.int64),
                in_keys=[in_key],
                out_keys=["_obs"]
                ),
            # we need this for LSTM
            Mod(
                lambda obs: torch.zeros(obs.shape[:-2], device=obs.device, dtype=torch.bool),
                in_keys=["_obs"],
                out_keys=["is_init"]
                ),
            Mod(
                lambda obs: torch.flip(obs, dims=(1, )),
                in_keys=["_obs"],
                out_keys=["_obs"]
                ),
            Mod(
                lambda obs: torch.unbind(obs, dim=-1),
                in_keys=["_obs"],
                out_keys=["l", "v", "act", "_"],
                ),
            Mod(
                lambda l: self.make_one_hot(l, self.latency_dim, ),
                in_keys=["l"],
                out_keys=["l"]
                ),
            Mod(
                lambda v: self.make_one_hot(v, self.victim_acc_dim),
                in_keys=["v"],
                out_keys=["v"]
                ),
            Seq(
                Mod(
                    lambda act: (act == -1),
                    in_keys=["act"],
                    out_keys=["mask"]
                    ),
                Mod(
                    lambda act, mask: act.masked_fill(mask, 0.0),
                    in_keys=["act", "mask"],
                    out_keys=["act"]
                    ),
                Mod(self.action_embed, in_keys=["act"], out_keys=["act"]),
                Mod(
                    lambda act, mask: act.masked_fill(mask.unsqueeze(-1), 0.0),
                    in_keys=["act", "mask"],
                    out_keys=["act"]
                    ),
            ),
            Mod(
                lambda l, v, act, stp: torch.cat((l, v, act, ), dim=-1),
                in_keys=["l", "v", "act", "stp"],
                out_keys=["x"]
                ),
            Mod(self.linear_i, in_keys=["x"], out_keys=["x"]),
            # Mod(lambda x: x.transpose(0, 1).contiguous(), in_keys=["x"], out_keys=["x"]),
            lstm,
            Mod(lambda h, c: torch.cat((h.mean(-2), c.mean(-2)), dim=-1), in_keys=["h", "c"], out_keys=["h"] ),
        )
        # out = symbolic_trace(out)
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
