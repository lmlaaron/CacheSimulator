from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from tensordict.prototype import symbolic_trace


class CachePPOTransformerModel(nn.Module):
    def __init__(
        self,
        latency_dim: int,
        victim_acc_dim: int,
        action_dim: int,
        step_dim: int,
        action_embed_dim: int,
        step_embed_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1
        ) -> None:
        super().__init__()

        self.latency_dim = latency_dim
        self.victim_acc_dim = victim_acc_dim
        self.action_dim = action_dim
        self.step_dim = step_dim
        # self.window_size = window_size

        self.action_embed_dim = action_embed_dim
        self.step_embed_dim = step_embed_dim
        self.input_dim = (self.latency_dim + self.victim_acc_dim +
                          self.action_embed_dim + self.step_embed_dim)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.action_embed = nn.Embedding(
            self.action_dim,
            self.action_embed_dim
            )
        self.step_embed = nn.Embedding(self.step_dim, self.step_embed_dim)

        self.linear_i = nn.Linear(self.input_dim, self.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dropout=0.0,
            batch_first=True
            )
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers, )

        self.action_head = nn.Linear(self.hidden_dim, self.output_dim)
        self.value_head = nn.Linear(self.hidden_dim, 1)

        self._device = None

    def make_one_hot(
        self, src: torch.Tensor, num_classes: int,
        mask: torch.Tensor
        ) -> torch.Tensor:
        try:
            # mask = (src == -1)
            src1 = src.masked_fill(mask, 0)
            ret = F.one_hot(src1, num_classes)
            return ret.masked_fill(mask.unsqueeze(-1), 0.0)
        except Exception as e:
            torch.save({'src': src, 'num_classes': num_classes, 'mask': mask}, 'dump.pt')
            raise e

    def make_embedding(
        self, src: torch.Tensor, embed: nn.Embedding,
        mask: torch.Tensor
        ) -> torch.Tensor:
        try:
            # mask = (src == -1)
            src1 = src.masked_fill(mask, 0)
            ret = embed(src1)
            return ret.masked_fill(mask.unsqueeze(-1), 0.0)
        except Exception as e:
            torch.save({'src': src, 'embed': embed, 'mask': mask}, 'dump.pt')
            raise e

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = obs.to(torch.int64)

        # batch_size = obs.size(0)
        l, v, act, stp = torch.unbind(obs, dim=-1)
        mask = (stp == -1)
        l = self.make_one_hot(l, self.latency_dim, mask)
        v = self.make_one_hot(v, self.victim_acc_dim, mask)
        act = self.make_embedding(act, self.action_embed, mask)
        stp = self.make_embedding(stp, self.step_embed, mask)

        x = torch.cat((l, v, act, stp), dim=-1)
        x = self.linear_i(x)
        # Before: 0 is batch, 1 is time, so they are inverted
        # now: we want 0 to be batch
        # x = x.transpose(0, 1).contiguous()
        h = self.encoder(x)
        # average over time
        h = h.mean(dim=-2)

        p = self.action_head(h)
        logpi = F.log_softmax(p, dim=-1)
        v = self.value_head(h)

        return logpi, v

    def act(
        self, obs: torch.Tensor, deterministic_policy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logpi, v = self.forward(obs)
            greedy_action = logpi.argmax(-1, keepdim=True)
            sample_action = logpi.exp().multinomial(1, replacement=True)
            action = torch.where(
                deterministic_policy, greedy_action,
                sample_action
                )
            logpi = logpi.gather(dim=-1, index=action)

        return action, logpi, v

    def get_backbone(self, in_key="observation"):
        out = Seq(
            Mod(
                lambda obs: obs.to(torch.int64),
                in_keys=[in_key],
                out_keys=["_obs"]
                ),
            Mod(
                lambda obs: torch.unbind(obs, dim=-1),
                in_keys=["_obs"],
                out_keys=["l", "v", "act", "stp"]
                ),
            Mod(lambda stp: (stp == -1), in_keys=["stp"], out_keys=["mask"]),
            Mod(
                lambda l, mask: self.make_one_hot(l, self.latency_dim, mask),
                in_keys=["l", "mask"],
                out_keys=["l"]
                ),
            Mod(
                lambda v, mask: self.make_one_hot(
                    v,
                    self.victim_acc_dim,
                    mask
                    ),
                in_keys=["v", "mask"],
                out_keys=["v"]
                ),
            Seq(
                Mod(
                    lambda act, mask: act.masked_fill(mask, 0),
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
            Seq(
                Mod(
                    lambda stp, mask: stp.masked_fill(mask, 0),
                    in_keys=["stp", "mask"],
                    out_keys=["stp"]
                    ),
                Mod(self.step_embed, in_keys=["stp"], out_keys=["stp"]),
                Mod(
                    lambda stp, mask: stp.masked_fill(mask.unsqueeze(-1), 0.0),
                    in_keys=["stp", "mask"],
                    out_keys=["stp"]
                    ),
            ),
            Mod(
                lambda l, v, act, stp: torch.cat((l, v, act, stp), dim=-1),
                in_keys=["l", "v", "act", "stp"],
                out_keys=["x"]
                ),
            Mod(self.linear_i, in_keys=["x"], out_keys=["x"]),
            Mod(self.encoder, in_keys=["x"], out_keys=["x"]),
            Mod(lambda x: x.mean(-2), in_keys=["x"], out_keys=["mean"]),
        )
        # out = symbolic_trace(out)
        return out

    def get_actor_head(self):
        out = Seq(
            Mod(self.action_head, in_keys=["mean"], out_keys=["logits"]),
            Mod(
                lambda p: F.log_softmax(p, dim=-1),
                in_keys=["logits"],
                out_keys=["logits"]
                )
        )
        return out

    def get_value_head(self):
        out = Mod(self.value_head, in_keys=["mean"], out_keys=["state_value"])
        return out
