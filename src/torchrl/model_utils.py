from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from cache_ppo_mlp_model import CachePPOMlpModel
from cache_ppo_lstm_model import CachePPOLstmModel
from cache_ppo_transformer_model import CachePPOTransformerModel

from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod, ProbabilisticTensorDictModule as ProbMod, ProbabilisticTensorDictSequential as ProbSeq

def get_model(cfg: Dict[str, Any],
              window_size: int,
              output_dim: int,
              checkpoint: Optional[str] = None) -> nn.Module:
    cfg.args.step_dim = window_size
    if "window_size" in cfg.args:
        cfg.args.window_size = window_size
    cfg.args.output_dim = output_dim

    model = None
    if cfg.type == "mlp":
        model = CachePPOMlpModel(**cfg.args)
    elif cfg.type == "lstm":
        model = CachePPOLstmModel(**cfg.args)
    elif cfg.type == "transformer":
        model = CachePPOTransformerModel(**cfg.args)

    model = Policy(model)

    if model is not None and checkpoint is not None:
        params = torch.load(checkpoint)
        model.load_state_dict(params)

    return model

class Policy(ProbSeq):
    def __init__(self, module):
        actor = module.get_actor_head()
        value = module.get_value_head()
        backbone = module.get_backbone()
        prob = ProbMod(in_keys=["logits"], out_keys=["action"], distribution_class=torch.distributions.OneHotCategorical, return_log_prob=True)
        super().__init__(backbone, actor, value, prob)
        for m in self.modules():
            assert not m._is_stateless, m

    def get_actor(self):
        return ProbSeq(self[0], self[1], self[3])

    def get_value_head(self):
        return self[2]

    def get_value(self):
        return Seq(self[0], self[2])
