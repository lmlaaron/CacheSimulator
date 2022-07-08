from rlmeta.core.callbacks import EpisodeCallbacks
from rlmeta.core.types import Action, TimeStep


# add guess correctness statistics during training
class MetricCallbacks(EpisodeCallbacks):
    def __init__(self):
        super().__init__()

    def on_episode_step(self, index: int, step: int, action: Action,
                        timestep: TimeStep) -> None:
        info = timestep.info
        if info['is_guess'] == True:
            self._custom_metrics["correct_rate"] = float(info["guess_correct"])
