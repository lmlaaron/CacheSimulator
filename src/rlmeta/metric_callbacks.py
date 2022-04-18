from rlmeta.core.callbacks import EpisodeCallbacks
from rlmeta.core.types import Action, TimeStep


class MetricCallbacks(EpisodeCallbacks):
    def __init__(self):
        super().__init__()

        self.tot_guess = 0
        self.acc_guess = 0

    def on_reset(self) -> None:
        self._custom_metrics = {}

        self.tot_guess = 0
        self.acc_guess = 0

    def on_episode_step(self, index: int, step: int, action: Action,
                        timestep: TimeStep) -> None:
        info = timestep.info
        if info["is_guess"]:
            self.tot_guess += 1
            self.acc_guess += float(info["guess_correct"])

        if timestep.done:
            self._custom_metrics[
                "correct_rate"] = self.acc_guess / self.tot_guess
