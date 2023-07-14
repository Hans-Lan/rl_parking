from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.trainer import Trainer
from ray.rllib.evaluation.episode import Episode


class ParkingCallbacks(DefaultCallbacks):

    def on_episode_end(self, *, worker, base_env, policies, episode: Episode, **kwargs) -> None:
        info = episode.last_info_for()
        episode.custom_metrics['parked'] = info['parked']
        episode.custom_metrics['collision'] = info['collision']
        episode.custom_metrics['easy'] = info['easy']


class CurriculumCallbacks(DefaultCallbacks):

    def on_train_result(self, *, trainer: Trainer, result: dict, **kwargs) -> None:
        iters = result['training_iteration']
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_jump_start(iters)
            )
        )