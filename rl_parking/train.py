import ray.tune, ray.tune.registry, ray.tune.utils.log
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.callbacks import MultiCallbacks

from rl_parking.callbacks import ParkingCallbacks, CurriculumCallbacks
from rl_parking.env import env_creator
from rl_parking.model import Keras_FullyConnectedNetwork, extract_config_from_default
from rl_parking.utils.fs import PROJECT_ROOT

ray.tune.registry.register_env('bosch_epf_parking', env_creator)

stop_criteria = {
    'timesteps_total': int(2e7),
}

agent_config = {
    # Env config
    'env': 'bosch_epf_parking',
    'env_config': {
        'mode': 'vertical',
        'total_iters': 300,
        'jump_start': True,
    },

    'callbacks': MultiCallbacks([ParkingCallbacks, CurriculumCallbacks]),
    'observation_filter': 'MeanStdFilter',
    'metrics_num_episodes_for_smoothing': 1000,

    # Worker config
    'framework': 'tf2',
    'eager_tracing': True,
    'num_workers': 1,
    'num_envs_per_worker': 4,
    'num_gpus': 0,

    # PPO config
    'rollout_fragment_length': 500,
    'batch_mode': 'complete_episodes',  # 'truncate_episodes',
    'train_batch_size': 20000,
    'sgd_minibatch_size': 20000,
    'num_sgd_iter': 30,
    'lr': 3e-4,
    'lambda': 0.97,
    'gamma': 0.99,
    'no_done_at_end': False,
    'entropy_coeff': 0.0,  # ray.tune.grid_search([0.0, 0.01]),
    'vf_clip_param': 10.0,

    # Model config
    'model': {
        'custom_model': Keras_FullyConnectedNetwork,

        'custom_model_config': {
            **extract_config_from_default(Keras_FullyConnectedNetwork),

            'fcnet_hiddens': [256, 256],
            'fcnet_activation': 'tanh',
            'vf_share_layers': False,
            'free_log_std': True,
            'init_log_std': 0.0,  # ray.tune.grid_search([0.0, 0.5]),
        }
    },
    'seed': 2,
}

if __name__ == "__main__":
    ray.tune.run(
        PPOTrainer,
        checkpoint_at_end=True,
        keep_checkpoints_num=3,
        checkpoint_freq=50,
        stop=stop_criteria,
        config=agent_config,
        verbose=ray.tune.utils.log.Verbosity.V1_EXPERIMENT,
        # progress_reporter=create_progress(),
        # callbacks=create_callbacks(),
        max_concurrent_trials=3,
        log_to_file=True,
        local_dir=PROJECT_ROOT / 'log3',
    )
