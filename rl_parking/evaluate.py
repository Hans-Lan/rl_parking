import ray.tune, ray.tune.registry
from ray.rllib.agents.ppo import PPOTrainer
from rl_parking.train import agent_config

# restore agent
agent_config['num_workers'] = 0
agent = PPOTrainer(config=agent_config)
agent.restore(
    'log/PPOTrainer_2023-07-10_23-23-34/PPOTrainer_bosch_epf_parking_bc51f_00000_0_2023-07-10_23-23-34/checkpoint_000094/checkpoint-94'
)
agent.compute_action