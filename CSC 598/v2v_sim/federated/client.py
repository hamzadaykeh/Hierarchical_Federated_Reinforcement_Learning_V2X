from stable_baselines3 import PPO
from envs.highway_env import HighwayEnv
import torch

class FederatedClient:
    def __init__(self, client_id):
        self.client_id = client_id
        self.env = HighwayEnv(num_vehicles=5)
        self.model = PPO(
            "MlpPolicy",
            self.env,
            verbose=0,
            learning_rate=3e-4,
            gamma=0.99
        )

    def set_weights(self, weights):
        self.model.policy.load_state_dict(weights)
    #here is the update weight function for the privacy
    def get_weights(self, noise_std=0.01):
        noisy_weights = {}

        for key, value in self.model.policy.state_dict().items():
            noise = torch.normal(
                mean=0.0,
                std=noise_std,
                size=value.shape
            )
            noisy_weights[key] = value + noise

        return noisy_weights

    def local_train(self, timesteps=2000):
        self.model.learn(total_timesteps=timesteps)
