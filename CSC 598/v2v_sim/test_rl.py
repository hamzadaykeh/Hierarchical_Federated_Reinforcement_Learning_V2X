from stable_baselines3 import PPO
from envs.highway_env import HighwayEnv
from evaluation.logger import MetricsLogger
import numpy as np

env = HighwayEnv(num_vehicles=5)
model = PPO.load("ppo_highway")

obs, _ = env.reset()

total_reward = 0
collisions = 0
speed_errors = []

for t in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, _, _, _ = env.step(action)

    total_reward += reward

    print(f"Step {t+1}")
    print("State:", obs)
    print("Collisions:", collisions)
    print("Reward:", reward)
