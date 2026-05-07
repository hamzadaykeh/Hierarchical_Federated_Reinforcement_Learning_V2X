from stable_baselines3 import PPO
from envs.highway_env import HighwayEnv
from evaluation.logger import MetricsLogger
import numpy as np
import os 
env = HighwayEnv()
model = PPO.load("ppo_highway")
collisions = 0
speed_errors = []

round_rewards = []

for r in range(20):   # 20 rounds
    obs, _ = env.reset()
    total_reward = 0

    for _ in range(500):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        # collision check
        if env.positions[1] - env.positions[0] < 5:
            collisions += 1

        speed_errors.append(abs(env.velocities[0] - 20))

    round_rewards.append(total_reward)

os.makedirs("results", exist_ok=True)
np.save("results/single_agent.npy", round_rewards)
print("Saved single_agent.npy")

