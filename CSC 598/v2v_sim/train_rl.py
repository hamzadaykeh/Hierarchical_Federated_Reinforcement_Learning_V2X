import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from envs.highway_env import HighwayEnv



# Wrap environment with Monitor to track rewards
env = HighwayEnv()
env = Monitor(env)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99
)

model.learn(total_timesteps=200000)

# Save model
model.save("results/ppo_highway")

# Save training rewards
training_rewards = env.get_episode_rewards()
np.save("results/single_agent.npy", training_rewards)

print("Training completed.")
print("Model saved in results/ppo_highway.zip")
print("Rewards saved in results/single_agent.npy")