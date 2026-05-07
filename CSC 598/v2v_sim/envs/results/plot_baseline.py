import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.highway_env import HighwayEnv

# Load trained model
env = HighwayEnv(num_vehicles=5)
model = PPO.load("ppo_highway")

obs, _ = env.reset()

speeds = []
distances = []
rewards = []

for t in range(300):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    speeds.append(env.velocities[0])
    distances.append(env.positions[1] - env.positions[0])
    rewards.append(reward)

    if terminated:
        break

time = np.arange(len(speeds))

# --------- GRAPH 1: Speed ----------
plt.figure()
plt.plot(time, speeds)
plt.xlabel("Time Step")
plt.ylabel("Speed (m/s)")
plt.title("Vehicle Speed vs Time (Baseline PPO)")
plt.grid()
plt.savefig("results/speed_baseline.png")
plt.close()

# --------- GRAPH 2: Distance ----------
plt.figure()
plt.plot(time, distances)
plt.xlabel("Time Step")
plt.ylabel("Distance to Front Vehicle (m)")
plt.title("Inter-Vehicle Distance vs Time (Baseline PPO)")
plt.grid()
plt.savefig("results/distance_baseline.png")
plt.close()

# --------- GRAPH 3: Reward ----------
plt.figure()
plt.plot(time, rewards)
plt.xlabel("Time Step")
plt.ylabel("Reward")
plt.title("Reward vs Time (Baseline PPO)")
plt.grid()
plt.savefig("results/reward_baseline.png")
plt.close()

print("Graphs saved in results folder.")
