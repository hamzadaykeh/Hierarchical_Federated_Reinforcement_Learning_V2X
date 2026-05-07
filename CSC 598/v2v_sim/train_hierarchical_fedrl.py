# train_hierarchical_fedrl.py

import os
import copy
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from envs.highway_env import HighwayEnv



NUM_RSUS = 2
VEHICLES_PER_RSU = 3
GLOBAL_ROUNDS = 20
LOCAL_STEPS = 10000
SIGMA = 0.0001   # Smaller noise for stability

os.makedirs("results", exist_ok=True)


def average_policy_weights(policy_list):
    """
    Average ONLY neural network weights (not optimizer state).
    """
    avg_policy = copy.deepcopy(policy_list[0])

    for key in avg_policy:
        if torch.is_tensor(avg_policy[key]):
            stacked = torch.stack([p[key] for p in policy_list])
            avg_policy[key] = torch.mean(stacked, dim=0)

    return avg_policy


def add_noise_to_policy(policy_dict, sigma):
    """
    Add Gaussian noise ONLY to safe layers.
    Do NOT touch log_std.
    """
    noisy = copy.deepcopy(policy_dict)

    for key in noisy:

        # Skip sensitive log_std parameter
        if "log_std" in key:
            continue

        if torch.is_tensor(noisy[key]):
            noise = torch.normal(
                mean=0,
                std=sigma,
                size=noisy[key].shape,
                device=noisy[key].device
            )
            noisy[key] = noisy[key] + noise

    return noisy




global_env = Monitor(HighwayEnv())
global_model = PPO("MlpPolicy", global_env, verbose=0)

round_rewards = []


for g in range(GLOBAL_ROUNDS):

    print(f"\n====== Global Round {g+1} ======")

    rsu_policies = []

    # Extract only policy network weights
    global_policy = global_model.policy.state_dict()

    for r in range(NUM_RSUS):

        print(f"\n--- RSU {r+1} ---")
        vehicle_policies = []


        for v in range(VEHICLES_PER_RSU):

            print(f"Vehicle {v+1} training...")

            env = Monitor(HighwayEnv())
            model = PPO("MlpPolicy", env, verbose=0)

            # Load global policy weights
            model.policy.load_state_dict(global_policy)

            model.learn(total_timesteps=LOCAL_STEPS)

            local_policy = model.policy.state_dict()

            # Add SAFE noise
            noisy_policy = add_noise_to_policy(local_policy, SIGMA)

            vehicle_policies.append(noisy_policy)

        # RSU aggregation
        rsu_policy = average_policy_weights(vehicle_policies)
        rsu_policies.append(rsu_policy)

 
    new_global_policy = average_policy_weights(rsu_policies)

    global_model.policy.load_state_dict(new_global_policy)

    obs, _ = global_env.reset()
    total_reward = 0

    for _ in range(500):
        action, _ = global_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = global_env.step(action)
        total_reward += reward

        if terminated or truncated:
            obs, _ = global_env.reset()

    avg_reward = total_reward / 500
    round_rewards.append(avg_reward)

    print("Global Reward:", avg_reward)



np.save("results/hierarchical_fedrl.npy", np.array(round_rewards))

print("\nTraining Completed Successfully.")
print("Saved to results/hierarchical_fedrl.npy")
