import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from envs.highway_env import HighwayEnv


# =========================
# CONFIG
# =========================
NUM_CLIENTS = 2000
ROUNDS = 200
LOCAL_STEPS = 50000  # per client per round
EVAL_STEPS = 5000

os.makedirs("results", exist_ok=True)


# =========================
# Initialize Global Model
# =========================
global_env = Monitor(HighwayEnv())
global_model = PPO("MlpPolicy", global_env, verbose=0)

round_rewards = []


# =========================
# Federated Training Loop
# =========================
for r in range(ROUNDS):

    print(f"\n========== Federated Round {r+1} ==========")

    client_weights = []

    # -------- Train Clients --------
    for c in range(NUM_CLIENTS):

        print(f"Client {c+1} training...")

        env = Monitor(HighwayEnv())
        model = PPO("MlpPolicy", env, verbose=0)

        # Load global weights into client
        model.policy.load_state_dict(global_model.policy.state_dict())

        # Local training
        model.learn(total_timesteps=LOCAL_STEPS)

        # Save only policy weights
        client_weights.append(model.policy.state_dict())


    # -------- FedAvg Aggregation --------
    print("Aggregating weights...")

    global_weights = global_model.policy.state_dict()
    new_weights = {}

    for key in global_weights.keys():

        stacked = torch.stack(
            [client_weights[i][key] for i in range(NUM_CLIENTS)],
            dim=0
        )

        new_weights[key] = torch.mean(stacked, dim=0)

    # Update global model
    global_model.policy.load_state_dict(new_weights)


    # -------- Evaluate Global Model --------
    obs, _ = global_env.reset()
    total_reward = 0

    for _ in range(EVAL_STEPS):
        action, _ = global_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = global_env.step(action)
        total_reward += reward

        if terminated or truncated:
            obs, _ = global_env.reset()

    avg_reward = total_reward / EVAL_STEPS
    round_rewards.append(avg_reward)

    print(f"Round {r+1} Average Reward: {avg_reward}")


np.save("results/fedavg.npy", np.array(round_rewards))
global_model.save("results/fedavg_model")

print("\nFedAvg training completed successfully.")
print("Saved:")
print(" - results/fedavg.npy")
print(" - results/fedavg_model.zip")
