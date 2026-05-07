import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from envs.highway_env import HighwayEnv

# ===== Settings =====
NUM_CLIENTS = 2000
ROUNDS = 200
LOCAL_STEPS = 50000
MU = 0.01
EVAL_STEPS = 5000
# ====================

os.makedirs("results", exist_ok=True)

# Initialize global model
global_env = Monitor(HighwayEnv())
global_model = PPO("MlpPolicy", global_env, verbose=0, device="cpu")

round_rewards = []

for r in range(ROUNDS):

    print(f"\n====== FedProx Round {r+1} ======")

    client_weights = []

    # Get global policy weights (pure tensors)
    global_policy_weights = global_model.policy.state_dict()

    # -------- Train Clients --------
    for c in range(NUM_CLIENTS):

        print(f"Client {c+1} training...")

        env = Monitor(HighwayEnv())
        model = PPO("MlpPolicy", env, verbose=0, device="cpu")

        # Load global weights
        model.policy.load_state_dict(global_policy_weights)

        # Local training
        model.learn(total_timesteps=LOCAL_STEPS)

        local_weights = model.policy.state_dict()

        # ===== FedProx correction (tensor level only) =====
        corrected_weights = {}

        for key in global_policy_weights.keys():
            local_tensor = local_weights[key]
            global_tensor = global_policy_weights[key]

            corrected_weights[key] = (
                local_tensor - MU * (local_tensor - global_tensor)
            )

        client_weights.append(corrected_weights)

    # -------- Aggregation (FedAvg) --------
    new_global_weights = {}

    for key in global_policy_weights.keys():

        stacked = torch.stack(
            [client_weights[i][key] for i in range(NUM_CLIENTS)],
            dim=0
        )

        new_global_weights[key] = torch.mean(stacked, dim=0)

    # Update global model
    global_model.policy.load_state_dict(new_global_weights)

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

    print("Round Reward:", avg_reward)

# Save results
np.save("results/fedprox.npy", np.array(round_rewards))
global_model.save("results/fedprox_model")

print("\nFedProx training completed successfully.")
print("Saved: results/fedprox.npy")
