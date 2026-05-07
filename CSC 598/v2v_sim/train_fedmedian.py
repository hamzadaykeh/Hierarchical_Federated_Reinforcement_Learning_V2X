import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from envs.highway_env import HighwayEnv


NUM_CLIENTS = 2000
ROUNDS = 200
LOCAL_STEPS = 50000
EVAL_STEPS = 5000
# ====================

os.makedirs("results", exist_ok=True)

# Global model
global_env = Monitor(HighwayEnv())
global_model = PPO("MlpPolicy", global_env, verbose=0, device="cpu")

round_rewards = []

for r in range(ROUNDS):

    print(f"\n====== FedMedian Round {r+1} ======")

    client_weights = []

    # Get global policy weights (pure tensors)
    global_weights = global_model.policy.state_dict()

    # -------- Train Clients --------
    for c in range(NUM_CLIENTS):

        print(f"Client {c+1} training...")

        env = Monitor(HighwayEnv())
        model = PPO("MlpPolicy", env, verbose=0, device="cpu")

        # Load global weights
        model.policy.load_state_dict(global_weights)

        # Local training
        model.learn(total_timesteps=LOCAL_STEPS)

        # Save only policy weights (pure tensors)
        client_weights.append(model.policy.state_dict())

    # -------- FedMedian Aggregation --------
    new_global_weights = {}

    for key in global_weights.keys():

        # Stack client tensors
        stacked = torch.stack(
            [client_weights[i][key] for i in range(NUM_CLIENTS)],
            dim=0
        )

        # Take element-wise median
        new_global_weights[key] = torch.median(stacked, dim=0).values

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
np.save("results/fedmedian.npy", np.array(round_rewards))
global_model.save("results/fedmedian_model")

print("\nFedMedian training completed successfully.")
print("Saved: results/fedmedian.npy")
