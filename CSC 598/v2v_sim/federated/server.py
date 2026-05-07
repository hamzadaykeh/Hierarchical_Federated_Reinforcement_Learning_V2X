import torch
from federated.client import FederatedClient

def federated_training(self, rounds=5, noise_std=0.01):
    from evaluation.evaluate_model import evaluate_model
    import numpy as np
    import os

    global_weights = None
    round_rewards = []

    for r in range(rounds):
        print(f"\n--- Federated Round {r+1} ---")

        client_weights = []

        for client in self.clients:
            if global_weights is not None:
                client.set_weights(global_weights)

            client.local_train()
            client_weights.append(client.get_weights(noise_std=noise_std))

        # 🔹 Aggregate weights
        global_weights = self.aggregate(client_weights)

        # 🔹 Evaluate global model
        self.global_model.set_weights(global_weights)
        avg_reward = evaluate_model(self.global_model)

        print(f"Round {r+1} Average Reward: {avg_reward}")
        round_rewards.append(avg_reward)

    # 🔹 Save results
    os.makedirs("results", exist_ok=True)
    np.save("results/single_agent.npy", np.array(round_rewards))

    return global_weights
def aggregate_prox(self, client_weights, mu=0.01):
    avg_weights = self.aggregate(client_weights)

    # simple proximal correction
    for key in avg_weights:
        avg_weights[key] -= mu * avg_weights[key]

    return avg_weights
    os.makedirs("results", exist_ok=True)
    np.save("results/fedprox.npy", np.avg_weights)
