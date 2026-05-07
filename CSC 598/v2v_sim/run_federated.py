from federated.server import FederatedServer

server = FederatedServer(num_clients=3)

# Privacy-preserving FL
global_model = server.federated_training(
    rounds=5,
    noise_std=0.02   # privacy strength
)

print("\nPrivacy-preserving federated training completed.")
