from envs.highway_env import HighwayEnv
from envs.v2v_graph import V2VGraph
import numpy as np
#pip install networkx

env = HighwayEnv(num_vehicles=5)
graph_builder = V2VGraph(communication_range=30.0)

obs, _ = env.reset()

print("Initial state:")
print(obs)

for t in range(5):
    actions = np.random.uniform(-1, 1, size=5)
    obs, _, _, _, _ = env.step(actions)

    positions = obs[:, 0]
    G = graph_builder.build_graph(positions)

    print(f"\nTime step {t+1}")
    print("Positions:", positions)
    print("Communication links:", list(G.edges(data=True)))