import numpy as np
import matplotlib.pyplot as plt

# Load episode rewards
single = np.load("results/single_agent.npy")

# Define number of rounds you want
num_rounds = 20

# Split rewards into 20 chunks
chunk_size = len(single) // num_rounds
single_round = []

for i in range(num_rounds):
    start = i * chunk_size
    end = (i + 1) * chunk_size
    single_round.append(np.mean(single[start:end]))

rounds = np.arange(1, num_rounds + 1)

plt.figure()
plt.plot(rounds, single_round, marker='s')
plt.xlabel("Round")
plt.ylabel("Average Reward")
plt.title("Single Agent Performance per Round")
plt.grid(True)
plt.show()