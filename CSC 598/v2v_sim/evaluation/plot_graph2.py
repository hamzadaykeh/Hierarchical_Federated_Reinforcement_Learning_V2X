import numpy as np
import matplotlib.pyplot as plt

fedavg = np.load("results/fedAvg.npy")
fedprox = np.load("results/fedprox.npy")
single = np.load("results/fedmedian.npy")


num_rounds = 20

# Split rewards into 20 chunks
chunk_size = len(fedavg) // num_rounds
single_round = []

for i in range(num_rounds):
    start = i * chunk_size
    end = (i + 1) * chunk_size
    single_round.append(np.mean(single[start:end]))

rounds = np.arange(1, num_rounds + 1)

plt.figure()
plt.figure()

plt.plot(rounds, fedavg[:num_rounds], marker='o', label="FedAvg")
plt.plot(rounds, fedprox[:num_rounds], marker='^', label="FedProx")
plt.plot(rounds, single_round, marker='s', label="Single")

plt.xlabel("Rounds")
plt.ylabel("Average Reward")
plt.title("FL Methods Comparison")
plt.legend()
plt.grid(True)

plt.show()
