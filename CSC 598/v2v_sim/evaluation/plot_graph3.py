import numpy as np
import matplotlib.pyplot as plt

fedavg = np.load("results/fedavg.npy")

smoothed = np.convolve(fedavg, np.ones(3)/3, mode='valid')

rounds = range(1, len(smoothed) + 1)

plt.figure()
plt.plot(rounds, smoothed, marker='o', label="FedAvg (Smoothed)")

plt.xlabel("Rounds")
plt.ylabel("Average Reward")
plt.title("Convergence Behavior")
plt.legend()
plt.grid(True)

plt.show()
