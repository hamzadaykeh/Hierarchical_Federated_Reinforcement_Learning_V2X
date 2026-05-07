# ==============================
# SMOOTHED PLOT (OPTIONAL)
# ==============================
import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, window=3):
    return np.convolve(data, np.ones(window)/window, mode='valid')

federatedAvg = np.load("results/fedAvg.npy")
federatedprox = np.load("results/fedprox.npy")
hierarchical = np.load("results/hierarchical_fedrl.npy")
federatedmedian = np.load("results/fedmedian.npy")


plt.figure()

plt.plot(moving_average(federatedAvg), label="federatedAvg PPO")
plt.plot(moving_average(federatedprox), label="Federatedprox RL")
plt.plot(moving_average(hierarchical), label="Hierarchical Federated RL")

plt.xlabel("Global Round")
plt.ylabel("Smoothed Reward")
plt.title("Smoothed Convergence Comparison")
plt.legend()
plt.grid(True)

plt.show()
