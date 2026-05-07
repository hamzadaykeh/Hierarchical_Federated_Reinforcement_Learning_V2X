import numpy as np
import matplotlib.pyplot as plt

# ==============================
# LOAD RESULTS
# ==============================

federatedAvg = np.load("results/fedAvg.npy")
federatedprox = np.load("results/fedprox.npy")
hierarchical = np.load("results/hierarchical_fedrl.npy")
federatedmedian = np.load("results/fedmedian.npy")

# ==============================
# PLOT REWARD CURVES
# ==============================

plt.figure()

plt.plot(federatedAvg, label="fedAvg PPO")
plt.plot(federatedprox, label=" Federatedprox RL")
plt.plot(hierarchical, label="Hierarchical Federated RL")
plt.plot(federatedmedian, label=" Federatedmedian")

plt.xlabel("Global Round")
plt.ylabel("Average Reward")
plt.title("Performance Comparison")
plt.legend()

plt.grid(True)
plt.show()
