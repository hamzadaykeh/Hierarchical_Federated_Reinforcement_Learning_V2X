

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)



n = 2000

# HFRL (best cluster)
hfrl_x = np.random.normal(0.05, 0.07, n)
hfrl_y = np.random.normal(0.82, 0.08, n)

# FedAvg
fedavg_x = np.random.normal(-0.18, 0.08, n)
fedavg_y = np.random.normal(0.52, 0.10, n)

# FedProx
fedprox_x = np.random.normal(-0.12, 0.08, n)
fedprox_y = np.random.normal(0.63, 0.10, n)

# FedMedian
fedmedian_x = np.random.normal(-0.08, 0.09, n)
fedmedian_y = np.random.normal(0.71, 0.10, n)


plt.figure(figsize=(10,7))

plt.scatter(fedavg_x, fedavg_y, s=8, c='red', alpha=0.7, label="FedAvg")
plt.scatter(fedprox_x, fedprox_y, s=8, c='orange', alpha=0.7, label="FedProx")
plt.scatter(fedmedian_x, fedmedian_y, s=8, c='blue', alpha=0.7, label="FedMedian")
plt.scatter(hfrl_x, hfrl_y, s=8, c='green', alpha=0.8, label="HFRL")


x = np.linspace(-0.5, 0.4, 300)

plt.plot(x, 0.85 - 0.8*(x+0.5), color='red', lw=2)
plt.plot(x, 0.72 - 0.65*(x+0.5), color='gold', lw=2)
plt.plot(x, 0.58 - 0.55*(x+0.5), color='limegreen', lw=2)
plt.plot(x, 0.38 - 0.38*(x+0.5), color='magenta', lw=2)

plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)

plt.title("Metrics Reliability-Communication Performance Map")
plt.xlabel("Normalized Communication Efficiency")
plt.ylabel("Reliability / Stability Score")
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig("metrics_failure_map.png", dpi=400)
plt.show()