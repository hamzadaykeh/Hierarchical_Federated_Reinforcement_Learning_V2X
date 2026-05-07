import numpy as np
import matplotlib.pyplot as plt
import os



RESULTS_DIR = "results"

files = {
    "HFRL": "hierarchical_fedrl.npy",
    "FedAvg": "fedAvg.npy",
    "FedProx": "fedprox.npy",
    "FedMedian": "fedmedian.npy"
}

data = {}
for method, file in files.items():
    data[method] = np.load(os.path.join(RESULTS_DIR, file))

methods = list(data.keys())


scalability = [np.mean(v[-5:]) for v in data.values()]

communication = [120, 200, 180, 170]

reliability = [1 / (np.std(v) + 1e-6) for v in data.values()]


colors = ['green', 'royalblue', 'orange', 'red']


bar_width = 0.38
x = np.arange(len(methods))

plt.figure(figsize=(8,5))
plt.bar(x, scalability, width=bar_width, color=colors, edgecolor='black')
plt.xticks(x, methods)
plt.title("Scalability Comparison")
plt.xlabel("Methods")
plt.ylabel("Average Final Reward")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("hist_scalability.png", dpi=300)
plt.show()


plt.figure(figsize=(8,5))
plt.bar(x, communication, width=bar_width, color=colors, edgecolor='black')
plt.xticks(x, methods)
plt.title("Communication Cost Comparison")
plt.xlabel("Methods")
plt.ylabel("Communication Cost")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("hist_communication.png", dpi=300)
plt.show()

plt.figure(figsize=(8,5))
plt.bar(x, reliability, width=bar_width, color=colors, edgecolor='black')
plt.xticks(x, methods)
plt.title("Reliability Comparison")
plt.xlabel("Methods")
plt.ylabel("Reliability Score")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("hist_reliability.png", dpi=300)
plt.show()