import numpy as np
import matplotlib.pyplot as plt

vehicles = np.array([250, 500, 750, 1000, 1250, 1500, 1750, 2000])

flat_fl_delay = np.array([116, 228, 342, 462, 585, 708, 831, 954])
rsu_layer_delay = np.array([35, 60, 85, 110, 135, 160, 185, 210])
flat_fl_cost = np.array([425, 850, 1275, 1700, 2125, 2550, 2975, 3400])
cloud_layer_cost = np.array([105, 210, 315, 420, 525, 630, 735, 840])


plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11

bar_width = 0.32
x = np.arange(len(vehicles))

plt.figure(figsize=(11,6))

bars1 = plt.bar(
    x - bar_width/2,
    flat_fl_delay,
    width=bar_width,
    color="#e85c54",
    edgecolor="black",
    label="Flat FL (Centralized)"
)

bars2 = plt.bar(
    x + bar_width/2,
    rsu_layer_delay,
    width=bar_width,
    color="#1f7a7a",
    edgecolor="black",
    label="Proposed RSU Layer"
)

# Labels above bars
for b in bars1:
    plt.text(
        b.get_x() + b.get_width()/2,
        b.get_height() + 8,
        f"{int(b.get_height())}",
        ha="center",
        color="red",
        fontsize=10
    )

for b in bars2:
    plt.text(
        b.get_x() + b.get_width()/2,
        b.get_height() + 8,
        f"{int(b.get_height())}",
        ha="center",
        color="#004d4d",
        fontsize=10
    )

plt.xticks(x, vehicles)
plt.xlabel("Number of Vehicles", fontweight="bold")
plt.ylabel("Aggregation Delay (ms)", fontweight="bold")
plt.title(
    "Regional Aggregation Performance vs Number of Vehicles\n(Aggregation Delay Comparison)",
    fontsize=14,
    fontweight="bold",
    color="#0d5c63"
)
plt.legend()
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig("figure_rsu_delay.png", dpi=400)
plt.show()


plt.figure(figsize=(11,6))

bars1 = plt.bar(
    x - bar_width/2,
    flat_fl_cost,
    width=bar_width,
    color="#e85c54",
    edgecolor="black",
    label="Flat FL (Centralized)"
)

bars2 = plt.bar(
    x + bar_width/2,
    cloud_layer_cost,
    width=bar_width,
    color="#6a4fb3",
    edgecolor="black",
    label="Proposed Cloud Layer"
)

# Labels above bars
for b in bars1:
    plt.text(
        b.get_x() + b.get_width()/2,
        b.get_height() + 45,
        f"{int(b.get_height())}",
        ha="center",
        color="red",
        fontsize=10
    )

for b in bars2:
    plt.text(
        b.get_x() + b.get_width()/2,
        b.get_height() + 45,
        f"{int(b.get_height())}",
        ha="center",
        color="#4527a0",
        fontsize=10
    )

plt.xticks(x, vehicles)
plt.xlabel("Number of Vehicles", fontweight="bold")
plt.ylabel("Global Communication Cost (MB)", fontweight="bold")
plt.title(
    "Cloud Scalability Comparison under Large Vehicular Density\n(Global Communication Cost Comparison)",
    fontsize=14,
    fontweight="bold",
    color="#4527a0"
)
plt.legend()
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig("figure_cloud_cost.png", dpi=400)
plt.show()


delay_reduction = (flat_fl_delay[-1] - rsu_layer_delay[-1]) / flat_fl_delay[-1] * 100
cost_reduction = (flat_fl_cost[-1] - cloud_layer_cost[-1]) / flat_fl_cost[-1] * 100

print("="*60)
print("RESULT SUMMARY")
print("="*60)
print(f"RSU hierarchy decreases delay by {delay_reduction:.1f}% at 2000 vehicles")
print(f"Cloud coordination reduces communication cost by {cost_reduction:.1f}% at 2000 vehicles")
print("Saved files:")
print(" - figure_rsu_delay.png")
print(" - figure_cloud_cost.png")