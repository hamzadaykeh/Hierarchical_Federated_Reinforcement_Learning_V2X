import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("evaluation/single_agent_metrics.csv")

plt.figure()
plt.plot(df["step"], df["speed"])
plt.xlabel("Step")
plt.ylabel("Speed (m/s)")
plt.title("Vehicle Speed Over Time")
plt.show()

plt.figure()
plt.plot(df["step"], df["distance"])
plt.xlabel("Step")
plt.ylabel("Distance to Front Vehicle (m)")
plt.title("Safety Distance Over Time")
plt.show()

plt.figure()
plt.plot(df["step"], df["reward"])
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Reward Over Time")
plt.show()
