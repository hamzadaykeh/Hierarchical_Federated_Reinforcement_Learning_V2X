import matplotlib.pyplot as plt

labels = ['HFRL','FedAvg','FedProx','FedMedian']
cost = [120,200,180,170]

plt.figure(figsize=(7,7))
plt.pie(cost, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("Communication Cost Distribution")

plt.savefig("piechart_cost.png", dpi=300)
plt.show()