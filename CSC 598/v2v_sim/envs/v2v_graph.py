import networkx as nx
import numpy as np

class V2VGraph:
    def __init__(self, communication_range=30.0):
        self.range = communication_range

    def build_graph(self, positions):
        """
        positions: array of vehicle positions
        """
        G = nx.Graph()
        num_vehicles = len(positions)

        for i in range(num_vehicles):
            G.add_node(i)

        for i in range(num_vehicles):
            for j in range(i + 1, num_vehicles):
                distance = abs(positions[i] - positions[j])

                if distance <= self.range:
                    weight = 1.0 / (distance + 1e-6)
                    G.add_edge(i, j, weight=weight)

        return G
