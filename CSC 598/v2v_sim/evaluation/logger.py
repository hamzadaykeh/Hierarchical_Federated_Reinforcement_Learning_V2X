import pandas as pd
#pip install pandas matplotlib

class MetricsLogger:
    def __init__(self):
        self.data = {
            "step": [],
            "speed": [],
            "distance": [],
            "reward": []
        }

    def log(self, step, speed, distance, reward):
        self.data["step"].append(step)
        self.data["speed"].append(speed)
        self.data["distance"].append(distance)
        self.data["reward"].append(reward)

    def save(self, filename):
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)
        print(f"Metrics saved to {filename}")
