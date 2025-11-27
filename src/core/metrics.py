import csv
import time
from typing import Dict, Any

class MetricsLogger:
    def __init__(self, filepath: str = "metrics.csv"):
        self.filepath = filepath
        self.data = []
        self.start_time = time.time()

    def log(self, tick: int, metrics: Dict[str, Any]):
        """
        Log a set of metrics for a specific tick.
        """
        entry = {"tick": tick, "timestamp": time.time() - self.start_time}
        entry.update(metrics)
        self.data.append(entry)

    def save(self):
        """
        Save logged data to CSV.
        """
        if not self.data:
            return
        
        keys = self.data[0].keys()
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.data)
