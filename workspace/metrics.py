from collections import defaultdict


class MetricTracker:

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = defaultdict(float)
        self.count = 0

    def update(self, metrics: dict[str, float]):
        for key, value in metrics.items():
            self.values[key] += float(value)

        self.count += 1

    def compute(self):
        return {key: value / self.count for key, value in self.values.items()}
