
import random

class ClassicalBaseline:
    def __init__(self, pools=3):
        self.pools = pools
        self.counts = {p: 0 for p in range(1, pools + 1)}
        self.successes = {p: 0 for p in range(1, pools + 1)}
        
    def select_pool(self):
        # Epsilon-greedy
        if random.random() < 0.1:
            return random.randint(1, self.pools)
        return max(self.successes, key=self.successes.get)
        
    def update(self, pool, reward):
        self.counts[pool] += 1
        if reward > 0.5: # Simple threshold
            self.successes[pool] += 1

