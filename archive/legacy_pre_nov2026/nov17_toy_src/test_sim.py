
import sys
import os
sys.path.append(os.getcwd())

from src.environment import ParadigmShiftSimulator
from src.baseline_learner import ClassicalBaseline
import json

# Setup
with open('configs/paradigm_probe.json', 'w') as f:
    json.dump({'shift_tick': 20, 'pools': 3}, f)

env = ParadigmShiftSimulator('configs/paradigm_probe.json')
learner = ClassicalBaseline(pools=3)

# Simple run loop
for i in range(40):
    job = {'size': 0.2, 'complexity': 0.8} # Paradigm A likes size, B likes complexity
    action = {'pool': learner.select_pool()}
    result = env.step(job, action)
    learner.update(action['pool'], result['reward'])
    print(f'Tick {i}: Paradigm {result['paradigm']}, Reward {result['reward']}')

