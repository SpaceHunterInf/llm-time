
import json
import random

class ParadigmShiftSimulator:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.tick = 0
        self.paradigm = 'A' 
        
    def step(self, job_attributes, action):
        # job_attributes: {'size': float, 'complexity': float}
        # action: {'pool': int}
        self.tick += 1
        if self.tick > self.config.get('shift_tick', 50):
            self.paradigm = 'B'
            
        reward = self._calculate_reward(job_attributes, action)
        return {'reward': reward, 'paradigm': self.paradigm}

    def _calculate_reward(self, job, action):
        # Paradigm A: Size-dependent processing
        if self.paradigm == 'A':
            # Efficiency is high if action.pool matches job.size bracket
            match = (action['pool'] == 1 and job['size'] < 0.3) or \
                    (action['pool'] == 2 and 0.3 <= job['size'] < 0.7) or \
                    (action['pool'] == 3 and job['size'] >= 0.7)
            return 1.0 if match else 0.1
        # Paradigm B: Complexity-dependent processing
        else:
            match = (action['pool'] == 1 and job['complexity'] < 0.3) or \
                    (action['pool'] == 2 and 0.3 <= job['complexity'] < 0.7) or \
                    (action['pool'] == 3 and job['complexity'] >= 0.7)
            return 1.0 if match else 0.1

