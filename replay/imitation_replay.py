import os
import pickle
import numpy as np

from replay.basic_replay import Replay
from utility.debug_tools import assert_colorize


class ImitationReplay:
    def __init__(self, filename, capacity, batch_size, state_space, action_dim):
        self.memory = dict(state=np.zeros((capacity, *state_space)), action=np.zeros((capacity, action_dim)))
        self.filename = filename
        self.batch_size = batch_size
        self.capacity = capacity
        self.idx = 0
        self.full = False

        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.loads(f.read())

            self.memory['state'] = data['state']
            self.memory['action'] = data['action']
            
            assert_colorize(len(self.memory['state']) == capacity)
            assert_colorize(len(self.memory['action']) == capacity)

            self.full = True

            assert_colorize(len(self.memory) == 2, 'Memory contains redundant data')
            assert_colorize(len(self.memory['state']) == len(self.memory['action']), 
                            f"Inconsistent lengths. #state: {len(self.memory['state'])}\t#action: {len(self.memory['action'])}")

    def __call__(self):
        while True:
            yield self.sample()

    def add(self, state, action):
        self.memory['state'][self.idx] = state
        self.memory['action'][self.idx] = action

        self.idx = (self.idx + 1) % self.capacity

    def sample(self):
        idx = np.random.randint(0, self.capacity, self.batch_size)

        return (
            self.memory['state'][idx],
            self.memory['action'][idx]
        )
