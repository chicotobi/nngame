from environment import BaseEnvironment

import numpy as np

class Bandit(BaseEnvironment):

    def __init__(self):
        self.actions = [0]
        self.reward_obs_term = (None, None, None)
        self.count = 0
        self.arms = []

    def env_init(self, env_info={"N":10}):
        self.arms = np.random.randn(env_info["N"])
        self.reward_obs_term = (0, self.actions[0], False)

    def env_start(self):
        return self.reward_obs_term[1]

    def env_step(self, action):
        reward = self.arms[action] + np.random.randn()
        obs = self.reward_obs_term[1]
        self.reward_obs_term = (reward, obs, False)
        return self.reward_obs_term

    def env_cleanup(self):
        pass

    def env_message(self, message):
        pass
