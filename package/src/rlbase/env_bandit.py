from environment import BaseEnvironment

import numpy as np

class BanditEnvironment(BaseEnvironment):

    def __init__(self):
        self.reward_obs_term = (None, None, None)
        self.count = 0
        self.arms = []

    def env_init(self, env_info={"N":10}):
        self.N = env_info["N"]
        self.states = [0]
        self.actions = list(range(self.N))
        self.arms = np.random.randn(self.N)
        self.reward_obs_term = (0, self.states[0], False)
        self.random = env_info.get("random",False)
        if env_info.get("offset"):
          self.arms += env_info.get("offset")

    def env_start(self):
        return self.reward_obs_term[1]

    def env_step(self, action):
        reward = self.arms[action] + np.random.randn()
        self.reward_obs_term = (reward, self.states[0], False)
        if self.random:
          self.arms += 0.01 * np.random.randn(self.N)
        return self.reward_obs_term

    def env_cleanup(self):
        pass

    def env_message(self, message):
        pass
