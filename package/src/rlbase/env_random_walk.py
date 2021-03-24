from environment import BaseEnvironment

import numpy as np

class RandomWalkEnvironment(BaseEnvironment):

    def __init__(self):
        pass

    def env_init(self, env_info={}):
      self.n = env_info.get("nmax",5  )
      self.p = env_info.get("p",.5)
      
      self.states = list(range(self.n+2))
      self.terminal_states = [0,self.n+1]
      self.actions = [0]
      self.rewards = [0,1]

    def env_step(self,s,a):
      if np.random.rand() > self.p:
        s += 1
      else:
        s -= 1
      if s == self.n + 1:
        return 1, None, True
      elif s == 0:
        return 0, None, True
      else:
        return 0, s, False
      
    def state_transition(self,s_prime,r,s,a):
      if s_prime == s + 1:
        if s_prime == self.n + 1 and r == 1:
          return self.p
        if r == 0:
          return self.p
      if s_prime == s - 1:
        if r == 0:
          return 1 - self.p
      return 0