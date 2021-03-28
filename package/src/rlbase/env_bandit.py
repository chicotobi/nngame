from environment import BaseEnvironment

import numpy as np

class BanditEnvironment(BaseEnvironment):

  def __init__(self,**kwargs):
    self.N = kwargs.get("N",10)
    
    self.states = [0]
    self.actions = list(range(self.N))
    self.set_all_actions_valid()
    
    self.arms = np.random.randn(self.N) + kwargs.get("offset",0)
    self.random = kwargs.get("random",False)

  def start(self):
    return self.states[0]

  def step(self, s, a):
    if self.random:
      self.arms += 0.01 * np.random.randn(self.N)
    r = self.arms[a] + np.random.randn()
    return r, self.states[0], False