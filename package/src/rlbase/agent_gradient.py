from agent import BaseAgent

import numpy as np
from misc import softmax

class GradientAgent(BaseAgent):
  def __init__(self,**kwargs):
    super().__init__(**kwargs)
    self.alpha = kwargs.get("alpha")
    self.baseline = kwargs.get("baseline",0)
    self.average_reward = 0

  def start(self, s):
    self.t = 0
    self.last_state = s
    self.last_action = self.pi.get(s)
    return self.last_action

  def step(self, r, s):
    self.t += 1
  
    if self.alpha:
      stepsize = self.alpha
    else:
      stepsize = 1 / self.nq[self.last_state][self.last_action]
    if self.baseline:
      self.average_reward += (r - self.average_reward) / self.t
  
    one_hot = np.zeros(self.pi.na)
    one_hot[self.last_action] = 1
      
    self.pi.h[self.last_state] += stepsize * (r - self.average_reward) * ( one_hot - softmax(self.pi.h[self.last_state]))
    
    self.pi.update(self.last_state)
    
    self.last_state = s        
    self.last_action = self.pi.get(s)
    return self.last_action