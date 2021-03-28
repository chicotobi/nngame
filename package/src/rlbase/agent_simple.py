from agent import BaseAgent
from misc import argmax_unique

import math

class SimpleAgent(BaseAgent):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.alpha = kwargs.get("alpha")
    self.ucb_c = kwargs.get("ucb_c")

  def start(self, s):
      self.t = 0
      self.last_action =self.pi.get(s)
      self.last_state = s
      return self.last_action

  def step(self, r, s):
    self.t += 1
    self.nv[self.last_state] += 1
    self.nq[self.last_state][self.last_action] += 1

    if self.alpha:
      stepsize = self.alpha
    else:
      stepsize = 1 / self.nq[self.last_state][self.last_action]
    self.q[self.last_state][self.last_action] += stepsize * (r - self.q[self.last_state][self.last_action])

    if self.ucb_c:
      tmp = {a : v + self.ucb_c*(math.log(self.t) / (1e-5+self.nq[self.last_state][a]))**.5 for (a,v) in self.q[self.last_state].items()}
    else:
      tmp = self.q[self.last_state]
    
    self.pi.update(self.last_state,argmax_unique(tmp))
    self.last_action = self.pi.get(s)
    return self.last_action