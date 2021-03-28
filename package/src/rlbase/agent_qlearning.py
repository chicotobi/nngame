from misc import argmax_dct

from agent import BaseAgent

class QlearningAgent(BaseAgent):
  def __init__(self,**kwargs):
    super().__init__(**kwargs)
    self.alpha = kwargs.get("alpha",0.5)

  def start(self, s):
    self.last_state = s
    self.last_action = self.pi.get(s)
    return self.last_action

  def step(self, r, s):
    a = self.pi.get(s)
    a1 = argmax_dct(self.q[self.s])
    self.q[self.last_state][self.last_action] += self.alpha * (r + self.gamma * self.q[s][a1] - self.q[self.last_state][self.last_action])

    a0 = argmax_dct(self.q[self.last_state])
    self.pi.update(self.last_state,a0)

    self.last_state = s
    self.last_action = a
    return self.last_action

  def end(self, r):
    self.q[self.last_state][self.last_action] += self.alpha * (r - self.q[self.last_state][self.last_action])

    a0 = argmax_dct(self.q[self.last_state])
    self.pi.update(self.last_state,a0)