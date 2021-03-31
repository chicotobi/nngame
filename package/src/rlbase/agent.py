import math
import numpy as np

import rlbase.misc as misc

class BaseAgent:
  def __init__(self,**kwargs):
    self.pi = kwargs.get("pi")
    env = kwargs.get("env")
    q_init = kwargs.get("q_init",0)
    self.v = {s:0 for s in env.states}
    self.nv = {s:0 for s in env.states}
    self.q = {s:{a:q_init for a in env.valid_actions[s]} for s in env.states}
    self.nq = {s:{a:0 for a in env.valid_actions[s]} for s in env.states}

  def start(self, s):
    return self.pi.get(s)

  def step(self, r, s):
    return self.pi.get(s)

  def end(self, r):
    pass

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

    self.pi.update(self.last_state,misc.argmax_unique(tmp))
    self.last_action = self.pi.get(s)
    return self.last_action

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

    self.pi.h[self.last_state] += stepsize * (r - self.average_reward) * ( one_hot - misc.softmax(self.pi.h[self.last_state]))

    self.pi.update(self.last_state)

    self.last_state = s
    self.last_action = self.pi.get(s)
    return self.last_action

class SarsaAgent(BaseAgent):
  def __init__(self,**kwargs):
    super().__init__(**kwargs)
    self.alpha = kwargs.get("alpha",0.5)
    env = kwargs.get("env")
    self.q = {s:{a:0 for a in env.valid_actions[s]} for s in env.states}

  def start(self, s):
    self.last_state = s
    self.last_action = self.pi.get(s)
    return self.last_action

  def step(self, r, s):
    a = self.pi.get(s)

    self.q[self.last_state][self.last_action] += self.alpha * (r + self.gamma * self.q[s][a] - self.q[self.last_state][self.last_action])

    a0 = misc.argmax_unique(self.q[self.last_state])
    self.pi.update(self.last_state,a0)

    self.last_state = s
    self.last_action = a
    return self.last_action

  def end(self, r):
    self.q[self.last_state][self.last_action] += self.alpha * (r - self.q[self.last_state][self.last_action])

    a0 = misc.argmax_unique(self.q[self.last_state])
    self.pi.update(self.last_state,a0)

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
    a1 = misc.argmax_unique(self.q[s])
    self.q[self.last_state][self.last_action] += self.alpha * (r + self.gamma * self.q[s][a1] - self.q[self.last_state][self.last_action])

    a0 = misc.argmax_unique(self.q[self.last_state])
    self.pi.update(self.last_state,a0)

    self.last_state = s
    self.last_action = a
    return self.last_action

  def end(self, r):
    self.q[self.last_state][self.last_action] += self.alpha * (r - self.q[self.last_state][self.last_action])

    a0 = misc.argmax_unique(self.q[self.last_state])
    self.pi.update(self.last_state,a0)
    
class ExpectedSarsaAgent(BaseAgent):
  def __init__(self,**kwargs):
    super().__init__(**kwargs)
    self.alpha = kwargs.get("alpha",0.5)

  def start(self, s):
    self.last_state = s
    self.last_action = self.pi.get(s)
    return self.last_action

  def step(self, r, s):
    a = self.pi.get(s)
    exp_val = sum(self.pi.prob(a1,s) * val for (a1,val) in self.q[s].items())
    self.q[self.last_state][self.last_action] += self.alpha * (r + self.gamma * exp_val - self.q[self.last_state][self.last_action])

    a0 = misc.argmax_unique(self.q[self.last_state])
    self.pi.update(self.last_state,a0)

    self.last_state = s
    self.last_action = a
    return self.last_action

  def end(self, r):
    self.q[self.last_state][self.last_action] += self.alpha * (r - self.q[self.last_state][self.last_action])

    a0 = misc.argmax_unique(self.q[self.last_state])
    self.pi.update(self.last_state,a0)