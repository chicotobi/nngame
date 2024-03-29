import numpy as np
import numpy.random as npr

import rlbase.misc as misc

from copy import deepcopy

def transform_Q_to_BestAction(Q):
  return {s:misc.all_argmax(Q[s]) for s in Q.keys()}

class Policy:
  def __init__(self,**kwargs):
    self.valid_actions = kwargs.get("env").valid_actions
    self.n_valid_actions = {s:len(a) for (s,a) in self.valid_actions.items()}  
  def prob(self,a,s):
    pass
  def get(self,s):
    pass
      
class UniformPolicy(Policy):
  def __init__(self,**kwargs):  
    super().__init__(**kwargs)
  def prob(self,a,s):
    return 1 / self.n_valid_actions[s]
  def get(self,s):
    return misc.sample(self.valid_actions[s])

class BestActionPolicy(Policy):
  def __init__(self,**kwargs):  
    super().__init__(**kwargs)
    self.best_actions = kwargs.get("best_actions")
    self.n_best_actions = {s:len(a) for (s,a) in self.best_actions.items()}
  def prob(self,a,s):
    if a in self.best_actions[s]:
      return 1 / self.n_best_actions[s]
    return 0
  def get(self,s):
    return misc.sample(self.best_actions[s])
  
class DeterministicPolicy(Policy):
  def __init__(self,**kwargs):  
    super().__init__(**kwargs)
    if kwargs.get("best_actions"):
      self.det_actions = {s:a[0] for (s,a) in kwargs.get("best_actions").items()}
    else:
      self.det_actions = {s:self.valid_actions[s][0] for s in self.valid_actions.keys()}      
  def prob(self,a,s):
    return self.det_actions[s] == a
  def get(self,s):
    return self.det_actions[s]
  def update(self,s,a):
    self.det_actions[s] = a
  
class EpsGreedy(Policy):
  def __init__(self,**kwargs):
    super().__init__(**kwargs)
    self.eps = kwargs.get("eps")
    if kwargs.get("det_policy"):
      pol = kwargs.get("det_policy")
      self.best_actions = {k:[v] for (k,v) in pol.det_actions.items()}
    else:
      self.best_actions = deepcopy(self.valid_actions)
  def prob(self,a,s):
    if a in self.best_actions[s]:
      return self.eps/self.n_valid_actions[s] + (1-self.eps) / len(self.best_actions[s])
    else:
      return self.eps/self.n_valid_actions[s]
  def get(self,s):
    if npr.rand() > self.eps:
      return misc.sample(self.best_actions[s])
    else:
      return misc.sample(self.valid_actions[s])
  def update(self,s,best_a):
    self.best_actions[s] = best_a
    
class Softmax(Policy):
  def __init__(self,**kwargs):
    super().__init__(**kwargs)
    env = kwargs.get("env")
    self.na = len(env.actions)
    self.h = {s:[0]*self.na for s in env.states}
    self.p = {s:[1/self.na]*self.na for s in env.states}
  def prob(self,a,s):
    return self.p[s][a]
  def get(self,s):
    return np.random.choice(self.na,p=self.p[s])
  def update(self,s):
    self.p[s] = misc.softmax(self.h[s])