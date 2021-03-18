import abc
import misc
import numpy.random as npr

def transform_Q_to_BestAction(Q):
  return {s:misc.all_argmax(Q[s]) for s in Q.keys()}

class Policy(abc.ABC):
  def __init__(self,states,valid_actions):
    self.states = states
    if isinstance(valid_actions,dict):
      self.valid_actions = valid_actions
    elif isinstance(valid_actions,list):
      self.valid_actions = {s:valid_actions for s in states}
    else:
      raise ValueError
    self.n_valid_actions = {s:len(a) for (s,a) in self.valid_actions.items()}  
  @abc.abstractmethod
  def prob(self,a,s):
    ...
  @abc.abstractmethod
  def get(self,s):
    ...      
      
class UniformPolicy(Policy):
  def __init__(self,states,valid_actions):  
    super().__init__(states,valid_actions)
  def prob(self,a,s):
    return 1 / self.n_valid_actions[s]
  def get(self,s):
    return misc.sample(self.valid_actions[s])

class BestActionPolicy(Policy):
  def __init__(self,states,valid_actions,best_actions):  
    super().__init__(states,valid_actions)
    self.best_actions = best_actions
    self.n_best_actions = {s:len(a) for (s,a) in best_actions.items()}
  def prob(self,a,s):
    if a in self.best_actions[s]:
      return 1 / self.n_best_actions[s]
    return 0
  def get(self,s):
    return misc.sample(self.best_actions[s])
  
class DeterministicPolicy(Policy):
  def __init__(self,states,valid_actions,best_actions=None):  
    super().__init__(states,valid_actions)
    if best_actions:
      self.det_actions = {s:a[0] for (s,a) in best_actions.items()}
    else:
      self.det_actions = {s:self.valid_actions[s][0] for s in states}      
  def prob(self,a,s):
    return self.det_actions[s] == a
  def get(self,s):
    return self.det_actions[s]
  def update(self,s,a):
    self.det_actions[s] = a
  
class EpsSoft(Policy):
  def __init__(self,states,valid_actions,eps,det_policy=None):
    super().__init__(states,valid_actions)
    self.eps = eps
    self.p_eps_soft = {k:1-eps+eps/v for (k,v) in self.n_valid_actions.items()}
    if det_policy:
      self.det_policy = det_policy
    else:
      self.det_policy = DeterministicPolicy(states, self.valid_actions)
    self.remaining_actions = {s:[a for a in self.valid_actions[s] if a != self.det_policy.get(s)] for s in states}
  def prob(self,a,s):
    if a == self.det_policy.get(s):
      return 1-self.eps+self.eps/self.n_valid_actions[s]
    else:
      return self.eps/self.n_valid_actions[s]
  def get(self,s):
    a0 = self.det_policy.get(s)
    if npr.rand() < self.p_eps_soft[s]:
      return a0
    else:
      return misc.sample(self.remaining_actions[s])
  def update(self,s,a0):
    self.det_policy.det_actions[s] = a0
    self.remaining_actions[s] = [a for a in self.det_policy.valid_actions[s] if a != a0]