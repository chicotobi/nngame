import abc
import numpy as np
import numpy.random as npr

def argmax(dct):
  v=list(dct.values())
  return list(dct.keys())[v.index(max(v))]    

def all_argmax(dct):
  vmax = -np.Infinity
  ans = []
  for (k,v) in dct.items():
    if v == vmax:
      ans += [k]
    if v > vmax:
      ans = [k]
      vmax = v
  return ans

class Policy(abc.ABC):
  def __init__(self,states,valid_actions):
    self.states = states
    if isinstance(valid_actions,dict):
      self.actions = valid_actions
    elif isinstance(valid_actions,list):
      self.actions = {s:valid_actions for s in states}
    else:
      raise ValueError
    self.n_valid_actions = {s:len(a) for (s,a) in self.actions.items()}  
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
    return npr.choice(self.valid_actions[s])

def transform_Q_to_BestAction(Q):
  return {s:all_argmax(Q[s]) for s in Q.keys()}

class BestActionPolicy(Policy):
  def __init__(self,states,valid_actions,best_actions):  
    super().__init__(states,valid_actions)
    self.best_actions = best_actions
    self.n_best_actions = {s:len(a) for (s,a) in best_actions.items()}
  def prob(self,a,s):
    if a in self.array[s]:
      return 1 / self.n_best_actions[s]
    return 0
  def get(self,s):
    return npr.choice(self.best_actions[s])
  
class DeterministicPolicy(Policy):
  def __init__(self,states,valid_actions,best_actions):  
    super().__init__(states,valid_actions)
    self.det_actions = {s:a[0] for (s,a) in best_actions.items()}
  def prob(self,a,s):
    return self.det_actions[s] == a
  def get(self,s):
    return self.det_actions[s]
  def update(self,s,a):
    self.det_actions[s] = a
  
class EpsSoft(Policy):
  def __init__(self,states,valid_actions,eps,det_policy):
    super().__init__(states,valid_actions)
    self.eps
    self.p_eps_soft = {k:1-eps+eps/v for (k,v) in self.n_valid_actions.items()}
    self.det_policy = det_policy
    self.remaining_actions = {s:[a for a in self.n_valid_actions if a!=det_policy.get[s]] for s in states}
  def prob(self,a,s):
    if a == self.det_policy.get[s]:
      return 1-self.eps+self.eps/self.n_valid_actions[s]
    else:
      return self.eps/self.n_valid_actions[s]
  def get(self,s):
    a0 = self.det_policy.get[s]
    if npr.rand() < self.p_eps_soft[s]:
      return a0
    else:
      return npr.choice(self.remaining_actions[s])

# def Q2eps_soft(Q,eps,valid_actions):
#   n_valid_actions = {k:len(v) for (k,v) in valid_actions.items()}
#   p_eps_soft = {k:1-eps+eps/v for (k,v) in n_valid_actions.items()}
    
#   # Policy function pi(s)
#   def tmp(s):
#     a0 = argmax(Q[s])
#     if npr.rand() < p_eps_soft[s]:
#       return a0
#     else:
#       while True:
#         a = valid_actions[s][npr.choice(n_valid_actions[s])]
#         if a != a0:
#           return a
#   # Policy probability function pi(a|s)
#   # Importtant for off-policy weighted-sampling
#   def tmp2(s,a):
#     if a == argmax(Q[s]):
#       return 1-eps+eps/n_valid_actions[s]
#     else:
#       return eps/n_valid_actions[s]
#   return tmp, tmp2
  
# def stochastic2deterministic(states,actions,policy_p):
#   dct = {}
#   for s in states:
#     largest_p = 0
#     for a in actions:
#       if policy_p(a,s) > largest_p:
#         largest_p = policy_p(a,s)
#         dct[s] = a
#   def policy(s):
#     return dct[s]
#   return policy


