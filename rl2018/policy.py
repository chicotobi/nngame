import abc

class Policy(abc.ABC):
  def __init__(self,states,valid_actions):
    self.states = states
    if isinstance(valid_actions,dict):
      self.actions = valid_actions
    elif isinstance(valid_actions,list):
      self.actions = {s:valid_actions for s in states}
    else:
      raise ValueError
  @abc.abstractmethod
  def eval(self,s):
    ...    
      
class UniformPolicy(Policy):
  def __init(self,states,valid_actions):    
    super().__init__(states,valid_actions)
    self.n_valid_actions = {k:len(v) for (k,v) in self.valid_actions.items()}
  def eval(self,s):
    return 1 / self.n_valid_actions[s]
  
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
  
  # def policy2eps_soft(policy,eps,valid_actions):
  #   n_valid_actions = {k:len(v) for (k,v) in valid_actions.items()}
  #   p_eps_soft = {k:1-eps+eps/v for (k,v) in n_valid_actions.items()}
  #   # Policy function pi(s)
  #   def pol(s):
  #     a0 = pp[s]
  #     if npr.rand() < p_eps_soft[s]:
  #       return a0
  #     else:
  #       while True:
  #         a = valid_actions[s][npr.choice(n_valid_actions[s])]
  #         if a != a0:
  #           return a
  #   # Policy probability function pi(a|s)
  #   # Importtant for off-policy weighted-sampling
  #   def pol_p(s,a):
  #     if a == pp[s]:
  #       return 1-eps+eps/n_valid_actions[s]
  #     else:
  #       return eps/n_valid_actions[s]
  #   return pol, pol_p
  
  
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
