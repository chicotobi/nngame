class Agent:
  def __init__(self,states,actions, policy):
    self.p = policy
    self.V = {s:0 for s in states}
    self.Q = {(s,a):0 for s in states for a in actions}
    self.NV = {s:0 for s in states}
    self.NQ = {(s,a):0 for s in states for a in actions}
    
  def update_state_value(self,G,s,fixed_stepsize):
    if fixed_stepsize > 0:
      self.V[s] += (G-self.V[s]) * fixed_stepsize
    else:
      self.NV[s] += 1
      self.V[s] += (G-self.V[s]) / self.NV[s]
          
  def update_action_value(self,G,s,a,fixed_stepsize):
    if fixed_stepsize > 0:
      self.Q[(s,a)] += (G-self.Q[(s,a)]) * fixed_stepsize
    else:
      self.NQ[(s,a)] += 1
      self.Q[(s,a)] += (G-self.Q[(s,a)]) / self.NQ[(s,a)]

  def update_policy(self,s):
    self.p.update(s, self.Q[(s,0)]<self.Q[(s,1)])