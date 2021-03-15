import misc

class Agent:
  def __init__(self, states, actions, policy, alpha = 1, gamma = 1, Qinit=0):
    self.p = policy
    self.V = {s:0 for s in states}
    if isinstance(actions,list):
      self.Q = {s:{a:Qinit for a in actions} for s in states}
    elif isinstance(actions,dict):
      self.Q = {s:{a:Qinit for a in actions[s]} for s in states}
    else:
      raise ValueError    
    self.alpha = alpha
    self.gamma = gamma
    
  def update_state_value(self,G,s):
    self.V[s] += ( G - self.V[s] ) * self.alpha
    
  def update_state_value_r(self,s,a,r,s_prime):
    self.V[s] += ( r + self.gamma * self.V[s_prime] - self.V[s] ) * self.alpha
          
  def update_action_value(self,G,s,a):
    self.Q[s][a] += ( G - self.Q[s][a] ) * self.alpha
    
  def update_action_value_r(self,s,a,r,s_prime,a_prime):
    self.Q[s][a] += ( r + self.gamma * self.Q[s_prime][a_prime] - self.Q[s][a] ) * self.alpha

  def update_policy(self,s):    
    a = misc.argmax(self.Q[s])
    self.p.update(s, a)