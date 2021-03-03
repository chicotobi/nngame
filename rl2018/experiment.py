from policy import EpsSoft

class Experiment:
  def __init__(self, environment, agent, algorithm=None):
    self.env = environment
    self.agent = agent
    self.varying_alpha = False
    if algorithm == "MC Prediction":
      self.policy_type = "On-policy"
      self.update_inside = False
      self.update_state_value = True
      self.update_action_value = False
      self.update_policy = False
      self.varying_alpha = True
    elif algorithm == "MC ES":
      self.policy_type = "On-policy"
      self.update_inside = False
      self.update_state_value = False
      self.update_action_value = True
      self.update_policy = True
      self.varying_alpha = False
    elif algorithm == "MC Off-policy":
      self.policy_type = "Off-policy"
      self.update_inside = False
      self.eps = 0.05
      self.varying_alpha = False    
      self.update_state_value = False
      self.update_action_value = True
      self.update_policy = True
    elif algorithm == "SARSA":
      self.policy_type = "On-policy"
      self.update_inside = True
      self.update_state_value = False
      self.update_action_value = True
      self.update_policy = True
      self.varying_alpha = False
    elif algorithm == "Qlearn":
      self.policy_type = "Off-policy"
      self.update_inside = True
      self.eps = 0.05
      self.varying_alpha = False    
      self.update_state_value = False
      self.update_action_value = True
      self.update_policy = True
      self.agent.qlearn = True
    if self.varying_alpha:
      self.NV = {s:0 for s in self.env.states}
      self.NQ = {s:{a:0 for a in self.agent.p.valid_actions[s]} for s in self.env.states}
    self.fixed_initial_state = None
    self.keep_episode = not self.update_inside
    
  def episode(self):
    if self.fixed_initial_state:
      s = self.fixed_initial_state
    else:
      s = self.env.get_random_initial_state()
    a = self.b.get(s)
    ep = []
    while True:
      if self.policy_type=="Off-policy" and self.update_inside:
        self.b = EpsSoft(self.env.states,self.env.valid_actions,self.eps,self.agent.p)         
      s_prime,r = self.env.step(s,a)
      if s_prime:
        a_prime = self.b.get(s_prime)
      if self.update_inside:
        if self.update_state_value:
          if self.varying_alpha:
            self.varying_alpha_state(s)
          self.agent.update_state_value_r(s,a,r,s_prime)
        if self.update_action_value:
          if self.varying_alpha:
            self.varying_alpha_action(s,a)
          if s_prime:
            self.agent.update_action_value_r(s,a,r,s_prime,a_prime)
        if self.update_policy:
          self.agent.update_policy(s)
      if self.keep_episode:
        ep.append((s,a,r))
      if not s_prime:
        break
      s = s_prime
      a = a_prime
    return ep  
    
  def train(self, n_episodes, callback):
    n_episodes = int(n_episodes)
    if self.policy_type == "On-policy":
      self.b = self.agent.p
    elif self.policy_type == "Off-policy": 
      C = {s:{a:0 for a in self.agent.p.valid_actions[s]} for s in self.env.states}  
    for i in range(n_episodes):
      if self.policy_type == "Off-policy":
        self.b = EpsSoft(self.env.states,self.env.valid_actions,self.eps,self.agent.p)          
      ep = self.episode()
      if not self.update_inside:
        G = 0
        W = 1
        for (s,a,r) in ep[::-1]:
          G = self.agent.gamma * G + r
          if self.policy_type == "Off-policy":                  
            C[s][a] += W
            self.agent.alpha = W/C[s][a]
          if self.update_state_value:
            if self.varying_alpha:
              self.varying_alpha_state(s)
            self.agent.update_state_value(G,s)
          if self.update_action_value:
            if self.varying_alpha:
              self.varying_alpha_action(s,a)
            self.agent.update_action_value(G,s,a)
          if self.update_policy:
            self.agent.update_policy(s)
          if self.policy_type == "Off-policy": 
            if self.agent.p.get(s) != a:
              break           
            W *= 1/self.b.prob(a,s)
      if callback:
        callback(i,ep)
        
  def varying_alpha_state(self,s):    
    self.NV[s] += 1
    self.agent.alpha = 1/self.NV[s]
    
  def varying_alpha_action(self,s,a):        
    self.NQ[s][a] += 1
    self.agent.alpha = 1/self.NQ[s][a]