import tqdm

import rlbase.misc as misc
import rlbase.policy as policy

class BaseExperiment:
    def __init__(self, **kwargs):
        self.env = kwargs.get("env")
        self.agent = kwargs.get("agent")
        self.gamma = kwargs.get("gamma",1)
        self.n_episodes = int(kwargs.get("n_episodes"))
        self.callback = kwargs.get("callback")
        self.agent.gamma = self.gamma
        self.disable = not kwargs.get("show_progress",True)
        
    def episode(self):
      pass
    
    def train(self):
      pass

class MC_EveryVisitExperiment(BaseExperiment):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.n_visits = {s:0 for s in self.env.states}
    self.V = {s:0 for s in self.env.states}

  def episode(self):
    s = self.env.get_initial_state()
    a = self.agent.start(s)
    ep = []
    while True:
      r, s_prime, terminal = self.env.step(s,a)
      ep.append((s,a,r))
      if terminal:
        self.agent.end(r)
        break
      a = self.agent.step(r,s_prime)
      s = s_prime
    return ep

  def train(self):
    for i in tqdm.tqdm(range(self.n_episodes),disable=self.disable):
      ep = self.episode()
      if self.callback:
        self.callback(i,ep)
      G = 0
      for (s,a,r) in ep[::-1]:
        G = self.gamma * G + r
        self.n_visits[s] += 1
        self.V[s] += 1. / self.n_visits[s] * (G - self.V[s])

class MC_ExploringStartsExperiment(BaseExperiment): 
  def __init__(self, **kwargs):
    super().__init__(**kwargs)    
    self.n_visits = {s:{a:0 for a in self.env.actions} for s in self.env.states}
    self.Q = {s:{a:0 for a in self.env.actions} for s in self.env.states}
    
  def episode(self):
    s = self.env.get_initial_state()
    a = self.agent.start(s)
    ep = []
    while True: 
      r, s_prime, terminal = self.env.step(s,a)
      ep.append((s,a,r))
      if terminal:
        self.agent.end(r)
        break
      a = self.agent.step(r,s_prime)
      s = s_prime
    return ep  
    
  def train(self):
    for i in tqdm.tqdm(range(self.n_episodes),disable=self.disable):
      ep = self.episode()
      G = 0
      set_sa = [(s,a) for (s,a,r) in ep[::-1]]
      for idx, (s,a,r) in enumerate(ep[::-1]):
        G = self.gamma * G + r
        if (s,a) not in set_sa[idx+1:]:
          self.n_visits[s][a] += 1
          self.Q[s][a] += 1. / self.n_visits[s][a] * (G - self.Q[s][a])
          a0 = misc.argmax_unique(self.Q[s])
          self.agent.pi.update(s,a0)

class MC_OffPolicyExperiment(BaseExperiment):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.n_visits = {s:{a:0 for a in self.env.valid_actions[s]} for s in self.env.states}
    q_init = kwargs.get("q_init",0)    
    self.eps = kwargs.get("eps",0)       
    self.Q = {s:{a:q_init for a in self.env.valid_actions[s]} for s in self.env.states}
    self.C = {s:{a:0 for a in self.env.valid_actions[s]} for s in self.env.states}
    
  def episode(self):
    s = self.env.get_initial_state()
    a = self.b.get(s)
    ep = []
    while True: 
      r, s_prime, terminal = self.env.step(s,a)
      ep.append((s,a,r))
      if terminal:
        break
      a = self.b.get(s_prime)
      s = s_prime
    return ep  
    
  def train(self):
    for i in tqdm.tqdm(range(self.n_episodes),disable=self.disable):
      self.b = policy.EpsGreedy(env=self.env,eps=self.eps,det_policy=self.agent.pi)
      ep = self.episode()
      if self.callback:
        self.callback(i,ep)
      G = 0
      W = 1
      for (s,a,r) in ep[::-1]:
        G = self.gamma * G + r
        self.C[s][a] += W
        self.Q[s][a] += W/self.C[s][a] * (G - self.Q[s][a])
        a0 = misc.argmax_unique(self.Q[s])
        self.agent.pi.update(s,a0)
        if a0 != a:
          break
        W *= 1/self.b.prob(a,s)

class TD_CtrlExperiment(BaseExperiment):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    
  def episode(self):
    s = self.env.get_initial_state()
    a = self.agent.start(s)
    ep = []
    while True: 
      r, s_prime, terminal = self.env.step(s,a)
      ep.append((s,a,r))
      if terminal:
        self.agent.end(r)
        break
      a = self.agent.step(r,s_prime)
      s = s_prime
    return ep
    
  def train(self):
    for i in tqdm.tqdm(range(self.n_episodes),disable=self.disable):
      ep = self.episode()
      if self.callback:
        self.callback(i,ep)     