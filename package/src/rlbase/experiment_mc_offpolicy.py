from experiment import BaseExperiment
import misc
import tqdm
from policy import EpsGreedy

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
    for i in tqdm.tqdm(range(self.n_episodes)): 
      self.b = EpsGreedy(env=self.env,eps=self.eps,det_policy=self.agent.pi)
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