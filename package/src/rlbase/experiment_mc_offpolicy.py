from experiment import BaseExperiment
import misc
import tqdm
from policy import EpsSoft

class MC_OffPolicyExperiment(BaseExperiment):
  
  def experiment_init(self, exp_init={}):
    self.n_visits = {s:{a:0 for a in self.env.valid_actions[s]} for s in self.env.states}
    Q_init = exp_init.get("Q_init",0)    
    self.eps = exp_init.get("eps",0)    
    self.callback = exp_init.get("callback")    
    self.Q = {s:{a:Q_init for a in self.env.valid_actions[s]} for s in self.env.states}
    self.C = {s:{a:0 for a in self.env.valid_actions[s]} for s in self.env.states}
    self.n_episodes = int(exp_init.get("n_episodes"))
    self.gamma = exp_init.get("gamma", 1)
    
  def episode(self):
    s = self.env.get_random_initial_state()
    a = self.b.get(s)
    ep = []
    while True: 
      r, s_prime, terminal = self.env.env_step(s,a)
      ep.append((s,a,r))
      if terminal:
        break
      a = self.b.get(s_prime)
      s = s_prime
    return ep  
    
  def train(self):
    for i in tqdm.tqdm(range(self.n_episodes)): 
      self.b = EpsSoft(self.env.states,self.env.valid_actions,self.eps,self.agent.pi)
      ep = self.episode()
      if self.callback:
        self.callback(i,ep)
      G = 0
      W = 1
      for idx, (s,a,r) in enumerate(ep[::-1]):
        G = self.gamma * G + r
        self.C[s][a] += W
        self.Q[s][a] += W/self.C[s][a] * (G - self.Q[s][a])
        a0 = misc.argmax_dct(self.Q[s])
        self.agent.pi.update(s,a0)
        if a0 != a:
          break
        W *= 1/self.b.prob(a,s)