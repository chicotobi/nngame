from experiment import BaseExperiment
import misc, tqdm

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
    for i in tqdm.tqdm(range(self.n_episodes)): 
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