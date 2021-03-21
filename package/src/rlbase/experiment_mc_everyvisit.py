from experiment import BaseExperiment

import tqdm

class MC_EveryVisitExperiment(BaseExperiment):
  
  def experiment_init(self, exp_init={}):
    self.n_visits = {s:0 for s in self.env.states}
    self.V = {s:0 for s in self.env.states}
    self.n_episodes = int(exp_init.get("n_episodes"))
    self.gamma = exp_init.get("gamma", 1)
    
  def episode(self):
    s = self.env.get_random_initial_state()
    a = self.agent.agent_start(s)
    ep = []
    while True: 
      r, s_prime, terminal = self.env.env_step(s,a)
      ep.append((s,a,r))
      if terminal:
        break
      a = self.agent.agent_step(r, s_prime)
      s = s_prime
    return ep  
    
  def train(self):
    for i in tqdm.tqdm(range(self.n_episodes)): 
      ep = self.episode()
      G = 0
      for (s,a,r) in ep[::-1]:
        G = self.gamma * G + r
        self.n_visits[s] += 1
        self.V[s] += 1. / self.n_visits[s] * (G - self.V[s])