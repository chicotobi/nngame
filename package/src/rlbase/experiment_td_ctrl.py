import experiment
import tqdm

class TD_CtrlExperiment(experiment.BaseExperiment):
      
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
      if self.callback:
        self.callback(i,ep)