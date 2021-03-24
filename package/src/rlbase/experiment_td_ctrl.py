from experiment import BaseExperiment

from tqdm import tqdm

class TD_CtrlExperiment(BaseExperiment):
  
  def experiment_init(self, exp_init={}):
    self.callback = exp_init.get("callback")    
    self.n_episodes = int(exp_init.get("n_episodes"))
    
  def episode(self):
    s = self.env.get_initial_state()
    a = self.agent.agent_start(s)
    ep = []
    while True: 
      r, s_prime, terminal = self.env.env_step(s,a)
      ep.append((s,a,r))
      if terminal:
        break
      a = self.agent.agent_step(r,s_prime)
      s = s_prime
    return ep
    
  def train(self):
    for i in tqdm(range(self.n_episodes)): 
      ep = self.episode()
      if self.callback:
        self.callback(i,ep)