from experiment import BaseExperiment
import misc
import tqdm

class MC_ExploringStartsExperiment(BaseExperiment):
  
  def experiment_init(self, exp_init={}):
    self.n_visits = {s:{a:0 for a in self.env.actions} for s in self.env.states}
    self.Q = {s:{a:0 for a in self.env.actions} for s in self.env.states}
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
      set_sa = [(s,a) for (s,a,r) in ep[::-1]]
      for idx, (s,a,r) in enumerate(ep[::-1]):
        G = self.gamma * G + r
        if (s,a) not in set_sa[idx+1:]:
          self.n_visits[s][a] += 1
          self.Q[s][a] += 1. / self.n_visits[s][a] * (G - self.Q[s][a])
          a0 = misc.argmax_dct(self.Q[s])
          self.agent.pi.update(s,a0)
          #if self.n_visits[s][a]%1000==0:
            #print("s",s," n_visits[s][0]",self.n_visits[s][0]," n_visits[s][1]",self.n_visits[s][1]," Q[s][0]",self.Q[s][0]," Q[s][1]",self.Q[s][1])