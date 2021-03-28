import experiment
import tqdm

class MC_EveryVisitExperiment(experiment.BaseExperiment):

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
    for i in tqdm.tqdm(range(self.n_episodes)):
      ep = self.episode()
      if self.callback:
        self.callback(i,ep)
      G = 0
      for (s,a,r) in ep[::-1]:
        G = self.gamma * G + r
        self.n_visits[s] += 1
        self.V[s] += 1. / self.n_visits[s] * (G - self.V[s])