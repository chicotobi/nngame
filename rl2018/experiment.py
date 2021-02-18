class Experiment:  
  def __init__(self, environment, agent,):
    self.env = environment
    self.agent = agent
    
  def train(self, n_episodes, stepsize, callback, gamma=1, update_policy=False):
    n_episodes = int(n_episodes)
    for i in range(n_episodes):
      callback(i)
      s = self.env.get_random_state()
      episode = []
      while s:
        a = self.agent.p.get(s)
        s_prime,r = self.env.step(s,a)
        episode.append((s,a,r))
        s = s_prime
      G = 0
      for (s,a,r) in episode[::-1]:
        G = gamma * G + r
        self.agent.update_state_value(G,s,stepsize)
        self.agent.update_action_value(G,s,a,stepsize)
        if update_policy:
          self.agent.update_policy(s)
    