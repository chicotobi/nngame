class BaseAgent:  
  def __init__(self,**kwargs):
    self.pi = kwargs.get("pi")
    env = kwargs.get("env")
    q_init = kwargs.get("q_init",0)
    self.v = {s:0 for s in env.states}
    self.nv = {s:0 for s in env.states}
    self.q = {s:{a:q_init for a in env.valid_actions[s]} for s in env.states}
    self.nq = {s:{a:0 for a in env.valid_actions[s]} for s in env.states}
  
  def start(self, s):
    return self.pi.get(s)
  
  def step(self, r, s):
    return self.pi.get(s)
    
  def end(self, r):
    pass