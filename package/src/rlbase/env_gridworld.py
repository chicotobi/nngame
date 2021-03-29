from environment import BaseEnvironment

import numpy as np
import tabulate as tb

class GridworldEnvironment(BaseEnvironment):

  def __init__(self, **kwargs):
    super().__init__()
    self.sx = kwargs.get("sx")
    self.sy = kwargs.get("sy")
    self.rewards = [-1,0]
    self.states = [(x,y) for x in range(self.sx) for y in range(self.sy)]    
    self.actions = ["up","left","down","right"]
    self.set_all_actions_valid()
      
  def add_diagonal(self):
    self.actions += ["upleft","upright","downleft","downright"]
    
  def add_stay(self):
    self.actions += ["stay"]
  
  def pretty_print(self,v):
    print(tb.tabulate(np.round(v,1)))
    print()
  
  def reshape(self,field):        
    v = np.zeros((self.sx,self.sy))
    for i in range(self.sx):
      for j in range(self.sy):
        v[i,self.sy-j-1] = field[(i,j)]
    return v
  
  def get_initial_state(self):
    return self.start
  
  def state_transition(self,s_prime, r, s, a):
    tmp1, tmp2, tmp3 = self.step(s,a)
    return tmp1 == r and tmp2 == s_prime

class GridworldEx35Environment(GridworldEnvironment):

  def __init__(self):
    super().__init__(sx=5,sy=5)
    self.rewards += [5,10]
    
  def step(self,s,a):
    state_A = (1,4)
    state_A_prime = (1,0)
    reward_A = 10
    state_B = (3,4)
    state_B_prime = (3,2)
    reward_B = 5    
    
    x, y = s 
    r = 0
    if s == state_A:
      return reward_A, state_A_prime, False
    if s == state_B:
      return reward_B, state_B_prime, False
    if "left" in a:
      if x == 0:
        r = -1
      else:
        x -= 1
    if "right" in a:
      if x == self.sx - 1:
        r = -1
      else:
        x += 1
    if "down" in a:
      if y == 0:
        r = -1
      else:
        y -= 1
    if "up" in a:
      if y == self.sy - 1:
        r = -1
      else:
        y += 1 
    return r, (x,y), False
    
  def plot(self,f):
    v = np.zeros((self.sx,self.sy))
    for i in range(self.sx):
      for j in range(self.sy):
        v[i,j] = f[(i,j)]
    print(tb.tabulate(np.flipud(np.round(v,1).transpose())))
    
  def plot_bestaction_policy(self,p):        
    tmp = np.ndarray((self.sx,self.sy), dtype = 'object')
    for s in self.states:  
      tmp[s[0],s[1]] = ""
      for a in self.actions:
        if p.prob(a,s):
          tmp[s[0],s[1]] += a[0]
    print(tb.tabulate(np.flipud(tmp.transpose())))
    
class GridworldEx41Environment(GridworldEnvironment):
  
  def __init__(self):
    super().__init__(sx=4,sy=4)
    self.rewards = [-1]
    self.terminal_states = (0,0)
  
  def step(self, s, a):
    x, y = s
    r = -1
    if (x,y)==(0,3):
      return 0, None, True
    if (x,y)==(3,0):
      return 0, None, True
    if "left" in a and x > 0:
      x -= 1
    if "right" in a and x < self.sx - 1:
      x += 1
    if "down" in a and y > 0:
      y -= 1
    if "up" in a and y < self.sy - 1:
      y += 1 
    return r, (x,y), False
  
  def plot_bestaction_policy(self,p):        
    tmp = np.ndarray((self.sx,self.sy), dtype = 'object')
    for s in self.states:  
      tmp[s[0],s[1]] = ""
      for a in self.actions:
        if p.prob(a,s):
          tmp[s[0],s[1]] += a[0]
    print(tb.tabulate(np.flipud(tmp.transpose())))
      
class WindyGridworldEnvironment(GridworldEnvironment):
    
  def __init__(self,**kwargs):
    super().__init__(sx=10,sy=7)
    self.rewards = [-1]
    self.stochastic = kwargs.get("stochastic",False)
    self.start = kwargs.get("start",(0,3))
    self.terminal_states = [kwargs.get("goal",(7,3))]
    self.set_all_actions_valid()
              
  def step(self,s,a):
    x, y = s
    r = -1
    
    if x in [3,4,5,8]:
      y = min(y+1,self.sy-1)
    if x in [6,7]:
      y = min(y+2,self.sy-1)
      
    if self.stochastic and x in [3,4,5,6,7,8]:
      v = np.random.rand()
      if v < 1/3:
        y = min(y+1,self.sy-1)
      elif v< 2/3:
        y = max(y-1,0)
      
    if "left" in a and x > 0:
      x -= 1
    if "right" in a and x < self.sx - 1:
      x += 1
    if "down" in a and y > 0:
      y -= 1
    if "up" in a and y < self.sy - 1:
      y += 1 
      
    if (x,y) in self.terminal_states:
      return r, None, True
    else:
      return r, (x,y), False
  
class CliffGridworldEnvironment(GridworldEnvironment):  
  
  def __init__(self):
    super().__init__(sx=12,sy=4)
    self.rewards = [-1]
    self.start = (0,0)
    self.terminal_states = [(self.sy - 1, 0)]
    self.cliff = [(self.sx - 1, i) for i in range(1, (self.sy - 1))]
    self.set_all_actions_valid()
      
  def step(self,s,a):
    x, y = s
    
    if "left" in a and x > 0:
      x -= 1
    if "right" in a and x < self.sx - 1:
      x += 1
    if "down" in a and y > 0:
      y -= 1
    if "up" in a and y < self.sy - 1:
      y += 1 
      
    if (x,y) in self.terminal_states:
      return -1, None, True
    elif 0 < x < self.sx-1 and y == 0:
      return -100, self.start, False
    return -1, (x,y), False