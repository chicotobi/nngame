from environment import BaseEnvironment

import numpy as np
import tabulate as tb

class GridworldEnvironment(BaseEnvironment):

    def __init__(self):
        self.terminal_states = None
        pass

    def env_init(self, env_info):
        self.sx = env_info["sx"]
        self.sy = env_info["sy"]
        self.rewards = [-1,0]
        self.states = [(x,y) for x in range(self.sx) for y in range(self.sy)]    
        self.actions = ["up","left","down","right"]
        
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


class GridworldEx35Environment(GridworldEnvironment):
  state_A = (1,4)
  state_A_prime = (1,0)
  reward_A = 10
  state_B = (3,4)
  state_B_prime = (3,2)
  reward_B = 5
  
  def __init__(self):
    super().__init__()
    self.env_init({"sx":5,"sy":5})
    self.rewards += [5,10]
  
  def state_transition(self,s_prime, r, s, a):
    tmp1, tmp2 = self.env_step(s,a)
    return tmp1 == s_prime and tmp2 == r
    
  def env_step(self,s,a):
    x,y = s 
    if s == self.state_A:
      return self.state_A_prime, self.reward_A
    if s == self.state_B:
      return self.state_B_prime, self.reward_B
    if x == 0 and a=="left":
      return s, -1
    if x == self.sx - 1 and a=="right":
      return s, -1
    if y == 0 and a=="down":
      return s, -1
    if y == self.sy - 1 and a=="up":
      return s, -1
    if a=="right":
      return (x + 1, y), 0
    if a=="left":
      return (x - 1, y), 0
    if a=="down":
      return (x, y - 1), 0
    if a=="up":
      return (x, y + 1), 0
    
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
    super().__init__()
    self.terminal_states = (0,0)
    self.env_init({"sx":4,"sy":4})
    self.rewards = [-1]
  
  def state_transition(self,s_prime, r, s, a):
    tmp1, tmp2 = self.env_step(s,a)
    return tmp1 == s_prime and tmp2 == r
  
  def env_step(self, s, a):
    x,y = s
    if (x,y)==(0,3):
      return s, 0
    if (x,y)==(3,0):
      return s, 0
    if x == 0 and a=="left":
      return s, -1
    if x == self.sx - 1 and a=="right":
      return s, -1  
    if y == 0 and a=="down":
      return s, -1   
    if y == self.sy - 1 and a=="up":
      return s, -1
    if a=="right":
      return (x + 1, y), -1
    if a=="left":
      return (x - 1, y), -1
    if a=="down":
      return (x, y - 1), -1
    if a=="up":
      return (x, y + 1), -1
      
class WindyGridworldEnvironment(GridworldEnvironment):
  
  def __init__(self):
    super().__init__({"sx":10,"sy":7})
    self.rewards = [-1]
    self.start = (0,3)
    self.goal = (7,3)
    self.stochastic = False
          
  def env_step(self,s,a):
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
      
    if (x,y) == self.goal:
      return None, r
    else:
      return (x,y), r
    
  def state_transition(self,s_prime, r, s, a):
    tmp1, tmp2 = self.env_step(s,a)
    return tmp1 == s_prime and tmp2 == r
  
  
class CliffGridworldEnvironment(GridworldEnvironment):  
  
  def __init__(self):
    super().__init__({"sx":12,"sy":4})
    self.rewards = [-1]
    
  def env_init(self, env_info={}):    
    reward = None
    state = None
    termination = None
    self.reward_state_term = (reward, state, termination)
    
    self.start = (0,0)
    self.goal = (self.sy - 1, 0)
    
    self.cliff = [(self.grid_h - 1, i) for i in range(1, (self.grid_w - 1))]
    
  def env_start(self):
    self.reward_state_term = (0, self.start, False)
    return self.reward_state_term
    
  def env_step(self,s,a):
    x, y = s
    r = -1
    
    if "left" in a and x > 0:
      x -= 1
    if "right" in a and x < self.sx - 1:
      x += 1
    if "down" in a and y > 0:
      y -= 1
    if "up" in a and y < self.sy - 1:
      y += 1 
      
    if (x,y) == self.goal:
      self.reward_state_term = (r, None, True)
    elif 0 < x < self.sx-1 and y == 0:
      self.reward_state_term = (-100, self.start, False)
    else:
      self.reward_state_term = (r, (x,y), False)
    return r, (x,y), False
    
  def state_transition(self,s_prime, r, s, a):
    tmp1, tmp2 = self.step(s,a)
    return tmp1 == s_prime and tmp2 == r