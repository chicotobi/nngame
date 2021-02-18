import numpy as np
import tabulate as tb

class Gridworld:
    
  def __init__(self,sx,sy):
    self.sx = sx
    self.sy = sy
    self.states = [(x,y) for x in range(sx) for y in range(sy)]    
    self.actions = ["up","left","down","right"]
    self.rewards = [-1,0]
    
class GridworldEx35(Gridworld):
  state_A = (1,4)
  state_A_prime = (1,0)
  reward_A = 10
  state_B = (3,4)
  state_B_prime = (3,2)
  reward_B = 5
  
  def __init__(self):
    super().__init__(5,5)
    self.rewards += [5,10]
  
  def state_transition(self,s_prime, r, s, a):
    x,y = s
    if s == self.state_A:
      return (s_prime == self.state_A_prime and r == self.reward_A)
    if s == self.state_B:
      return (s_prime == self.state_B_prime and r == self.reward_B)  
    if x == 0 and a=="left":
      return (r==-1 and s_prime==s) 
    if x == self.sx - 1 and a=="right":
      return (r==-1 and s_prime==s)      
    if y == 0 and a=="down":
      return (r==-1 and s_prime==s)    
    if y == self.sy - 1 and a=="up":
      return (r==-1 and s_prime==s)
    if a=="right":
      return (s_prime == (x+1,y) and r == 0)
    if a=="left":
      return (s_prime == (x-1,y) and r == 0)
    if a=="down":
      return (s_prime == (x,y-1) and r == 0)
    if a=="up":
      return (s_prime == (x,y+1) and r == 0)
    
  def step(self,s,a):
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
    