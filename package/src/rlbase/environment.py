import math
import numpy as np
import numpy.random as npr
import functools
import scipy.stats 
import tabulate as tb

import rlbase.misc as misc

class BaseEnvironment:
  states = None
  actions = None
  valid_actions = None

  def __init__(self):
    pass

  # Return state
  def start(self):
    pass

  # Get state, action, return (reward, state, terminal)
  def step(self):
    pass

  def get_initial_state(self):
    n_states = len(self.states)
    return self.states[npr.choice(n_states)]

  def set_all_actions_valid(self):
    self.valid_actions = {s: self.actions for s in self.states}


class BanditEnvironment(BaseEnvironment):

  def __init__(self,**kwargs):
    self.N = kwargs.get("N",10)

    self.states = [0]
    self.actions = list(range(self.N))
    self.set_all_actions_valid()

    self.arms = np.random.randn(self.N) + kwargs.get("offset",0)
    self.random = kwargs.get("random",False)

  def start(self):
    return self.states[0]

  def step(self, s, a):
    if self.random:
      self.arms += 0.01 * np.random.randn(self.N)
    r = self.arms[a] + np.random.randn()
    return r, self.states[0], False

class BlackjackEnvironment(BaseEnvironment):
  def __init__(self,**kwargs):
    self.player = range(12,22)
    self.dealer = list(range(2,12))
    usable = [0,1]
    self.states = [(a,b,c) for a in self.player for b in self.dealer for c in usable]
    self.actions = [0,1]
    self.type_random_initial_state = 1
    self.set_all_actions_valid()

  def get_player_dealer(self):
    return self.player, self.dealer

  def card(self):
    c = min(10,math.ceil(npr.random()*13))
    return c+(c==1)*10

  def step(self,s,a):
    player, dealer, usable = s

    # Player turn
    if a:
      c = self.card()
      player += c
      if player > 21:
        if c==11:
          if player-10>21:
            return -1, None, True
          else:
            return 0, (player-10,dealer,usable), False
        elif usable:
          return 0, (player-10,dealer,0), False
        else:
          return -1, None, True
      else:
        return 0, (player,dealer,usable), False

    # Dealer turn
    aces_dealer = dealer==11
    while dealer<17:
      c = self.card()
      dealer += c
      aces_dealer += c==11
      if dealer > 21:
        if aces_dealer>0:
          dealer -= 10
          aces_dealer -= 1
        else:
          return 1, None, True
    if dealer>player:
      return -1, None, True
    elif dealer==player:
      return 0, None, True
    else:
      return 1, None, True
    
  def plot_value_function(self,v):  
    player, dealer = self.get_player_dealer()
    arr = np.zeros((len(player),len(dealer),2))
    for (ix,x) in enumerate(dealer):
      for (iy,y) in enumerate(player):
        for j in range(2):
          if x==11:
            arr[iy,0,j] = v[(y,x,j)]
          else:
            arr[iy,ix+1,j] = v[(y,x,j)]   
    self.plot_helper(arr)
    
  def plot_bestaction_policy(self,pi):
    player, dealer = self.get_player_dealer()
    arr = np.zeros((len(player),len(dealer),2))
    for (ix,x) in enumerate(dealer):
      for (iy,y) in enumerate(player):
        for j in range(2):
          s = (y,x,j)
          if x==11:
            arr[iy,0,j] = pi.prob(0,s) > pi.prob(1,s)
          else:
            arr[iy,ix+1,j] = pi.prob(0,s) > pi.prob(1,s)
    self.plot_helper(arr)
            
  def plot_helper(self,arr):   
    import matplotlib.pyplot as plt
    player, dealer = self.get_player_dealer()
    vmin = np.min(arr)
    vmax = np.max(arr)
    plt.figure() 
    dealer_label = ["A"]+list(range(2,11))
    plt.subplot(1,2,1)
    plt.imshow(arr[:,:,0],origin="lower",vmin=vmin,vmax=vmax)
    plt.xticks(range(len(dealer_label)),dealer_label)
    plt.xlabel("Dealer")
    plt.yticks(range(len(player)),player)
    plt.ylabel("Player")
    plt.title("No usable ace")
    plt.subplot(1,2,2)
    plt.imshow(arr[:,:,1],origin="lower",vmin=vmin,vmax=vmax)
    plt.xticks(range(len(dealer_label)),dealer_label)
    plt.xlabel("Dealer")
    plt.yticks(range(len(player)),player)
    plt.ylabel("Player")
    plt.title("Usable ace")
    
# Memoization of the Poisson distribution for acceleration
@functools.lru_cache(1000000)
def mypois(n,l,maxn):
  if n < maxn:
      return scipy.stats.poisson.pmf(n ,l)
  if n == maxn:
      val = 1
      for i in range(maxn):
          val -= mypois(i,l,maxn)
      return val
  if n > maxn:
      return 0
    
class CarRentalEnvironment(BaseEnvironment):
  def __init__(self,**kwargs):
    self.nmax = kwargs.get("nmax",20)
    self.max_move = kwargs.get("max_move",5)
    self.max_return = kwargs.get("max_return",3)
    
    self.states = [(i,j) for i in range(self.nmax+1) for j in range(self.nmax+1)]
    self.terminal_states = []
    
    # Number of cars moved from first to second location
    self.actions = list(range(-self.max_move,self.max_move+1))
    self.set_all_actions_valid() # TODO This is wrong, but difficult
    
  def reshape(self,field):     
    v = np.zeros((self.nmax,self.nmax))
    for i in range(self.nmax):
      for j in range(self.nmax):
        v[i,self.nmax-j-1] = field[(i,j)]
    return v

  def state_transition_two_args(self, s, a):
  
    n_first = s[0]
    n_second = s[1]
  
    #If the action is impossible, return 0 probability
    if a > n_first or -a > n_second:
      return []
  
    #Reward starts with number of moved cars * -2$
    r0 = - 2 * abs(a)
    n_first -= a
    n_second += a
  
    lambda_request_first  = 3
    lambda_request_second = 4
    lambda_return_first   = 3
    lambda_return_second  = 2
  
    ans = []
    for n_request_first in range(n_first+1):
      for n_request_second in range(n_second+1):
        r = r0 + 10*(n_request_first+n_request_second)
        s_prime = (n_first-n_request_first,n_second-n_request_second)
        for n_return_first in range(self.max_return+1):
          for n_return_second in range(self.max_return+1):
            p = mypois(n_request_first ,lambda_request_first , n_first   ) * \
                mypois(n_request_second,lambda_request_second, n_second  ) * \
                mypois(n_return_first  ,lambda_return_first  , self.max_return) * \
                mypois(n_return_second ,lambda_return_second , self.max_return)
            s_prime = (\
              min(self.nmax,n_first -n_request_first +n_return_first),\
              min(self.nmax,n_second-n_request_second+n_return_second)\
             )
            if p>0:
              ans.append((s_prime, r, p))
    return ans
  
class GamblerEnvironment(BaseEnvironment):
    def __init__(self,**kwargs):
      self.n = kwargs.get("n",100)
      self.p_h = kwargs.get("p_h",0.4)      
      self.states = list(range(self.n+1))            
      self.terminal_states = [0,self.n]     
      self.actions = list(range(self.n//2+1))      
      self.set_all_actions_valid() # TODO This is wrong, but difficult
    
    def state_transition_two_args(self, s, a):
      
      if s in self.terminal_states:
        return [(s,0,1)]      
    
      #If the action is impossible, return 0 probability
      if a > s or a > self.n-s:
        return []
      
      if a+s >= self.n:
        r = 1
      else:
        r = 0    
      return [(s-a,0,1-self.p_h),(min(s+a,self.n), r, self.p_h)]

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
    self.terminal_states = [(self.sx - 1, 0)]
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
  
class RacetrackEnvironment(BaseEnvironment):
  
  min_vx = 0
  max_vx = 5
  
  min_vy = 0
  max_vy = 5
  
  min_ax = -1
  max_ax = 1
  
  min_ay = -1
  max_ay = 1  
  
  def get_track(self,id):
    if id==0:
      sx = 17
      sy = 32
      track = "\
00011111111111113\
00111111111111113\
00111111111111113\
01111111111111113\
11111111111111113\
11111111111111113\
11111111110000000\
11111111100000000\
11111111100000000\
11111111100000000\
11111111100000000\
11111111100000000\
11111111100000000\
11111111100000000\
01111111100000000\
01111111100000000\
01111111100000000\
01111111100000000\
01111111100000000\
01111111100000000\
01111111100000000\
01111111100000000\
00111111100000000\
00111111100000000\
00111111100000000\
00111111100000000\
00111111100000000\
00111111100000000\
00111111100000000\
00011111100000000\
00011111100000000\
00022222200000000"
    if id==1:
      sx = 31
      sy = 13
      track = "\
0000000001111111111111000000000\
0000000111111111111111111111113\
0000011111111111111111111111113\
0001111111111111111111111111113\
0001111111111100000000000000000\
0001111111111000000000000000000\
0001111111110000000000000000000\
0001111111110000000000000000000\
0001111111100000000000000000000\
0001111111100000000000000000000\
0001111111000000000000000000000\
0001111111000000000000000000000\
0002222222000000000000000000000"
    return sx,sy,track
  
  def __init__(self,**kwargs):     
    
    self.sx, self.sy, self.field = self.get_track(kwargs.get("track_id",0)) 
    self.field = np.reshape(np.array([int(i) for i in self.field]),(self.sy,self.sx))
    self.field = np.flip(self.field,axis=0)
    self.field = np.swapaxes(self.field,0,1)

    self.states = []
    self.start_positions = []
    self.final_positions = []
    for x in range(self.sx):
      for y in range(self.sy):
        if self.field[x,y] != 0:
          self.states += [(x,y,vx,vy) for vx in range(self.min_vx, self.max_vx + 1) for vy in range(self.min_vy, self.max_vy + 1)]
        if self.field[x,y] == 2:
          self.start_positions += [(x,y)]
        if self.field[x,y] == 3:
          self.final_positions += [(x,y)]        
    
    self.actions = [(ax,ay) for ax in range(self.min_ax, self.max_ax + 1) for ay in range(self.min_ay, self.max_ay + 1)]
        
    self.valid_actions = {s:[] for s in self.states}
    for s in self.states:
      _,_,vx,vy = s
      for a in self.actions:
        ax,ay = a
        if self.min_vx <= vx+ax <= self.max_vx and self.min_vy <= vy+ay <= self.max_vy:
          self.valid_actions[s] += [a]
    
  def get_initial_state(self):
    x,y = misc.sample(self.start_positions)
    vx = 0
    vy = 0
    return (x, y, vx, vy)
    
  def step(self, s, a):
    x,y,vx,vy = s
    ax, ay = a
    
    # Car is in goal
    if (x,y) in self.final_positions:
      return -1, None, True
    
    # Calculate next state
    vx += ax
    vy += ay
    x += vx
    y += vy
  
    # Car still on track ?
    on_track = 0 < x < self.sx and 0 < y < self.sy and self.field[x,y] != 0
    if not on_track:
      x, y, vx, vy = self.get_initial_state()
      
    return -1, (x,y,vx,vy), False
  
class RandomWalkEnvironment(BaseEnvironment):
    def __init__(self,**kwargs):
      self.n = kwargs.get("nmax",5  )
      self.p = kwargs.get("p",.5)
      
      self.states = list(range(self.n+2))
      self.terminal_states = [0,self.n+1]
      self.actions = [0]
      self.rewards = [0,1]
      self.set_all_actions_valid()

    def step(self,s,a):
      if np.random.rand() > self.p:
        s += 1
      else:
        s -= 1
      if s == self.n + 1:
        return 1, None, True
      elif s == 0:
        return 0, None, True
      else:
        return 0, s, False
      
    def state_transition(self,s_prime,r,s,a):
      if s_prime == s + 1:
        if s_prime == self.n + 1 and r == 1:
          return self.p
        if r == 0:
          return self.p
      if s_prime == s - 1:
        if r == 0:
          return 1 - self.p
      return 0
    
class Ex67Environment(BaseEnvironment):
  def __init__(self,**kwargs):
    self.n = kwargs.get("n",10)
    self.states = ["A","B"]
    self.valid_actions = {}
    self.valid_actions["A"] = ["l","r"]
    self.valid_actions["B"] = list("l"+str(i) for i in range(self.n))
    self.actions = self.valid_actions["A"] + self.valid_actions["B"]

  def get_initial_state(self):
    return "A"
  
  def step(self,s,a):
    if s == "A":
      if a == "l":
        return 0, "B", False
      else:
        return 0, None, True
    else:
      return np.random.randn()-0.1, None, True
    
class MazeEnvironment(GridworldEnvironment):
  def __init__(self,**kwargs):
    super().__init__(sx=9,sy=6)
    self.rewards = [0, 1]
    self.start = kwargs.get("start",(0,3))
    self.terminal_states = [kwargs.get("goal",(8,5))]
    self.wall = [(2,2),(2,3),(2,4),(5,1),(7,3),(7,4),(7,5)]
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
      
    if (x,y) in self.wall:
      return 0, s, False
    if (x,y) in self.terminal_states:
      return 1, None, True
    return 0, (x,y), False
  
class ChangingMazeEnvironment(GridworldEnvironment):
  def __init__(self,**kwargs):
    super().__init__(sx=9,sy=6)
    self.rewards = [0, 1]
    self.start = kwargs.get("start",(3,0))
    self.terminal_states = [kwargs.get("goal",(8,5))]
    self.wall_always = [(1,2),(2,2),(3,2),(4,2),(5,2),(6,2),(7,2)]
    self.set_all_actions_valid()
    self.t = 0
              
  def step(self,s,a):
    self.t += 1
    
    x, y = s
    
    if "left" in a and x > 0:
      x -= 1
    if "right" in a and x < self.sx - 1:
      x += 1
    if "down" in a and y > 0:
      y -= 1
    if "up" in a and y < self.sy - 1:
      y += 1 
      
    if (x,y) in self.wall:
      return 0, s, False
    if (x,y) in self.terminal_states:
      return 1, None, True
    return 0, (x,y), False

class BlockingMazeEnvironment(ChangingMazeEnvironment):
  def __init__(self,**kwargs):
    super().__init__(**kwargs)
    self.wall = self.wall_always + [(0,2)]
    
  def change(self):
    self.wall = self.wall_always + [(8,2)]
    
class ShortcutMazeEnvironment(ChangingMazeEnvironment):
  def __init__(self,**kwargs):
    super().__init__(**kwargs)
    self.wall = self.wall_always + [(8,2)]
    
  def change(self):
    self.wall = self.wall_always
    
class RodManeuverEnvironment(BaseEnvironment):   
  
  polys_coords = [[(3.03,16.32),(3.7,14.06),(5.37,15.75),(4.72,17.99)],
               [(1.63,13.28),(4.19,11.59),(7.17,10.9),(4.63,12.59)],
               [(4.1,10.19),(3.88,6),(7.26,3.49),(7.46,7.68)],
               [(9.03,15.84),(9.83,13.79),(11.97,13.44),(11.19,15.46)],
               [(16.23,16.5),(15.23,18.61),(17.37,17.63),(18.39,15.52)],
               [(13.68,14.46),(12.99,10.42),(14.12,6.51),(14.81,10.5)],
               [(12.9,5.44),(13.08,3.6),(14.5,2.4),(14.3,4.24)],
               [(14.03,1.87),(15.92,3.09),(18.1,2.55),(16.21,1.31)]]
  polys = None    
  
  solution = [(3,6,275),(3,6,285),(3,6,295),(3,6,305),(3,6,315),
                  (3,6,325),(4,5,325),(5,4,325),(6,3,325),(7,2,325),
                  (7,2,335),(8,2,335),(9,2,335),(10,2,335),(11,2,335),
                  (11,2,325),(10,3,325),(10,3,315),(10,3,305),(10,3,295),
                  (10,4,295),(10,5,295),(10,5,305),(9,6,305),(8,7,305),
                  (8,7,295),(8,7,285),(8,7,275),(8,7,275),(8,7,275),
                  (8,8,275),(8,9,275),(8,10,275),(8,11,275),(8,12,275),
                  (8,13,275),(8,13,275),(8,14,275),(8,15,275),(8,15,265),
                  (8,15,255),(8,15,245),(8,15,235),(9,16,235),(9,16,225),
                  (9,16,215),(10,17,215),(10,17,205),(11,17,205),(11,17,195),
                  (12,17,195),(12,17,185),(12,17,175),(12,17,165),(12,17,155),
                  (12,17,145),(13,16,145),(14,15,145),(15,14,145),(16,13,145),
                  (16,13,135),(16,13,125),(16,13,115),(16,13,105),(16,13,95),
                  (16,13,85)]
    
  def state_to_line(self,s):
    x, y, theta = s
    theta = theta / 180 * math.pi
    l = 3
    return (x+l*math.cos(theta),
            y+l*math.sin(theta),
            x-l*math.cos(theta),
            y-l*math.sin(theta)) 
  
  def is_valid_state(self,s):
    import shapely.geometry
    is_valid = True
    x1, y1, x2, y2 = self.state_to_line(s)
    if 0 <= x1 <= 19 and 0 <= x2 <= 19 and 0 <= y1 <= 19 and 0 <= y2 <= 19:
      line = shapely.geometry.LineString([(x1,y1),(x2,y2)])
      for poly in self.polys:
        if poly.crosses(line):
          is_valid = False
      return is_valid
    else:
      return False
  
  def plot_states(self,states=[]):
    import matplotlib.pyplot as plt
    for i in range(20):
      plt.plot([0,19],[i,i],"#aaaaaa",zorder=0,linewidth=1)
      plt.plot([i,i],[0,19],"#aaaaaa",zorder=0)
    for s in states:
      x1, y1, x2, y2 = self.state_to_line(s)
      if self.is_valid_state(s):
        c = "b"
      else:
        c = "r"
      plt.plot([x1,x2],[y1,y2],c)
      plt.plot(x1,y1,c+"o")
      plt.plot(s[0],s[1],c+"x")
    for p in self.polys:
      x = [x for (x,y) in p.exterior.coords]
      y = [y for (x,y) in p.exterior.coords]
      plt.fill(x,y,"k")
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    plt.draw()
  
  def step(self,s,a):
    x, y, theta = s
    
    if a == "l":
      theta  = (theta+10)%360
    elif a == "r":
      theta  = (theta-10)%360
    elif a in ["f","b"]:
      if a == "b":
        t0 = (theta + 180) % 360
      else:
        t0 = theta
      
      if 0 < t0 < 60 or 300 < t0 < 360:
        x += 1
      elif 120 < t0 < 240:
        x -= 1
      if 30 < t0 < 150:
        y += 1
      elif 210 < t0 < 330:
        y -= 1
    else:
      raise ValueError("Invalid action")
      
    if self.is_valid_state((x,y,theta)):
      s_prime = (x,y,theta)  
    else:
      s_prime = s
    return s_prime, -1
    
  def __init__(self,**kwargs):
    import shapely.geometry        
    self.polys = []
    for p in self.polys_coords:
      self.polys.append(shapely.geometry.Polygon(p))
          
    self.states = []
    for x in range(20):
      for y in range(20):
        for t in range(36):
          self.states.append((x,y,5+10*t))
    
    self.actions = ["l","r","f","b"]
    
    self.set_all_actions_valid()
    
    self.start = (3,6,275)
    self.goal = (16,13,85)

  def start(self):
    return self.start
  