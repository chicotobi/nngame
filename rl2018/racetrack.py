import numpy as np

import misc

from base_environment import BaseEnvironment

class Racetrack(BaseEnvironment):
  
  min_vx = 0
  max_vx = 5
  
  min_vy = 0
  max_vy = 5
  
  min_ax = -1
  max_ax = 1
  
  min_ay = -1
  max_ay = 1  
  
  def __init__(self):        
    p = "C:/Users/hofmant3/nngame/rl2018/chap5.7_exercise5.12_racetrack_1"  
    
    self.field = np.array([[int(i) for i in l] for l in open(p).read().splitlines()])
    self.field = np.flip(self.field,axis=0)
    self.field = np.swapaxes(self.field,0,1)
    
    # plt.imshow(np.swapaxes(field,0,1),origin="lower")
    
    self.sx, self.sy = self.field.shape
    
    states = []
    self.start_positions = []
    self.final_positions = []
    for x in range(self.sx):
      for y in range(self.sy):
        if self.field[x,y] != 0:
          states += [(x,y,vx,vy) for vx in range(self.min_vx, self.max_vx + 1) for vy in range(self.min_vy, self.max_vy + 1)]
        if self.field[x,y] == 2:
          self.start_positions += [(x,y)]
        if self.field[x,y] == 3:
          self.final_positions += [(x,y)]        
    
    actions = [(ax,ay) for ax in range(self.min_ax, self.max_ax + 1) for ay in range(self.min_ay, self.max_ay + 1)]
        
    valid_actions = {s:[] for s in states}
    for s in states:
      _,_,vx,vy = s
      for a in actions:
        ax,ay = a
        if self.min_vx <= vx+ax <= self.max_vx and self.min_vy <= vy+ay <= self.max_vy:
          valid_actions[s] += [a]
    
    super().__init__(states,actions,valid_actions=valid_actions)
    
  def get_random_initial_state(self):
      x,y = misc.sample(self.start_positions)
      vx = 0
      vy = 0
      return (x, y, vx, vy)
    
    
  def step(self, s, a):
    x,y,vx,vy = s
    ax, ay = a
    
    # Car is in goal
    if (x,y) in self.final_positions:
      return None, -1
    
    # Calculate next state
    vx += ax
    vy += ay
    x += vx
    y += vy
  
    # Car still on track ?
    on_track = 0 < x < self.sx and 0 < y < self.sy and self.field[x,y] != 0
    if not on_track:
      x, y, vx, vy = self.get_random_initial_state()
      
    return (x,y,vx,vy), -1