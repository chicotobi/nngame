from environment import BaseEnvironment

import numpy as np
import misc

class RacetrackEnvironment(BaseEnvironment):
  
  min_vx = 0
  max_vx = 5
  
  min_vy = 0
  max_vy = 5
  
  min_ax = -1
  max_ax = 1
  
  min_ay = -1
  max_ay = 1  
  
  def __init__(self,**kwargs):        
    from pathlib import Path
    track_path = kwargs.get("track_path",str(Path.home())+"/nngame/package/src/rlbase/chap5.7_exercise5.12_racetrack_1")
    
    self.field = np.array([[int(i) for i in l] for l in open(track_path).read().splitlines()])
    self.field = np.flip(self.field,axis=0)
    self.field = np.swapaxes(self.field,0,1)
        
    self.sx, self.sy = self.field.shape
    
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