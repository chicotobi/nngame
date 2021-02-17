class Gridworld:
    
  def __init__(self,sx,sy):
    self.sx = sx
    self.sy = sy
    self.states = [(x,y) for x in range(sx) for y in range(sy)]    
    self.actions = ["up","left","down","right"]
    self.rewards = [0]
    
class GridworldEx35(Gridworld):
  state_A = (1,4)
  state_A_prime = (1,0)
  reward_A = 10
  state_B = (3,4)
  state_B_prime = (3,2)
  reward_B = 5
  
  def __init__(self):
    super().__init__(5,5)
    self.rewards += []
  
  def state_transition(self,s_prime, r, s, a):
    x,y = s 
    xp, yp = s_prime
    if x == 0 and a=="down":
      return (r==-1 and s_prime==s) 
    if x == self.sx - 1 and a=="up":
      return (r==-1 and s_prime==s)      
    if y == 0 and a=="left":
      return (r==-1 and s_prime==s)    
    if y == self.sy - 1 and a=="right":
      return (r==-1 and s_prime==s)
    if s == self.state_A:
      return (s_prime == self.state_A_prime and r == self.reward_A)
    if s == self.state_B:
      return (s_prime == self.state_B_prime and r == self.reward_B)  
    if a=="right":
      return (r==0 and xp == x + 1 and yp == y)
    if a=="left":
      return (r==0 and xp == x - 1 and yp == y)
    if a=="down":
      return (r==0 and xp == x and yp == y - 1)
    if a=="up":
      return (r==0 and xp == x and yp == y + 1)