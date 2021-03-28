from environment import BaseEnvironment
import math
import numpy.random as npr

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