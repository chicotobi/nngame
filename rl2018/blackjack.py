from base_environment import BaseEnvironment
import math
import numpy.random as npr

class Blackjack(BaseEnvironment):  
  def __init__(self):        
    self.player = range(12,22)
    self.dealer = list(range(2,12))
    usable = [0,1]
    
    states = [(a,b,c) for a in self.player for b in self.dealer for c in usable]
    
    actions = [0,1]
    
    super().__init__(states,actions)    
  
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
            return None,-1
          else:
            return (player-10,dealer,usable),0
        elif usable:
          return (player-10,dealer,0),0
        else:
          return None,-1
      else:
        return (player,dealer,usable),0
  
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
          return None,1
    if dealer>player:
      return None,-1
    elif dealer==player:
      return None,0
    else:
      return None,1