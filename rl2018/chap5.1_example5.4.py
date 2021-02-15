import math
import numpy.random as npr
import numpy as np
import matplotlib.pyplot as plt

player = range(11,22)
dealer = list(range(2,12))
usable = [0,1]

states = [(a,b,c) for a in player for b in dealer for c in usable]

actions = [0,1]

gamma = 1

npr.seed(0)

def card():
  c = min(10,math.ceil(npr.random()*13))
  return c+(c==1)*10

def step(s,a):
  player, dealer, usable = s
  
  # Player turn
  if a:
    c = card()
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
    c = card()
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
  
# First, evaluate a state value
s0 = (13,2,1)
pi = {s:(s[0]<20) for s in states}
# n_episodes = int(1e8)
# G = 0
# for i in range(n_episodes):
#   if i%100000==0:
#     print("Game",i)
#   s = s0
#   while s:
#     a = pi[s]
#     s,r = step(s,a)
#   G += r
# G0 = G / n_episodes
# I get -27717616 / 1e8 = -0.27717616
G0 = -0.27726

# Now use behavorial policy
def b(s):
  return npr.rand()<.5

n_trials = 100
n_episodes = int(1e4)
ordinary = np.zeros((n_trials,n_episodes))
weighted = np.zeros((n_trials,n_episodes))
for trial in range(n_trials):
  my_sum = 0
  my_sum2 = 0
  for i in range(n_episodes):
    s = s0
    episode = []
    rho = 1
    while s:
      a = b(s)
      if a:
        rho *= (s[0]<20) / (1/2)
      else: 
        rho *= (s[0]>19) / (1/2)
      s,r = step(s,a)
    my_sum += rho*r
    my_sum2 += rho
    ordinary[trial,i] = my_sum / (i+1)
    if my_sum2> 0:
      weighted[trial,i] = my_sum / my_sum2
ordinary_mse = np.sum((ordinary-G0)**2,axis=0)/n_trials
weighted_mse = np.sum((weighted-G0)**2,axis=0)/n_trials
plt.semilogx(ordinary_mse,"g")
plt.semilogx(weighted_mse,"r")