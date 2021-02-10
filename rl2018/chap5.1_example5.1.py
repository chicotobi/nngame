import math
import numpy.random as npr
import numpy as np
import matplotlib.pyplot as plt

player = range(12,22)
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
  
pi = {s:(s[0]<20) for s in states}

V  = {s:0 for s in states}
NV = {s:0 for s in states}

n_episodes = 500000
for i in range(n_episodes):
  if i%10000==0:
    print("Game",i)
  s = states[math.floor(npr.random()*200)]
  episode = []
  while s:
    a = pi[s]
    s_prime,r = step(s,a)
    episode.append((s,a,r))
    s = s_prime
  G = 0
  for (s,a,r) in episode[::-1]:
    G = gamma * G + r
    NV[s] += 1
    V[s] += (G-V[s])/NV[s]

# Plot
arr = np.zeros((10,10,2))
for (ix,x) in enumerate(dealer):
  for (iy,y) in enumerate(player):
    for j in range(2):
      arr[iy,ix,j] = V[(y,x,j)]
plt.subplot(1,2,1)
plt.imshow(arr[:,:,0],origin="lower",vmin=-1,vmax=1)
plt.xticks(range(10),dealer)
plt.xlabel("Dealer")
plt.yticks(range(10),player)
plt.ylabel("Player")
plt.title("No usable ace")
plt.subplot(1,2,2)
plt.imshow(arr[:,:,1],origin="lower",vmin=-1,vmax=1)
plt.xticks(range(10),dealer)
plt.xlabel("Dealer")
plt.yticks(range(10),player)
plt.ylabel("Player")
plt.title("Usable ace")