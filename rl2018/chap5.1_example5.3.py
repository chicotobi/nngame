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
  
pi = {s:(s[0]<20) for s in states}

V  = {s:0 for s in states}
NV = {s:0 for s in states}

Q = {(s,a):0 for s in states for a in actions}
NQ = {(s,a):0 for s in states for a in actions}

stepsize = 1e-3

n_episodes = int(1e7)
for i in range(n_episodes):
  if i%100000==0:
    print("Game",i)
  s = states[math.floor(npr.random()*len(states))]
  episode = []
  while s:
    a = pi[s]
    s_prime,r = step(s,a)
    episode.append((s,a,r))
    s = s_prime
  G = 0
  for (s,a,r) in episode[::-1]:
    G = gamma * G + r
    if stepsize > 0:
      V[s] += (G-V[s]) * stepsize
      Q[(s,a)] += (G-Q[(s,a)]) * stepsize
    else:
      NV[s] += 1
      V[s] += (G-V[s]) / NV[s]
      NQ[(s,a)] += 1
      Q[(s,a)] += (G-Q[(s,a)]) / NQ[(s,a)]
    pi[s] = Q[(s,0)]<Q[(s,1)]

# Dealer-Label
dealer_label = ["A"]+list(range(2,11))

# Plot
plt.figure()
arr = np.zeros((len(player),len(dealer),2))
for (ix,x) in enumerate(dealer):
  for (iy,y) in enumerate(player):
    for j in range(2):
      if x==11:
        arr[iy,0,j] = V[(y,x,j)]
      else:
        arr[iy,ix+1,j] = V[(y,x,j)]        
plt.subplot(1,2,1)
plt.imshow(arr[:,:,0],origin="lower",vmin=-1,vmax=1)
plt.xticks(range(len(dealer_label)),dealer_label)
plt.xlabel("Dealer")
plt.yticks(range(len(player)),player)
plt.ylabel("Player")
plt.title("No usable ace")
plt.subplot(1,2,2)
plt.imshow(arr[:,:,1],origin="lower",vmin=-1,vmax=1)
plt.xticks(range(len(dealer_label)),dealer_label)
plt.xlabel("Dealer")
plt.yticks(range(len(player)),player)
plt.ylabel("Player")
plt.title("Usable ace")

# Plot
plt.figure()
arr = np.zeros((len(player),len(dealer),2))
for (ix,x) in enumerate(dealer):
  for (iy,y) in enumerate(player):
    for j in range(2):
      if x==11:
        arr[iy,0,j] = pi[(y,x,j)]
      else:
        arr[iy,ix+1,j] = pi[(y,x,j)]  
plt.subplot(1,2,1)
plt.imshow(arr[:,:,0],origin="lower",vmin=-1,vmax=1)
plt.xticks(range(len(dealer_label)),dealer_label)
plt.xlabel("Dealer")
plt.yticks(range(len(player)),player)
plt.ylabel("Player")
plt.title("No usable ace")
plt.subplot(1,2,2)
plt.imshow(arr[:,:,1],origin="lower",vmin=-1,vmax=1)
plt.xticks(range(len(dealer_label)),dealer_label)
plt.xlabel("Dealer")
plt.yticks(range(len(player)),player)
plt.ylabel("Player")
plt.title("Usable ace")