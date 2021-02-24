import numpy as np
import matplotlib.pyplot as plt

from blackjack import Blackjack
from policy import DeterministicPolicy
from agent import Agent
from experiment import Experiment

env = Blackjack()
p = DeterministicPolicy(env.states,env.actions,{s:[s[0]<20] for s in env.states})
agent = Agent(env.states,env.actions, p,alpha=0.005)
experiment = Experiment(env, agent, "MC ES")

def callback(i):
  if i%100000==0:
    print("Game",i)
experiment.train(n_episodes = 1e7, callback=callback)

# Dealer-Label
dealer_label = ["A"]+list(range(2,11))

player, dealer = env.get_player_dealer()

# Plot
plt.figure()
arr = np.zeros((len(player),len(dealer),2))
for (ix,x) in enumerate(dealer):
  for (iy,y) in enumerate(player):
    for j in range(2):
      if x==11:
        arr[iy,0,j] = agent.V[(y,x,j)]
      else:
        arr[iy,ix+1,j] = agent.V[(y,x,j)]        
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
      s = (y,x,j)
      if x==11:
        arr[iy,0,j] = agent.p.get(s)
      else:
        arr[iy,ix+1,j] = agent.p.get(s)
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