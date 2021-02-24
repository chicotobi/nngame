import numpy as np
import matplotlib.pyplot as plt

from racetrack import Racetrack
from policy import DeterministicPolicy
from agent import Agent
from experiment import Experiment

env = Racetrack()

p = DeterministicPolicy(env.states,env.valid_actions,best_actions={s:[(0,0)] for s in env.states})

agent = Agent(env.states,env.valid_actions,p,Qinit=-1e6)

exp = Experiment(env,agent,"Off-policy")
exp.eps = 0.01

lens = []

def plot(i, episode):  
  global lens
  lens += [len(episode)]
  if i%50==0:
    print("Episode ",i, " Length",len(episode))
    x = [x for ((x,_,_,_),_,_) in episode]
    y = [y for ((_,y,_,_),_,_) in episode]
  
    plt.subplot(2,2,1)
    plt.imshow(np.swapaxes(env.field,0,1),origin="lower")
    plt.plot(x,y,'rx-')
    
    plt.subplot(2,2,2)
    plt.imshow(np.swapaxes(env.field,0,1),origin="lower")
    arr_u = np.zeros((env.sx,env.sy))
    arr_v = np.zeros((env.sx,env.sy))
    for s in env.states:
      ax,ay = p.get(s)
      x,y,_,_ = s
      arr_u[x,y] += ax
      arr_v[x,y] += ay
    for x in range(env.sx):
      for y in range(env.sy):
        v = (arr_u[x,y]**2 + arr_v[x,y]**2)**.5
        if v>0:
          arr_u[x,y] /= v
          arr_v[x,y] /= v
    plt.quiver(np.swapaxes(arr_u,0,1),np.swapaxes(arr_v,0,1),units='xy',angles='xy',scale=1)
        
    plt.subplot(2,2,(3,4))
    plt.semilogy(lens,'.')
    
    plt.show()
    
exp.train(1e3,plot)
