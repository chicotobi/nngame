import matplotlib.pyplot as plt

from gridworld import WindyGridworld
from policy import EpsSoft
from agent import Agent
from experiment import Experiment

env = WindyGridworld()

p = EpsSoft(env.states,env.actions,eps=0.1)

agent = Agent(env.states,env.actions,p,alpha=0.5)

exp = Experiment(env,agent,"SARSA")
exp.fixed_initial_state = env.start
exp.keep_episode = True

lens = []
timesteps = [0]
eee = []

def plot(i,episode):
  global lens, timesteps
  lens += [len(episode)]
  timesteps += [timesteps[-1]+len(episode)]
  if i%100==0:
    x = [a[0][0] for a in episode] + [env.goal[0]]
    y = [a[0][1] for a in episode] + [env.goal[1]]
    plt.plot(x,y,'-x')
    for j in range(env.sx+1):
      plt.plot([j-.5,j-.5],[-.5,env.sy-.5],'k-')
    for j in range(env.sy+1):
      plt.plot([-.5,env.sx-.5],[j-.5,j-.5],'k-')
    for a in episode:
      if a[1] == "right":
        dx, dy = (.5,0)
      elif a[1] == "up":
        dx, dy = (0,.5)
      elif a[1] == "left":
        dx, dy = (-.5,0)
      elif a[1] == "down":
        dx, dy = (0,-.5)
      plt.arrow(a[0][0],a[0][1],dx,dy,color="r")
    plt.xlim(-.5,env.sx-.5)
    plt.ylim(-.5,env.sy-.5)
    plt.show()
    
exp.train(2000,plot)
    
plt.plot(timesteps,list(range(len(timesteps))))