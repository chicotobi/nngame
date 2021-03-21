import matplotlib.pyplot as plt

from gridworld import CliffGridworld
from policy import EpsSoft
from agent import Agent
from experiment import Experiment

env = CliffGridworld()

p = EpsSoft(env.states,env.actions,eps=0.1)

agent = Agent(env.states,env.actions,p,alpha=0.5)

exp = Experiment(env,agent,"Qlearn")
exp.fixed_initial_state = env.start
exp.keep_episode = True

sumr = []
def plot(i,episode):
  global sumr, timesteps
  sumr += [sum([a[2] for a in episode])]
  if i%100==0:
    x = [a[0][0] for a in episode] + [env.goal[0]]
    y = [a[0][1] for a in episode] + [env.goal[1]]
    plt.plot(x,y,'-x')
    for j in range(env.sx+1):
      plt.plot([j-.5,j-.5],[-.5,env.sy-.5],'k-')
    for j in range(env.sy+1):
      plt.plot([-.5,env.sx-.5],[j-.5,j-.5],'k-')
    for step in episode:
      s, a, r = step
      dx, dy = (0,0)
      if "right" in a:
        dx += .5
      if "up" in a:
        dy += .5
      if "left" in a:
        dx -= .5
      if "down" in a:
        dy -= .5
      if dx != 0 or dy != 0:
        plt.arrow(s[0],s[1],dx,dy,color="r")
      else:
        plt.plot(s[0],s[1],"r")
    plt.xlim(-.5,env.sx-.5)
    plt.ylim(-.5,env.sy-.5)
    plt.show()
    
exp.train(500,plot)
    
plt.plot(sumr)
plt.ylim([-100, -20])