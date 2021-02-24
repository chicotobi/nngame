import matplotlib.pyplot as plt

from gridworld import WindyGridworld
from policy import EpsSoft
from agent import Agent
from experiment import Experiment

alpha = 0.5
eps = 0.1

env = WindyGridworld()

p = EpsSoft(env.states,env.actions,eps)

agent = Agent(env.states,env.actions,p)

exp = Experiment(env,agent)

max_ep = 170
n_episodes = int(max_ep)

lens = []
timesteps = [0]

def plot(i,episode):
  global lens, timesteps
  lens += [len(episode)]
  timesteps += [timesteps[-1]+len(episode)]
  if i%500==0:
    x = [a[0] for a in episode]
    y = [a[1] for a in episode]
    plt.plot(x,y,'-x')
    plt.xlim(-1,sx)
    plt.ylim(-1,sy)
    plt.show()
    
exp.train_sarsa(170,alpha)
    
plt.plot(timesteps,list(range(len(timesteps))))