import matplotlib.pyplot as plt

from gridworld import WindyGridworld
from policy import EpsSoft,DeterministicPolicy
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
def plot(i,episode):
  global lens, timesteps
  lens += [len(episode)]
  timesteps += [timesteps[-1]+len(episode)]
exp.train(170,plot)
    
plt.plot(timesteps,list(range(len(timesteps))),'rx-')

p = DeterministicPolicy(env.states,env.actions)
agent = Agent(env.states,env.actions,p,alpha=0.5)
exp = Experiment(env,agent,"Qlearn")
exp.fixed_initial_state = env.start
exp.keep_episode = True

lens = []
timesteps = [0]
def plot(i,episode):
  global lens
  lens += [len(episode)]
exp.train(170,plot)
    
plt.plot(timesteps,list(range(len(timesteps))),'bo-')
plt.show()