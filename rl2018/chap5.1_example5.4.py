import numpy as np
import matplotlib.pyplot as plt

from blackjack import Blackjack
from policy import DeterministicPolicy
from agent import Agent
from experiment import Experiment

env = Blackjack()
p = DeterministicPolicy(env.states,env.actions,{s:[s[0]<20] for s in env.states})
agent = Agent(env.states,env.actions, p,alpha=0.005)

s0 = (13,2,1)
# n_episodes = int(1e8)
# G = 0
# for i in range(n_episodes):
#   if i%100000==0:
#     print("Game",i)
#   s = s0
#   while s:
#     a = p.get(s)
#     s,r = env.step(s,a)
#   G += r
# G0 = G / n_episodes
# I get -27717616 / 1e8 = -0.27717616
G0 = -0.27726

# Now use behavorial policy
def b(s):
  return np.random.rand()<.5

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
      s,r = env.step(s,a)
    my_sum += rho*r
    my_sum2 += rho
    ordinary[trial,i] = my_sum / (i+1)
    if my_sum2> 0:
      weighted[trial,i] = my_sum / my_sum2
ordinary_mse = np.sum((ordinary-G0)**2,axis=0)/n_trials
weighted_mse = np.sum((weighted-G0)**2,axis=0)/n_trials
plt.semilogx(ordinary_mse,"g")
plt.semilogx(weighted_mse,"r")