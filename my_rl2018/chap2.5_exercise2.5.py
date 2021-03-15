import numpy as np
import numpy.random as npr
import math
import matplotlib.pyplot as plt

def create_bandit(n):
  v = npr.normal(size=n)
  def f(i):
    return v[i]+npr.normal()
  best_action = list(v).index(max(v))
  return f, best_action

k = 10

N_bandits = 1000
N_timesteps = 10000

eps = 0.1

results = []
optimal_actions = []
for stepsize_strategy in ["sample_average","constant"]:
  result = [0]*N_timesteps
  optimal_action = [0]*N_timesteps
  for bandit in range(N_bandits):
    
    # My bandit
    v = np.zeros(10)
    
    Q = [0]*k
    N = [0]*k
    for t in range(N_timesteps):
      
      #Change non-stationary bandit
      v += npr.normal(scale=0.01,size=k)
      best_action = list(v).index(max(v))
      
      # Choose action
      if npr.uniform() > eps:
        # Exploit
        action = Q.index(max(Q))
      else:
        # Explore
        action = math.floor(npr.uniform()*k)
      reward = v[action]
      
      # Benchmark
      result[t] += reward / N_bandits
      optimal_action[t] += (action==best_action) / N_bandits
      
      N[action] += 1
      if stepsize_strategy=="sample_average":
        stepsize = 1. / N[action]
      else:
        stepsize = 0.1        
      Q[action] += (reward - Q[action]) * stepsize
  results.append(result)
  optimal_actions.append(optimal_action)
plt.subplot(2,1,1)
plt.plot(results[0])
plt.plot(results[1])
plt.legend(["strat="+str(e) for e in ["sample_average","constant"]])
plt.subplot(2,1,2)
plt.plot(optimal_actions[0])
plt.plot(optimal_actions[1])
plt.legend(["strat="+str(e) for e in ["sample_average","constant"]])
  
    
