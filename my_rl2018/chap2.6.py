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

N_bandits = 2000
N_timesteps = 1000

arr_eps = [0,0.1]

results = []
optimal_actions = []
for eps in arr_eps:
  result = [0]*N_timesteps
  optimal_action = [0]*N_timesteps
  for bandit in range(N_bandits):
    my_bandit, best_action = create_bandit(k)
    if eps==0:
      Q = [5]*k
    else:
      Q = [0]*k      
    N = [0]*k
    for t in range(N_timesteps):
      # Choose action
      if npr.uniform() > eps:
        # Exploit
        action = Q.index(max(Q))
      else:
        # Explore
        action = math.floor(npr.uniform()*10)
      reward = my_bandit(action)
      
      # Benchmark
      result[t] += reward / N_bandits
      optimal_action[t] += (action==best_action) / N_bandits
      
      N[action] += 1
      Q[action] += (reward - Q[action]) * 0.1
  results.append(result)
  optimal_actions.append(optimal_action)
plt.subplot(2,1,1)
plt.plot(results[0])
plt.plot(results[1])
plt.legend(["Optimistic Q=5 eps=0","Realistic Q=0 eps=0.1"])
plt.subplot(2,1,2)
plt.plot(optimal_actions[0])
plt.plot(optimal_actions[1])
plt.legend(["Optimistic Q=5 eps=0","Realistic Q=0 eps=0.1"])
  
    
