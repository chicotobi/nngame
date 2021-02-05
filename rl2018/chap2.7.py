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

def action_function_eps(Q,eps):
  k = len(Q)
  if npr.uniform() > eps:
    # Exploit
    action = Q.index(max(Q))
  else:
    # Explore
    action = math.floor(npr.uniform()*k)
  return action
    
def action_function_ucb(Q,N,t,c):
  ucb = []
  for i in range(len(Q)):
    q = Q[i]
    n = N[i]
    if n>0:
      val = q+c*(math.log(t+1)/n)**.5
    else:
      val = 1e9
    ucb.append(val)
  action = ucb.index(max(ucb))
  return action

results = []
optimal_actions = []
for action_function in ["eps","ucb"]:
  result = [0]*N_timesteps
  optimal_action = [0]*N_timesteps
  for bandit in range(N_bandits):
    my_bandit, best_action = create_bandit(k)
    Q = [0]*k      
    N = [0]*k
    for t in range(N_timesteps):
      # Choose action
      if action_function == "eps":
        action = action_function_eps(Q, 0.1)
      else:
        action = action_function_ucb(Q, N, t+1, 2)      
      # Get reward
      reward = my_bandit(action)
      
      # Benchmark
      result[t] += reward / N_bandits
      optimal_action[t] += (action==best_action) / N_bandits
      
      N[action] += 1
      Q[action] += (reward - Q[action]) / N[action]
  results.append(result)
  optimal_actions.append(optimal_action)
plt.subplot(2,1,1)
plt.plot(results[0])
plt.plot(results[1])
plt.legend(["eps","ucb"])
plt.subplot(2,1,2)
plt.plot(optimal_actions[0])
plt.plot(optimal_actions[1])
plt.legend(["eps","ucb"])
  
    
