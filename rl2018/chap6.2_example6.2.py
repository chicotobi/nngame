import rl_functions
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

states = [0,1,2,3,4,5,6]
actions = [0]
rewards = [0,1]

def state_transition(s_prime,r,s,a):
  if abs(s-s_prime)==1:
    if r==1 and s_prime == 6:
      return 0.5
    if r==0:
      return 0.5
  return 0

def policy(a,s):
  return 1

v_true = rl_functions.evaluate_policy_linear_system(states,actions,rewards,state_transition,policy,terminal_states=[0,6])

gamma = 1

runs = 100
n_episodes = 100

legends = []
for alpha in [0.15,0.1,0.05,0.04,0.03,0.02,0.01]:
  e_mc = np.zeros((n_episodes,1))
  e_td = np.zeros((n_episodes,1))
  for run in range(runs):    
    v_init = np.ones((7,1))*0.5
    v_mc = v_init.copy()
    v_td = v_init.copy()  
    
    # plt.subplot(1,2,1)
    # plt.plot(v_init[1:6],'-xk')
    # plt.plot(v_true[1:6],'-xb')
    
    for i in range(n_episodes):
      s = 3
      episode = []
      while True:
        if npr.rand()<.5:
          s_prime = s + 1
        else:
          s_prime = s - 1
        if s_prime == 6:
          r = 1
        else:
          r = 0
        episode += [(s,r)]
        if s_prime in [0,6]:
          v_td[s] += alpha * (r - v_td[s] )
          break
        else:
          v_td[s] += alpha * (r + gamma * v_td[s_prime] - v_td[s])
        s = s_prime
      G = 0
      for (s,r) in episode[::-1]:
        G = gamma * G + r
        v_mc[s] += alpha * ( G - v_mc[s] )  
      e_mc[i] += (np.sum((v_mc[1:6]-v_true[1:6])**2))**.5/runs/5**.5
      e_td[i] += (np.sum((v_td[1:6]-v_true[1:6])**2))**.5/runs/5**.5
      # if i%10==9:
      #   plt.subplot(1,2,1)
      #   plt.plot(v_mc[1:6],'--g')
      #   plt.plot(v_td[1:6],'--r')
  
  plt.subplot(1,2,2)
  if alpha>=0.05:
    plt.plot(e_td)
    legends += ["TD "+str(alpha)]
  else:
    plt.plot(e_mc)
    legends += ["MC "+str(alpha)]
plt.legend(tuple(legends))
  
