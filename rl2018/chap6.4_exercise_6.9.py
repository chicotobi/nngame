import rl_functions
import numpy as np
import matplotlib.pyplot as plt

states = [(i,j) for i in range(10) for j in range(7)]

actions = ["up","left","down","right","upright","upleft","downright","downleft"]

start = (0,3)
goal = (7,3)

gamma = 1
alpha = 0.5
eps = 0.1

valid_actions = {s:actions for s in states}

sx = 10
sy = 7

gx, gy = goal

def step(s,a):
  x, y = s
  r = -1
  if "left" in a and x > 0:
    x -= 1
  if "right" in a and x < sx - 1:
    x += 1
  if "down" in a and y > 0:
    y -= 1
  if "up" in a and y < sy - 1:
    y += 1
  
  if x in [3,4,5,8]:
    y = min(y+1,sy-1)
  if x in [6,7]:
    y = min(y+2,sy-1)
    
  if (x,y)==goal:
    return None, r
  else:
    return (x,y), r
  
Q = {s:{a:0 for a in valid_actions[s]} for s in states}

max_ep = 10000
n_episodes = int(max_ep)

lens = []
timesteps = [0]

for i in range(n_episodes):
  
  s = start
  
  pi, _ = rl_functions.generate_eps_greedy_policy(Q,eps,valid_actions)
  a = pi(s)
  
  # Generate an episode using b
  episode = [s]
  while s:
    s_prime,r = step(s,a)
    if not s_prime:
      break
    pi, _ = rl_functions.generate_eps_greedy_policy(Q,eps,valid_actions)
    a_prime = pi(s_prime)
    Q[s][a] += alpha * (r + gamma * Q[s_prime][a_prime] - Q[s][a])
    s = s_prime
    a = a_prime
    episode += [s]
  episode += [goal]
  lens += [len(episode)]
  timesteps += [timesteps[-1]+len(episode)]
  if i%100==0:
    x = [a[0] for a in episode]
    y = [a[1] for a in episode]
    plt.plot(x,y,'-x')
    plt.xlim(-1,sx)
    plt.ylim(-1,sy)
    plt.show()

plt.plot(timesteps,list(range(max_ep+1)))