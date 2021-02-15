import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr

p = "C:/Users/hofmant3/nngame/rl2018/chap5.7_exercise5.12_racetrack_1"

field = np.array([[int(i) for i in l] for l in open(p).read().splitlines()])
field = np.flip(field,axis=0)
field = np.swapaxes(field,0,1)

plt.imshow(np.swapaxes(field,0,1),origin="lower")

sx, sy = field.shape

states = []
start_positions = []
final_positions = []
for x in range(sx):
  for y in range(sy):
    if field[x,y] != 0:
      states += [(x,y,vx,vy) for vx in range(-5,6) for vy in range(0,6)]
    if field[x,y] == 2:
      start_positions += [(x,y)]
    if field[x,y] == 3:
      final_positions += [(x,y)]
    

actions = [(ax,ay) for ax in range(-1,2) for ay in range(-1,2)]

def step(s,a):
  x,y,vx,vy = s
  ax, ay = a
  
  # Car is in goal
  if (x,y) in final_positions:
    return None, -1
  
  # Calculate next state
  vx += ax
  vy += ay
  x += vx
  y += vy

  # Car still on track
  on_track = True
  if 0<x<sx and 0<y<sy:
    if field[x,y] == 0:
      on_track = False
  else:
    on_track = False
    
  # Reset onto track
  if not on_track:
    x,y = start_positions[npr.choice(range(len(start_positions)))]
    vx = 0
    vy = 0
    
  return (x,y,vx,vy), -1

eps = 0.2
n_eps_red = 10
max_ep = 1e4

gamma = 1

valid_actions = {s:[] for s in states}
for s in states:
  _,_,vx,vy = s
  for a in actions:
    ax,ay = a
    if -6 < vx+ax < 6 and -1 < vy+ay < 6:
      valid_actions[s] += [a]

n_valid_actions = {k:len(v) for (k,v) in valid_actions.items()}

def generate_eps_soft_policy(pi,eps):
  pp = pi
  p_eps_soft = {k:1-eps+eps/v for (k,v) in n_valid_actions.items()}
  def tmp(s):
    a0 = pp[s]
    if npr.rand() < p_eps_soft[s]:
      return a0
    else:
      while True:
        a = valid_actions[s][npr.choice(n_valid_actions[s])]
        if a != a0:
          return a
  def tmp2(s,a):
    if a == pp[s]:
      return 1-eps+eps/n_valid_actions[s]
    else:
      return eps/n_valid_actions[s]
  return tmp, tmp2


C = {(s,a):0 for s in states for a in valid_actions[s]}
Q = {(s,a):-1e6 for s in states for a in valid_actions[s]}
pi = {s:(0,0) for s in states}

n_episodes = int(max_ep)
lens = np.zeros(n_episodes)
for i in range(n_episodes):
  if i%n_eps_red==n_eps_red-1:
    eps /= 2
  # Create soft policy (every ... episodes to avoid tight coupling)
  #if i%100 == 0:
  b, b_prob = generate_eps_soft_policy(pi,eps)
  
  # Generate an episode using b
  pos = start_positions[npr.choice(range(len(start_positions)))]
  s = (pos[0], pos[1], 0, 0)
  episode = []
  while s:
    a = b(s)
    s_prime,r = step(s,a)
    episode.append((s,a,r))
    s = s_prime
    
  G = 0
  W = 1
  # Loop for each step of episode
  for (s,a,r) in episode[::-1]:
     G = gamma * G + r
     C[(s,a)] += W
     Q[(s,a)] += W/C[(s,a)]*(G-Q[(s,a)])
     
     # Update target policy
     best_value =  - np.Infinity
     for a1 in valid_actions[s]:
       if Q[(s,a1)] > best_value:
         best_value = Q[(s,a1)]
         pi[s] = a1
         
     # If A_t != pi[S] exit loop 
     if a != pi[s]:
       break
     
     W *= 1/b_prob(s,a)
  
  lens[i] = len(episode)
       
  if i%n_eps_red==0:
    print("Episode ",i, " Length",len(episode))
    x = [x for ((x,_,_,_),_,_) in episode]
    y = [y for ((_,y,_,_),_,_) in episode]
  
    plt.subplot(2,2,1)
    plt.imshow(np.swapaxes(field,0,1),origin="lower")
    plt.plot(x,y,'rx-')
    
    plt.subplot(2,2,2)
    plt.imshow(np.swapaxes(field,0,1),origin="lower")
    arr_u = np.zeros((sx,sy))
    arr_v = np.zeros((sx,sy))
    for s in states:
      ax,ay = pi[s]
      x,y,_,_ = s
      arr_u[x,y] += ax #/ 66
      arr_v[x,y] += ay #/ 66
    for x in range(sx):
      for y in range(sy):
        v = (arr_u[x,y]**2 + arr_v[x,y]**2)**.5
        if v>0:
          arr_u[x,y] /= v
          arr_v[x,y] /= v
    plt.quiver(np.swapaxes(arr_u,0,1),np.swapaxes(arr_v,0,1),units='xy',angles='xy',scale=1)
        
    plt.subplot(2,2,(3,4))
    plt.semilogy(lens)
    plt.xlim(0,n_episodes)
    
    plt.show()

