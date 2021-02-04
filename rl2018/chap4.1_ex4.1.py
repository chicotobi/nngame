# Gridworld
import numpy as np
import tabulate

# States are indices 0 to 14 - 0 is terminal
# +--+--+--+--+
# |  | 1| 2| 3|
# +--+--+--+--+
# | 4| 5| 6| 7|
# +--+--+--+--+
# | 8| 9|10|11|
# +--+--+--+--+
# |12|13|14|  |
# +--+--+--+--+
states = list(range(15))
nstates = len(states)

# Actions are "left", "up", "right", "down"
actions = ["left","up","right","down"]

# Rewards are -1
rewards = [-1,0]

# Discount parameter
gamma = 1

def state_transition(s_prime, r, s, a):
  if s==0:
    return (r==0 and s_prime==s)    
  if s in [1,2,3] and a=="up":
    return (r==-1 and s_prime==s)    
  if s in [12,13,14] and a=="down":
    return (r==-1 and s_prime==s)    
  if s in [4,8,12] and a=="left":
    return (r==-1 and s_prime==s)    
  if s in [3,7,11] and a=="right":
    return (r==-1 and s_prime==s)
  if s==1 and a=="left":
    return (r==-1 and s_prime==0)
  if s==4 and a=="up":
    return (r==-1 and s_prime==0)
  if s==14 and a=="right":
    return (r==-1 and s_prime==0)
  if s==11 and a=="down":
    return (r==-1 and s_prime==0)
  if a=="up":
    return (r==-1 and s_prime == s-4)
  if a=="down":
    return (r==-1 and s_prime == s+4)
  if a=="left":
    return (r==-1 and s_prime == s-1)
  if a=="right":
    return (r==-1 and s_prime == s+1)
  
def policy(a,s):
  return 1 / len(actions)

# Iterative policy evaluation
v_pi = np.zeros((nstates,1))
its = 0
while True:
  # Iterate
  its += 1
  Delta = 0
  v_pi_new = np.zeros((nstates,1))
  for s in states:
    for a in actions:
      for s_prime in states:
        for r in rewards:
          v_pi_new[s] += policy(a,s) * state_transition(s_prime,r,s,a) * (r + gamma * v_pi[s_prime])
    Delta = max(Delta, abs( v_pi_new[s]-v_pi[s]))
  v_pi = v_pi_new.copy()
  if its in [1,2,3,10] or Delta < 1e-10:
    print(its,Delta)
    print(tabulate.tabulate(np.round(np.array(v_pi.tolist()+[v_pi[0]]).reshape((4,4)),1)))
    print()
    if Delta < 1e-8:
      break
    
def q(s,a):
  val = 0
  for s_prime in states:
    for r in rewards:
      val += state_transition(s_prime, r, s, a) * (r + gamma * v_pi[s_prime])
  return val
      
print('q(11,"down")',q(11,"down"))
print('q( 7,"down")',q( 7,"down"))