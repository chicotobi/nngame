# Gridworld
import numpy as np

# States are indices 0 to 24
# +--+--+--+--+--+
# | 0| 1| 2| 3| 4|
# +--+--+--+--+--+
# | 5| 6| 7| 8| 9|
# +--+--+--+--+--+
# |10|11|12|13|14|
# +--+--+--+--+--+
# |15|16|17|18|19|
# +--+--+--+--+--+
# |20|21|22|23|24|
# +--+--+--+--+--+
states = list(range(25))

# Actions are "left", "up", "right", "down"
actions = ["left","up","right","down"]

# Rewards are -1, 0, 5, 10
rewards = [-1, 0, 5, 10]

state_A = 1
state_A_prime = 21
reward_A = 10

state_B = 3
state_B_prime = 13
reward_B = 5

# Discount parameter
gamma = 0.9

def state_transition(s_prime, r, s, a):
  if s == state_A:
    return (s_prime==state_A_prime and r==reward_A)
  if s == state_B:
    return (s_prime==state_B_prime and r==reward_B)
  if s in [0,1,2,3,4] and a=="up":
    return (r==-1 and s_prime==s)    
  if s in [20,21,22,23,24] and a=="down":
    return (r==-1 and s_prime==s)    
  if s in [0,5,10,15,20] and a=="left":
    return (r==-1 and s_prime==s)    
  if s in [4,9,14,19,24] and a=="right":
    return (r==-1 and s_prime==s)    
  if a=="up":
    return (r==0 and s_prime == s-5)
  if a=="down":
    return (r==0 and s_prime == s+5)
  if a=="left":
    return (r==0 and s_prime == s-1)
  if a=="right":
    return (r==0 and s_prime == s+1)
  
def policy(a,s):
  return 1 / len(actions)

# Compute the Bellman equation:
# For all s:
# (1 - Sum(a,s_prime,r) pi(a|s) * p(s_prime,r|s,a) * gamma ) * v_pi(s) = Sum(a,s_prime,r) pi(a|s) * p(s_prime,r|s,a) * r
A = np.zeros((25,25))
b = np.zeros((25,1))
for s in states:
  A[s,s] = 1
  for s_prime in states:
    for a in actions:
      for r in rewards:
        b[s] += policy(a,s) * state_transition(s_prime,r,s,a) * r
        A[s,s_prime] -= policy(a,s) * state_transition(s_prime,r,s,a) * gamma
    
value = np.linalg.solve(A,b)
print(np.round(value,1).reshape((5,5)))



  