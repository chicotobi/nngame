# Gridworld
import numpy as np
import tabulate as tb
import rl_functions

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

# Actions
actions = ["left","up","right","down"]

# Rewards
rewards = [-1,0]

# Discount parameter
gamma = 1

terminal_states = [0]

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

v = rl_functions.evaluate_policy_linear_system(states,actions,rewards,state_transition,policy,gamma,terminal_states)
_, arr_v = rl_functions.evaluate_policy_iterative(states,actions,rewards,state_transition,policy,gamma)

for (i,x) in enumerate(arr_v):
  if i in [1,2,3,10,425]:
    print("Iteration: ",i)
    print(tb.tabulate(np.round(np.array(v.tolist()+[v[0]]).reshape((4,4)),1)))
    print()

q = rl_functions.get_action_value_function(states,rewards,state_transition,gamma,v)

print('q(11,"down")',round(q(11,"down"),1))
print('q( 7,"down")',round(q( 7,"down"),1))