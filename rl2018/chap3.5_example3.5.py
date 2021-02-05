# Gridworld
import numpy as np
import tabulate as tb
import rl_functions

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

# Actions
actions = ["left","up","right","down"]

# Rewards
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

value, _ = rl_functions.evaluate_policy_iterative(states,actions,rewards,state_transition,policy,gamma)
print(np.round(value,1).reshape((5,5)))

# Compute improved policy from value function
value = rl_functions.evaluate_policy_linear_system(states,actions,rewards,state_transition,policy,gamma)
improved_policy = rl_functions.improve_policy_from_value_function(states,actions,rewards,state_transition,value,gamma)
value = rl_functions.evaluate_policy_linear_system(states,actions,rewards,state_transition,improved_policy,gamma)
improved_policy = rl_functions.improve_policy_from_value_function(states,actions,rewards,state_transition,value,gamma)
value = rl_functions.evaluate_policy_linear_system(states,actions,rewards,state_transition,improved_policy,gamma)
print(np.round(value,1).reshape((5,5)))

tmp = [""]*25
for s in states:
  for a in actions:
    if improved_policy(a,s)>0:
      tmp[s] += a[0]
print(tb.tabulate(np.array(tmp).reshape((5,5))))