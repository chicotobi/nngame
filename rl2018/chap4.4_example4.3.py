import rl_functions
import numpy as np
import matplotlib.pyplot as plt

states = list(range(101))
terminal_states = [0,100]

actions = list(range(51))

p_h = .4

def state_transition(s, a):
  
  if s in terminal_states:
    return [(s,0,1)]

  #If the action is impossible, return 0 probability
  if a > s or a > 100-s:
    return []
  
  if a+s >= 100:
    r = 1
  else:
    r = 0

  return [(s-a,0,1-p_h),(min(s+a,100), r, p_h)]

# Evaluate policy and visualize value function
v, arr_v = rl_functions.value_iteration(states,actions,state_transition,tol=1e-4)
for i in arr_v:
  plt.plot(i[1:99])

plt.figure()
arr = np.zeros((101,101))
for s in range(1,100):
  for a in actions:
    v1 = 0
    for (s_prime, r, p) in state_transition(s,a):
      v1 += p * (r + v[s_prime])
    arr[s,a] = v1
plt.imshow(arr[1:99,:])

policy = rl_functions.improve_policy_from_value_function_two_arg(states,actions,state_transition,v,tol=1e-3)

plt.figure()
aa = []
ss = []
for s in range(1,100):
  for a in range(1,100):
    if policy(a,s)>0:
      aa.append(a)
      ss.append(s)
      break
plt.plot(ss,aa,'x')

pol = rl_functions.get_deterministic_policy_from_policy_function(states,actions,policy)