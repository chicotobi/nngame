# Gridworld
import numpy as np
import tabulate as tb
import rl_functions

from gridworld import *
from policy import *

g = GridworldEx35()

p = UniformPolicy(g.states,g.actions)

# Discount parameter
gamma = 0.9

for it in range(5):
  value, _ = rl_functions.evaluate_policy_iterative(g.states,g.actions,g.rewards,g.state_transition,p.prob,gamma)
  v = np.zeros((g.sx,g.sy))
  for i in range(g.sx):
    for j in range(g.sy):
      v[i,j] = value[(i,j)]
  print("Iteration",it)
  print(np.flipud(np.round(v,1).transpose()))
  p = rl_functions.improve_policy_from_value_function(g.states,g.actions,g.rewards,g.state_transition,value,gamma)

tmp = np.ndarray((g.sx,g.sy),dtype=str)
for s in g.states:  
  for a in g.actions:
    if p.prob(a,s)>0:
      tmp[s[0],s[1]] += a[0]
print(tb.tabulate(np.array(tmp).reshape((5,5))))