import numpy as np

from rlbase.environment import RodManeuverEnvironment

env = RodManeuverEnvironment()
     
#env.plot_states(env.solution)

# s0 = env.start
# while True:
#   print("State: ", s0)
#   env.plot_states([s0])
#   a0 = input()
#   s0, r = env.step(s0, a0) 

valid_states = [s for s in env.states if env.is_valid_state(s)]
n_states = len(valid_states)

idx0 = valid_states.index(env.start)
idx1 = valid_states.index(env.goal)

mat = np.zeros((n_states,n_states))

for (i,s) in enumerate(valid_states):
  for a in env.actions:
    s_prime, _ = env.step(s,a)
    j = valid_states.index(s_prime)
    mat[i,j] = 1
    
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

graph = csr_matrix(mat)

dist_matrix, predecessors = dijkstra(csgraph=graph, indices=idx0, return_predecessors=True)

tmp = idx1
my_solution = [valid_states[idx1]]
while predecessors[tmp] != -9999:
  tmp = predecessors[tmp]
  my_solution = [valid_states[tmp]] + my_solution
  
env.plot_states(my_solution)