import rl_functions

from gridworld import GridworldEx35
from policy import UniformPolicy

g = GridworldEx35()

p = UniformPolicy(g.states,g.actions)

for it in range(5):
  print("Iteration",it)
  value, _ = rl_functions.evaluate_policy_iterative(g.states,g.actions,g.rewards,g.state_transition,p.prob,gamma=0.9,tol=1e-3)
  g.plot(value)
  p = rl_functions.improve_policy_from_value_function(g.states,g.actions,g.rewards,g.state_transition,value,gamma)
  g.plot_bestaction_policy(p)
  