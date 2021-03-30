import numpy as np

import rlbase.policy as policy

# For all s:
# (1 - Sum(a,s_prime,r) pi(a|s) * p(s_prime,r|s,a) * gamma ) * v_pi(s) = Sum(a,s_prime,r) pi(a|s) * p(s_prime,r|s,a) * r
def evaluate_policy_linear_system(env,pi,gamma=1):
  nstates = len(env.states)
  A = np.zeros((nstates,nstates))
  b = np.zeros((nstates,1))
  for (i,s) in enumerate(env.states):
    A[i,i] = 1
    for (j,s_prime) in enumerate(env.states):
      for a in env.actions:
        for r in env.rewards:
          b[i] += pi.prob(a,s) * env.state_transition(s_prime,r,s,a) * r
          A[i,j] -= pi.prob(a,s) * env.state_transition(s_prime,r,s,a) * gamma       
  idx_states_plus = [i for (i,s) in enumerate(env.states) if s not in env.terminal_states]
  value = np.zeros((nstates,1))
  value[idx_states_plus] = np.linalg.solve(A[np.ix_(idx_states_plus,idx_states_plus)],b[idx_states_plus])
  return dict(zip(env.states,value))
  
def evaluate_policy_iterative(env,pi,gamma=1,tol=1e-10):
  v = {s:0 for s in env.states}
  arr_v = []
  its = 0
  while True:
    arr_v.append(v)
    its += 1
    v_new = {s:0 for s in env.states}
    for s in env.states:
      for a in env.actions:
        for s_prime in env.states:
          for r in env.rewards:
            v_new[s] += pi.prob(a,s) * env.state_transition(s_prime,r,s,a) * (r + gamma * v[s_prime])
    Delta = max([abs(v[s]-v_new[s]) for s in env.states])
    v = v_new.copy()
    if Delta < tol:
      break
  return v, arr_v

def improve_policy_from_value_function(env,values,gamma=1,tol=1e-5):
  improved_policy = {}
  for s in env.states:
    improved_actions = []
    improved_value = - np.Infinity
    for a in env.actions:
      v = 0
      for s_prime in env.states:
        for r in env.rewards:
          v += env.state_transition(s_prime, r, s, a) * (r + gamma * values[s_prime])
      if v >= improved_value-tol:
        if abs(improved_value-v)<tol:
          improved_actions.append(a)
        else:
          improved_value = v          
          improved_actions = [a]
    improved_policy[s] = improved_actions
  return policy.BestActionPolicy(env=env,best_actions=improved_policy)

def evaluate_policy_linear_system_two_args(env,pi,gamma=1):
  idx = {j:i for (i,j) in enumerate(env.states)}
  nstates = len(env.states)
  A = np.zeros((nstates,nstates))
  b = np.zeros((nstates,1))
  for s in env.states:
    A[idx[s],idx[s]] = 1
    for a in env.actions:
      for (s_prime, r, p) in env.state_transition_two_args(s,a):
        b[idx[s]] += pi.prob(a,s) * p * r
        A[idx[s],idx[s_prime]] -= pi.prob(a,s) * p * gamma

  idx_states_plus = [idx[s] for s in env.states if s not in env.terminal_states]
  value = np.zeros((nstates,1))
  value[idx_states_plus] = np.linalg.solve(A[np.ix_(idx_states_plus,idx_states_plus)],b[idx_states_plus])
  return dict(zip(env.states,value))

def evaluate_policy_iterative_two_args(env,pi,gamma=1,tol=1e-10):
  v = {s:0 for s in env.states}
  arr_v = []
  its = 0
  while True:
    arr_v.append(v)
    its += 1
    v_new = {s:0 for s in env.states}
    for s in env.states:
      for a in env.actions:
        for (s_prime, r, p) in env.state_transition_two_args(s,a):
          v_new[s] += pi.prob(a,s) * env.state_transition(s_prime,r,s,a) * (r + gamma * v[s_prime])
    Delta = max([abs(v[s]-v_new[s]) for s in env.states])
    v = v_new.copy()
    if Delta < tol:
      break
  return v, arr_v

def improve_policy_from_value_function_two_args(env,values,gamma=1,tol=1e-10):
  improved_policy = {}
  for s in env.states:
    improved_actions = []
    improved_value = - np.Infinity
    for a in env.actions:
      v = 0
      for (s_prime, r, p) in env.state_transition_two_args(s,a):
        v += p * (r + gamma * values[s_prime])
      if v > improved_value:
        improved_value = v
        improved_actions = []
      if abs(improved_value-v)<tol:
        improved_actions.append(a)
    improved_policy[s] = improved_actions
  return policy.BestActionPolicy(env=env,best_actions=improved_policy)

def value_iteration_two_args(env,gamma=1,tol=1e-10):
  v = {s:0 for s in env.states}
  v_new = v.copy()
  arr_v = []
  while True:
    arr_v.append(v)
    Delta = 0
    for s in env.states:
      v_new[s] = 0
      for a in env.actions:
        tmp = 0
        for (s_prime, r, p) in env.state_transition_two_args(s,a):
          tmp += p * (r + gamma * v[s_prime])
        v_new[s] = max(v_new[s], tmp)
      Delta = max(Delta,abs(v_new[s]-v[s]))
    v = v_new.copy()
    if Delta < tol:
      break
  return v, arr_v

def get_action_value_function(env,v,gamma=1):
  def q(s,a):
    val = 0
    for s_prime in env.states:
      for r in env.rewards:
        val += env.state_transition(s_prime, r, s, a) * (r + gamma * v[s_prime])
    return val
  return q