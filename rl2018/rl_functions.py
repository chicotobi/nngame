import numpy as np

# For all s:
# (1 - Sum(a,s_prime,r) pi(a|s) * p(s_prime,r|s,a) * gamma ) * v_pi(s) = Sum(a,s_prime,r) pi(a|s) * p(s_prime,r|s,a) * r
def evaluate_policy_linear_system(states,actions,rewards,state_transition,policy,gamma,terminal_states=[]):
  nstates = len(states)
  A = np.zeros((nstates,nstates))
  b = np.zeros((nstates,1))
  for s in states:
    A[s,s] = 1
    for s_prime in states:
      for a in actions:
        for r in rewards:
          b[s] += policy(a,s) * state_transition(s_prime,r,s,a) * r
          A[s,s_prime] -= policy(a,s) * state_transition(s_prime,r,s,a) * gamma

  states_plus = [s for s in states if s not in terminal_states]

  value = np.zeros((nstates,1))
  value[states_plus] = np.linalg.solve(A[np.ix_(states_plus,states_plus)],b[states_plus])
  return value

def evaluate_policy_iterative(states,actions,rewards,state_transition,policy,gamma,tol=1e-10):
  nstates = len(states)
  v = np.zeros((nstates,1))
  arr_v = []
  its = 0
  while True:
    # Iterate
    arr_v.append(v)
    its += 1
    Delta = 0
    v_new = np.zeros((nstates,1))
    for s in states:
      for a in actions:
        for s_prime in states:
          for r in rewards:
            v_new[s] += policy(a,s) * state_transition(s_prime,r,s,a) * (r + gamma * v[s_prime])
      Delta = max(Delta, abs(v_new[s]-v[s]))
    v = v_new.copy()
    if Delta < tol:
      break
  return v, arr_v

def improve_policy_from_value_function(states,actions,rewards,state_transition,value_function,gamma,tol=1e-5):
  improved_policy = {}
  for s in states:
    improved_actions = []
    improved_value = - np.Infinity
    for a in actions:
      for s_prime in states:
        for r in rewards:
          if state_transition(s_prime, r, s, a)>0:
            v = r + gamma * state_transition(s_prime, r, s, a) * value_function[s_prime]
            if v > improved_value:
              improved_value = v
              improved_actions = []
            if abs(improved_value-v)<tol:
              improved_actions.append(a)
    improved_policy[s] = improved_actions
  def p(a,s):
    if a in improved_policy[s]:
      return 1/len(improved_policy[s])
    else:
      return 0
  return p

def get_action_value_function(states,rewards,state_transition,gamma,value_function):
  def q(s,a):
    val = 0
    for s_prime in states:
      for r in rewards:
        val += state_transition(s_prime, r, s, a) * (r + gamma * value_function[s_prime])
    return val[0]
  return q

def evaluate_policy_linear_system_two_arg(states,actions,state_transition,policy,gamma,terminal_states=[]):
  idx = {j:i for (i,j) in enumerate(states)}
  nstates = len(states)
  A = np.zeros((nstates,nstates))
  b = np.zeros((nstates,1))
  for s in states:
    print(s)
    A[idx[s],idx[s]] = 1
    for a in actions:
      tmp = state_transition(s,a)
      for (s_prime, r, p) in tmp:
        b[idx[s]] += policy(a,s) * p * r
        A[idx[s],idx[s_prime]] -= policy(a,s) * p * gamma

  idx_states_plus = [idx[s] for s in states if s not in terminal_states]

  value = np.zeros((nstates,1))
  value[idx_states_plus] = np.linalg.solve(A[np.ix_(idx_states_plus,idx_states_plus)],b[idx_states_plus])
  return value

def improve_policy_from_value_function_two_arg(states,actions,state_transition,value_function,gamma,tol=1e-5):
  idx = {j:i for (i,j) in enumerate(states)}
  improved_policy = {}
  for s in states:
    improved_actions = []
    improved_value = - np.Infinity
    for a in actions:
      print(s,a)
      tmp = state_transition(s,a)
      v = 0
      for (s_prime, r, p) in tmp:
        v += r + gamma * p * value_function[idx[s_prime]]
      if v > improved_value:
        improved_value = v
        improved_actions = []
      if abs(improved_value-v)<tol:
        improved_actions.append(a)
    improved_policy[s] = improved_actions
  def myf(a,s):
    if a in improved_policy[s]:
      return 1/len(improved_policy[s])
    else:
      return 0
  return myf