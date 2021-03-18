import numpy as np

def argmax(v):
    top_value = float("-inf")
    ties = []    
    for i in range(len(v)):
        if v[i] > top_value:
            ties = [i]
            top_value = v[i]
        elif v[i] == top_value:
            ties.append(i)
    return np.random.choice(ties)

def softmax(v):
  vmax = np.max(v)    
  exp_preferences = np.exp(v-vmax)  
  sum_of_exp_preferences = np.sum(exp_preferences)
  return exp_preferences / sum_of_exp_preferences    

# For all s:
# (1 - Sum(a,s_prime,r) pi(a|s) * p(s_prime,r|s,a) * gamma ) * v_pi(s) = Sum(a,s_prime,r) pi(a|s) * p(s_prime,r|s,a) * r
# def evaluate_policy_linear_system(env,pi,gamma=1,terminal_states=[]):
#   nstates = len(env.states)
#   A = np.zeros((nstates,nstates))
#   b = np.zeros((nstates,1))
#   for (i,s) in enumerate(env.states):
#     A[i,i] = 1
#     for (j,s_prime) in enumerate(env.states):
#       for a in env.actions:
#         for r in env.rewards:
#           b[i] += pi(a,s) * env.state_transition(s_prime,r,s,a) * r
#           A[i,j] -= pi(a,s) * env.state_transition(s_prime,r,s,a) * gamma       
#   idx_states_plus = [i for (i,s) in enumerate(env.states) if s not in terminal_states]
#   value = np.zeros((nstates,1))
#   value[idx_states_plus] = np.linalg.solve(A[np.ix_(idx_states_plus,idx_states_plus)],b[idx_states_plus])
#   return value
  
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
  import policy
  improved_policy = {}
  for s in env.states:
    improved_actions = []
    improved_value = - np.Infinity
    for a in env.actions:
      v = 0
      for s_prime in env.states:
        for r in env.rewards:
          v += env.state_transition(s_prime, r, s, a) * (r + gamma * values[s_prime])
      if v > improved_value:
        improved_value = v
        improved_actions = []
      if abs(improved_value-v)<tol:
        improved_actions.append(a)
    improved_policy[s] = improved_actions
  return policy.BestActionPolicy(env.states,env.actions,improved_policy)

# def get_action_value_function(states,rewards,state_transition,gamma,value_function):
#   def q(s,a):
#     val = 0
#     for s_prime in states:
#       for r in rewards:
#         val += state_transition(s_prime, r, s, a) * (r + gamma * value_function[s_prime])
#     return val[0]
#   return q

# def evaluate_policy_linear_system_two_arg(states,actions,state_transition,policy,gamma=1,terminal_states=[]):
#   idx = {j:i for (i,j) in enumerate(states)}
#   nstates = len(states)
#   A = np.zeros((nstates,nstates))
#   b = np.zeros((nstates,1))
#   for s in states:
#     A[idx[s],idx[s]] = 1
#     for a in actions:
#       for (s_prime, r, p) in state_transition(s,a):
#         b[idx[s]] += policy(a,s) * p * r
#         A[idx[s],idx[s_prime]] -= policy(a,s) * p * gamma

#   idx_states_plus = [idx[s] for s in states if s not in terminal_states]
#   value = np.zeros((nstates,1))
#   value[idx_states_plus] = np.linalg.solve(A[np.ix_(idx_states_plus,idx_states_plus)],b[idx_states_plus])
#   return value

# def evaluate_policy_iterative_two_args(states,actions,state_transition,policy,gamma=1,tol=1e-10):
#   idx = {j:i for (i,j) in enumerate(states)}
#   nstates = len(states)
#   v = np.zeros((nstates,1))
#   arr_v = []
#   its = 0
#   while True:
#     arr_v.append(v)
#     its += 1
#     v_new = np.zeros((nstates,1))
#     for (i,s) in enumerate(states):
#       for a in actions:
#         for (s_prime, r, p) in state_transition(s,a):
#           v_new[i] += policy(a,s) * p * (r + gamma * v[idx[s_prime]])
#     Delta = max(abs(v_new-v))
#     v = v_new.copy()
#     if Delta < tol:
#       break
#   return v, arr_v

# def improve_policy_from_value_function_two_arg(states,actions,state_transition,value_function,gamma=1,tol=1e-10):
#   idx = {j:i for (i,j) in enumerate(states)}
#   improved_policy = {}
#   for (i,s) in enumerate(states):
#     improved_actions = []
#     improved_value = - np.Infinity
#     for a in actions:
#       v = 0
#       for (s_prime, r, p) in state_transition(s,a):
#         v += p * (r + gamma * value_function[idx[s_prime]])
#       if v > improved_value:
#         improved_value = v
#         improved_actions = []
#       if abs(improved_value-v)<tol:
#         improved_actions.append(a)
#     improved_policy[s] = improved_actions
#   def p(a,s):
#     if a in improved_policy[s]:
#       return 1/len(improved_policy[s])
#     else:
#       return 0
#   return p

# def get_deterministic_policy_from_policy_function(states,actions,policy):
#   pol = {s:0 for s in states}
#   for s in states:
#     largest_p = 0
#     for a in actions:
#       if policy(a,s) > largest_p:
#         largest_p = policy(a,s)
#         pol[s] = a
#   return pol


# def value_iteration(states,actions,state_transition,gamma=1,tol=1e-10):
#   idx = {j:i for (i,j) in enumerate(states)}
#   nstates = len(states)
#   v = np.zeros((nstates,1))
#   v_new = np.zeros((nstates,1))
#   arr_v = []
#   while True:
#     arr_v.append(v)
#     for (i,s) in enumerate(states):
#       for a in actions:
#         tmp = 0
#         for (s_prime, r, p) in state_transition(s,a):
#           tmp += p * (r + gamma * v[idx[s_prime]])
#         v_new[i] = max(v_new[i], tmp)
#     Delta = max(abs(v_new-v))
#     print(Delta)
#     v = v_new.copy()
#     if Delta < tol:
#       break
#   return v, arr_v