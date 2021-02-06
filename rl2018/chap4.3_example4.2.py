import rl_functions
import numpy as np
from functools import lru_cache
from scipy.stats import poisson
import matplotlib.pyplot as plt

# States are tuples (i,j) ranging from (0,0) to (20,20)
# This means 21*21=441 states
nmax = 8
max_move = 3
max_return = 3

states = [(i,j) for i in range(nmax+1) for j in range(nmax+1)]

# Actions
# Number of cars moved from first to second location
actions = list(range(-max_move,max_move+1))

# Discount parameter
gamma = 0.9

@lru_cache(1000000)
def mypois(n,l,maxn):
  if n < maxn:
      return poisson.pmf(n ,l)
  if n == maxn:
      val = 1
      for i in range(maxn):
          val -= mypois(i,l,maxn)
      return val
  if n > maxn:
      return 0

def state_transition(s, a):

  n_first = s[0]
  n_second = s[1]

  #If the action is impossible, return 0 probability
  if a > n_first or -a > n_second:
    return []

  #Reward starts with number of moved cars * -2$
  r0 = - 2 * abs(a)
  n_first -= a
  n_second += a

  lambda_request_first  = 3
  lambda_request_second = 4
  lambda_return_first   = 3
  lambda_return_second  = 2

  ans = []
  for n_request_first in range(n_first+1):
    for n_request_second in range(n_second+1):
      r = r0 + 10*(n_request_first+n_request_second)
      s_prime = (n_first-n_request_first,n_second-n_request_second)
      for n_return_first in range(max_return+1):
        for n_return_second in range(max_return+1):
          p = mypois(n_request_first ,lambda_request_first , n_first   ) * \
              mypois(n_request_second,lambda_request_second, n_second  ) * \
              mypois(n_return_first  ,lambda_return_first  , max_return) * \
              mypois(n_return_second ,lambda_return_second , max_return)
          s_prime = (\
            min(nmax,n_first -n_request_first +n_return_first),\
            min(nmax,n_second-n_request_second+n_return_second)\
           )
          if p>0:
            ans.append((s_prime, r, p))
  return ans

def policy(a,s):
  return (a==0)

# Evaluate policy and visualize value function
v = rl_functions.evaluate_policy_linear_system_two_arg(states,actions,state_transition,policy,gamma)
plt.imshow(v.reshape(nmax+1,nmax+1), cmap='hot', interpolation='nearest',origin='lower')
plt.show()

# Improve policy and visualize policy function
pol1 = rl_functions.improve_policy_from_value_function_two_arg(states, actions, state_transition, v, gamma)
arr = np.zeros((nmax+1,nmax+1))
for i in range(nmax+1):
  for j in range(nmax+1):
    most_probable = 0
    for a in actions:
      if pol1(a,(i,j)) > most_probable:
        arr[i,j] = a
        most_probable = pol1(a,(i,j))
plt.imshow(arr,origin="lower")
plt.colorbar()
