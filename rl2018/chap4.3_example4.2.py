# Gridworld
import rl_functions
import numpy as np
from functools import lru_cache
from scipy.stats import poisson
import matplotlib.pyplot as plt

# States are tuples (i,j) ranging from (0,0) to (20,20)
# This means 21*21=441 states
nmax = 5
max_move = 1
max_return = 0

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

  lambda_request_first  = 1
  lambda_request_second = 1
  lambda_return_first   = 0
  lambda_return_second  = 0

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
          if p>1e-10:
            ans.append((s_prime, r, p))
  return ans

def policy(a,s):
  return (a==0)


def aaa(states,actions,state_transition,value_function,gamma,tol=1e-5):
  idx = {j:i for (i,j) in enumerate(states)}
  improved_policy = {}
  for s in states:
    improved_actions = []
    improved_value = - np.Infinity
    if s==(5,0):
      test=2*3
    for a in actions:
      print(s,a)
      tmp = state_transition(s,a)
      for (s_prime, r, p) in tmp:
        if p>0:
          v = r + gamma * p * value_function[idx[s_prime]]
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

v = rl_functions.evaluate_policy_linear_system_two_arg(states,actions,state_transition,policy,gamma)

plt.imshow(v.reshape(nmax+1,nmax+1), cmap='hot', interpolation='nearest',origin='lower')
plt.show()

pol1 = aaa(states,actions,state_transition,v,gamma)
arr = np.zeros((nmax+1,nmax+1))
for i in range(nmax+1):
  for j in range(nmax+1):
    most_probable = 0
    for a in actions:
      if pol1(a,(i,j)) > most_probable:
        arr[i,j] = a
        most_probable = pol1(a,(i,j))
print(arr)
