from environment import BaseEnvironment

from functools import lru_cache
from scipy.stats import poisson

import numpy as np

# Memoization of the Poisson distribution for acceleration
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
    
class CarRentalEnvironment(BaseEnvironment):

    def __init__(self,**kwargs):
      self.nmax = kwargs.get("nmax",20)
      self.max_move = kwargs.get("max_move",5)
      self.max_return = kwargs.get("max_return",3)
      
      self.states = [(i,j) for i in range(self.nmax+1) for j in range(self.nmax+1)]
      self.terminal_states = []
      
      # Number of cars moved from first to second location
      self.actions = list(range(-self.max_move,self.max_move+1))
      self.set_all_actions_valid() # TODO This is wrong, but difficult
      
    def reshape(self,field):     
      v = np.zeros((self.nmax,self.nmax))
      for i in range(self.nmax):
        for j in range(self.nmax):
          v[i,self.nmax-j-1] = field[(i,j)]
      return v

    # This function returns a list of s_prime and corresponding probabilities
    def state_transition_two_args(self, s, a):
    
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
          for n_return_first in range(self.max_return+1):
            for n_return_second in range(self.max_return+1):
              p = mypois(n_request_first ,lambda_request_first , n_first   ) * \
                  mypois(n_request_second,lambda_request_second, n_second  ) * \
                  mypois(n_return_first  ,lambda_return_first  , self.max_return) * \
                  mypois(n_return_second ,lambda_return_second , self.max_return)
              s_prime = (\
                min(self.nmax,n_first -n_request_first +n_return_first),\
                min(self.nmax,n_second-n_request_second+n_return_second)\
               )
              if p>0:
                ans.append((s_prime, r, p))
      return ans
