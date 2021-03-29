import numpy as np

def sample(v):
  return v[np.random.choice(len(v))]

def argmax(dct):
  vmax = -np.Infinity
  ans = []
  for (k,v) in dct.items():
    if v == vmax:
      ans += [k]
    if v > vmax:
      ans = [k]
      vmax = v
  return ans

def argmax_unique(dct):
  return sample(argmax(dct))

def softmax(v):
  vmax = np.max(v)    
  exp_preferences = np.exp(v-vmax)  
  sum_of_exp_preferences = np.sum(exp_preferences)
  return exp_preferences / sum_of_exp_preferences    
