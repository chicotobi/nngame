import numpy as np
import numpy.random as npr

def argmax(dct):
  v=list(dct.values())
  return list(dct.keys())[v.index(max(v))]    

def all_argmax(dct):
  vmax = -np.Infinity
  ans = []
  for (k,v) in dct.items():
    if v == vmax:
      ans += [k]
    if v > vmax:
      ans = [k]
      vmax = v
  return ans

def sample(v):
  return v[npr.choice(len(v))]