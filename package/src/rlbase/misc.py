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