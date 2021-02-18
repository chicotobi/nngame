import abc
import numpy.random as npr

class BaseEnvironment:
  def __init__(self, states, actions):
    self.states = states
    self.actions = actions
    self.n_states = len(states)
    
  def get_random_state(self):
    return self.states[npr.choice(self.n_states)]
    
  @abc.abstractmethod
  def step(self,s,a):
    ...      
      