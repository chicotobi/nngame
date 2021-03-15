import abc
import numpy.random as npr

class BaseEnvironment(metaclass=abc.ABCMeta):
  def __init__(self, states, actions,valid_actions=None):
    self.states = states
    self.actions = actions
    self.n_states = len(states)
    self.type_random_initial_state = 0
    self.counter = 0
    if valid_actions:
      self.valid_actions = valid_actions
    
  def get_random_state(self):
    return self.states[npr.choice(self.n_states)]
  
  def get_random_initial_state(self):
    if self.type_random_initial_state == 0:
      return self.get_random_state()
    else:
      self.counter = (self.counter + 1) % self.n_states
      return self.states[self.counter]
    
  @abc.abstractmethod
  def step(self,s,a):
    ...      
  