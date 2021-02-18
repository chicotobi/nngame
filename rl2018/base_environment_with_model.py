import abc
from base_environment import BaseEnvironment

class BaseEnvironmentWithModel(BaseEnvironment):
  def __init__(self, states, actions, rewards):
    self.rewards = rewards
    super().__init__(states,actions)
    
  @abc.abstractmethod
  def state_transition(self, s_prime, r, s, a):
    ...