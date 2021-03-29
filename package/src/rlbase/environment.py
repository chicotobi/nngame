import numpy.random as npr

class BaseEnvironment:
  states = None
  actions = None
  valid_actions = None

  def __init__(self):
    pass

  # Return state
  def start(self):
    pass

  # Get state, action, return (reward, state, terminal)
  def step(self):
    pass

  def get_initial_state(self):
    n_states = len(self.states)
    return self.states[npr.choice(n_states)]

  def set_all_actions_valid(self):
    self.valid_actions = {s: self.actions for s in self.states}