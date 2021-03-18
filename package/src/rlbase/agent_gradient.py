from agent import BaseAgent
from misc import argmax

import numpy as np
import math
from misc import softmax

class GradientAgent(BaseAgent):
    def __init__(self):
        self.last_action = None
        self.num_actions = None
        self.h_values = None
        self.step_size = None
        self.epsilon = None
        self.initial_value = None
        self.action_count = None

    def agent_init(self, agent_info={}):
        self.num_actions = agent_info["num_actions"]
        self.initial_value = agent_info.get("initial_value", 0.0)
        self.h_values = np.ones(self.num_actions) * self.initial_value
        self.step_size = agent_info["step_size"]
        self.epsilon = agent_info.get("epsilon",0)
        self.last_action = 0
        self.action_count = np.zeros(self.num_actions)
        self.baseline = agent_info.get("baseline")

    def agent_start(self, observation):
        self.last_action = np.random.choice(self.num_actions)
        self.t = 0
        self.average_reward = 0
        return self.last_action

    def agent_step(self, reward, observation):
      
        pi = softmax(self.h_values)
        
        self.t += 1
        if self.baseline:
          self.average_reward += (reward - self.average_reward) / self.t
      
        for a in range(self.num_actions):
          if a == self.last_action:
            self.h_values[a] += self.step_size * (reward - self.average_reward) * (1-pi[a])
          else:
            self.h_values[a] -= self.step_size * (reward - self.average_reward) * pi[a]
            
        current_action = np.random.choice(list(range(self.num_actions)),p=list(pi))
    
        self.last_action = current_action        
        return current_action

    def agent_end(self, reward):
        pass

    def agent_cleanup(self):
        pass

    def agent_message(self, message):
        pass