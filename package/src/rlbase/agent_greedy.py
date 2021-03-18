from agent import BaseAgent
from misc import argmax

import numpy as np
import math

class EpsGreedyAgent(BaseAgent):
    def __init__(self):
        self.last_action = None
        self.num_actions = None
        self.q_values = None
        self.step_size = None
        self.epsilon = None
        self.initial_value = None
        self.action_count = None

    def agent_init(self, agent_info={}):
        self.num_actions = agent_info["num_actions"]
        self.initial_value = agent_info.get("initial_value", 0.0)
        self.q_values = np.ones(self.num_actions) * self.initial_value
        self.step_size = agent_info.get("step_size")
        self.epsilon = agent_info.get("epsilon",0)
        self.ucb_c = agent_info.get("ucb_c")
        self.last_action = 0
        self.action_count = np.zeros(self.num_actions)

    def agent_start(self, observation):
        self.last_action = np.random.choice(self.num_actions)
        if self.ucb_c:
          self.t = 0
        return self.last_action

    def agent_step(self, reward, observation):
        if not self.step_size or self.ucb_c:
          self.action_count[self.last_action] += 1
          
        if self.step_size:
          stepsize = self.step_size
        else:
          stepsize = 1 / self.action_count[self.last_action]
        self.q_values[self.last_action] += stepsize * (reward - self.q_values[self.last_action])
        
        if np.random.rand() > self.epsilon:
          if self.ucb_c:
            self.t += 1
            current_action = argmax(self.q_values + self.ucb_c * (math.log(self.t) / np.maximum(1,self.action_count))**.5 )
          else:
            current_action = argmax(self.q_values)            
        else:
          current_action = np.random.choice(self.num_actions)
    
        self.last_action = current_action        
        return current_action

    def agent_end(self, reward):
        pass

    def agent_cleanup(self):
        pass

    def agent_message(self, message):
        pass