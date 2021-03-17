from agent import BaseAgent
from misc import argmax

import numpy as np

class UcbAgent(BaseAgent):
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
        self.c = agent_info.get("c",0)
        self.last_action = 0
        self.action_count = np.zeros(self.num_actions)

    def agent_start(self, observation):
        self.last_action = np.random.choice(self.num_actions)
        return self.last_action

    def agent_step(self, reward, observation):
        if self.step_size:
          stepsize = self.step_size
        else:
          self.action_count[self.last_action] += 1
          stepsize = 1 / self.action_count[self.last_action]
        self.q_values[self.last_action] += stepsize * (reward - self.q_values[self.last_action])
        
        if np.random.rand() > self.epsilon:
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