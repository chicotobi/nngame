from abc import ABCMeta, abstractmethod

class BaseExperiment:
  
    __metaclass__ = ABCMeta

    def __init__(self, environment, agent):
        self.env = environment
        self.agent = agent
        
    @abstractmethod
    def episode(self):
      pass
      
    @abstractmethod
    def train(self):
      pass              