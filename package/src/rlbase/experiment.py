from abc import ABCMeta, abstractmethod

class BaseExperiment:
  
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.env = kwargs.get("env")
        self.agent = kwargs.get("agent")
        self.gamma = kwargs.get("gamma",1)
        self.n_episodes = int(kwargs.get("n_episodes"))
        self.callback = kwargs.get("callback")
        
        self.agent.gamma = self.gamma
        
    @abstractmethod
    def episode(self):
      pass
      
    @abstractmethod
    def train(self):
      pass              