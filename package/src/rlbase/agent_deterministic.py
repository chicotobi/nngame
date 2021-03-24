from agent import BaseAgent

class DeterministicAgent(BaseAgent):

    def agent_init(self, agent_info={}):
        self.pi = agent_info["pi"]  

    def agent_start(self, observation):
        return self.pi.get(observation)

    def agent_step(self, reward, observation):
        return self.pi.get(observation)

    def agent_end(self, reward):
        pass

    def agent_cleanup(self):
        pass

    def agent_message(self, message):
        pass