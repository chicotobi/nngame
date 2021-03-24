from agent import BaseAgent
from misc import argmax_dct

class QlearningAgent(BaseAgent):
    def __init__(self):
        pass

    def agent_init(self, agent_info={}):
        self.alpha = agent_info.get("alpha",0.5)
        self.gamma = agent_info.get("gamma",1)
        self.pi = agent_info["pi"]
        env = agent_info["env"]
        self.q = {s:{a:0 for a in env.valid_actions[s]} for s in env.states}

    def agent_start(self, s):
        self.last_state = s
        self.last_action = self.pi.get(s)
        return self.last_action

    def agent_step(self, r, s):
      
        a = self.pi.get(s)   
        a0 = argmax_dct(self.q[s])
        
        self.q[self.last_state][self.last_action] += self.alpha * (r + self.gamma * self.q[s][a0] - self.q[self.last_state][self.last_action])
                
        self.pi.update(s,a0)
        
        self.last_state = s
        self.last_action = a
        return self.last_action

    def agent_end(self, reward):
        pass

    def agent_cleanup(self):
        pass

    def agent_message(self, message):
        pass