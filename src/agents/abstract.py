from src.core.agent import Agent

class TaskAgent(Agent):
    def __init__(self, origin: str, destination: str, reward: float = 10.0):
        super().__init__(role="task")
        self.origin = origin
        self.destination = destination
        self.reward = reward
        self.status = "waiting" # waiting, in_transit, completed

    def perceive(self, world):
        pass

    def decide(self):
        return []
