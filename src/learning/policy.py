from abc import ABC, abstractmethod
from typing import Any, List, Dict
import random

class Policy(ABC):
    @abstractmethod
    def decide(self, agent_state: Dict[str, Any], world_view: Any) -> List[Any]:
        """
        Decide on an action based on the agent's state and its view of the world.
        """
        pass

    def update(self, reward: float, done: bool):
        """
        Update the policy based on feedback (for RL).
        """
        pass

class RandomPolicy(Policy):
    def __init__(self, action_space: List[Any]):
        self.action_space = action_space

    def decide(self, agent_state: Dict[str, Any], world_view: Any) -> List[Any]:
        if not self.action_space:
            return []
        return [random.choice(self.action_space)]

class RuleBasedPolicy(Policy):
    def __init__(self, rules: callable):
        """
        rules: A function that takes (state, world_view) and returns actions.
        """
        self.rules = rules

    def decide(self, agent_state: Dict[str, Any], world_view: Any) -> List[Any]:
        return self.rules(agent_state, world_view)
