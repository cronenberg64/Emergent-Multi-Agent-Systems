import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
from src.learning.policy import Policy

class RLPolicy(Policy, nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        nn.Module.__init__(self)
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.saved_log_probs = []
        self.rewards = []

    def decide(self, agent_state: Dict[str, Any], world_view: Any) -> List[Any]:
        # Convert state to tensor
        # NOTE: This assumes agent_state contains a 'vector' key or similar that is already numerical
        # In a real app, we'd need a preprocessing step here.
        if 'vector' not in agent_state:
            return []
            
        state_tensor = torch.FloatTensor(agent_state['vector'])
        probs = self.network(state_tensor)
        
        # Sample action
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        
        self.saved_log_probs.append(m.log_prob(action))
        
        return [action.item()]

    def update(self, reward: float, done: bool):
        self.rewards.append(reward)
