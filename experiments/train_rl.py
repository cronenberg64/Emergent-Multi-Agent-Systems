import torch
import torch.optim as optim
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation.environment import LogisticsEnv
from src.learning.rl_policy import RLPolicy

def train():
    # Use default 12 agents (4 trucks, 8 warehouses)
    env = LogisticsEnv()
    
    # Hyperparameters
    # Obs: 2 features * 4 trucks + 8 warehouses = 8 + 8 = 16
    # Note: This assumes full observability of all agents.
    input_dim = (2 * 4) + 8 
    output_dim = 5 # 5 discrete actions
    learning_rate = 0.001
    episodes = 200 # Increase episodes for more complex task
    gamma = 0.99

    # Independent PPO: Shared policy for all homogeneous agents (trucks)
    policy = RLPolicy(input_dim=input_dim, output_dim=output_dim)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    
    print(f"Starting MARL training with {env.num_trucks} trucks...")
    
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        # Store trajectory
        # For IPPO with shared weights, we can just flatten all agent experiences into one buffer
        log_probs = []
        rewards = []
        
        while not done:
            # Prepare state for policy
            # In this simple env, obs is global state vector, shared by all agents
            state_dict = {'vector': obs}
            
            # Get actions for ALL trucks
            actions = []
            current_log_probs = []
            
            # We need to run the policy N times, or batch it.
            # RLPolicy.decide returns a list of actions.
            # If we pass the same state, it will sample independently.
            
            for _ in range(env.num_trucks):
                # In a real MARL setting, each agent might have a different observation (partial view).
                # Here we share the global view.
                action_list = policy.decide(state_dict, None)
                actions.append(action_list[0])
                # We need to capture the log_prob of this specific action.
                # RLPolicy stores it in self.saved_log_probs.
                # Since we called decide() multiple times, saved_log_probs grew by N.
            
            # Step env
            next_obs, reward, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            
            # Reward is global in this env (-0.01 per tick, +10 per delivery).
            # We assign this global reward to ALL agents (Cooperative).
            # So we push 'reward' N times to match the N actions.
            for _ in range(env.num_trucks):
                rewards.append(reward)
            
            episode_reward += reward
            obs = next_obs
            
        # Update Policy (REINFORCE)
        # We have N * T actions and N * T rewards.
        R = 0
        policy_loss = []
        returns = []
        
        # Calculate returns in reverse
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # policy.saved_log_probs contains all log_probs from all agents across time
        if len(policy.saved_log_probs) != len(returns):
            print(f"Warning: Mismatch in log_probs ({len(policy.saved_log_probs)}) and returns ({len(returns)})")
            # Truncate to match (shouldn't happen if logic is correct)
            min_len = min(len(policy.saved_log_probs), len(returns))
            policy.saved_log_probs = policy.saved_log_probs[:min_len]
            returns = returns[:min_len]

        for log_prob, R in zip(policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        optimizer.zero_grad()
        if policy_loss:
            policy_loss = torch.stack(policy_loss).sum()
            policy_loss.backward()
            optimizer.step()
            
        # Clear memory
        policy.saved_log_probs = []
        policy.rewards = []
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")

    print("Training finished.")
    
    # Save model
    torch.save(policy.state_dict(), "outputs/rl_policy_marl.pth")

if __name__ == "__main__":
    train()
