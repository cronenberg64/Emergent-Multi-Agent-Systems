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
    # Use 1 truck for simplified training demo
    env = LogisticsEnv(num_trucks=1, num_warehouses=5)
    
    # Hyperparameters
    # Obs: 2 features * 1 truck + 5 warehouses = 7
    input_dim = 7 
    output_dim = 5 # 5 discrete actions
    learning_rate = 0.01
    episodes = 100
    gamma = 0.99

    policy = RLPolicy(input_dim=input_dim, output_dim=output_dim)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    
    print("Starting training...")
    
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        # Store trajectory
        log_probs = []
        rewards = []
        
        while not done:
            # Prepare state for policy
            # Policy expects a dict with 'vector' or similar, but our env returns np array
            # Let's adapt here or in policy.
            # Policy.decide takes (agent_state, world_view).
            # But here we are training a centralized policy or shared policy?
            # Let's treat 'obs' as the state vector.
            
            state_dict = {'vector': obs}
            
            # Policy forward pass
            # We need to access the internal network directly or modify decide to return log_probs
            # RLPolicy.decide returns [action_item] and stores log_prob internally
            
            action_list = policy.decide(state_dict, None)
            action = action_list[0]
            
            # Step env
            # Env expects list of actions for all trucks. We have 1 truck.
            next_obs, reward, terminated, truncated, _ = env.step([action])
            done = terminated or truncated
            
            rewards.append(reward)
            episode_reward += reward
            obs = next_obs
            
        # Update Policy (REINFORCE)
        R = 0
        policy_loss = []
        returns = []
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
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
    torch.save(policy.state_dict(), "outputs/rl_policy.pth")

if __name__ == "__main__":
    train()
