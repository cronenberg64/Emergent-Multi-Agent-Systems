import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation.environment import LogisticsEnv
from src.learning.rl_policy import RLPolicy
from src.vis.visualizer import Visualizer
from src.core.metrics import MetricsLogger

def run_trained_demo():
    # Initialize environment with 1 truck and 5 warehouses (same as training)
    env = LogisticsEnv(num_trucks=1, num_warehouses=5)
    obs, _ = env.reset()
    
    # Initialize Policy
    input_dim = 7
    output_dim = 5
    policy = RLPolicy(input_dim=input_dim, output_dim=output_dim)
    
    # Load trained weights
    model_path = "outputs/rl_policy.pth"
    if os.path.exists(model_path):
        policy.load_state_dict(torch.load(model_path))
        print(f"Loaded trained model from {model_path}")
    else:
        print(f"Model not found at {model_path}. Running with random weights.")

    vis = Visualizer(env.world, output_dir="outputs")
    logger = MetricsLogger(filepath="outputs/trained_metrics.csv")
    
    print("Starting trained agent simulation...")
    
    total_reward = 0
    
    # Run for 50 ticks
    for i in range(50):
        # Prepare state
        state_dict = {'vector': obs}
        
        # Get action from policy (deterministic or stochastic? decide samples stochastically)
        # For demo, stochastic is fine, or we could modify to take argmax.
        # Let's stick to the policy's decide method.
        action_list = policy.decide(state_dict, None)
        action = action_list[0]
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step([action])
        
        total_reward += reward
        obs = next_obs
        
        # Visualize
        vis.draw()
        
        # Find the truck agent for logging
        truck = next((a for a in env.world.agents.values() if a.role == "truck"), None)
        cargo_status = 1 if truck and truck.cargo else 0
        
        # Log metrics
        metrics = {
            "tick": i,
            "reward": reward,
            "total_reward": total_reward,
            "cargo": cargo_status
        }
        logger.log(i, metrics)
        
        if i % 10 == 0:
            print(f"Tick {i}: Reward {reward:.2f}, Total: {total_reward:.2f}")

    print(f"Simulation finished. Total Reward: {total_reward:.2f}")
    vis.close()
    logger.save()

if __name__ == "__main__":
    run_trained_demo()
