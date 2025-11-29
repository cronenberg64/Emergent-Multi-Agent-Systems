import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import random
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation.environment import LogisticsEnv
from src.vis.visualizer import Visualizer
from src.core.metrics import MetricsLogger

def run_complex_demo():
    # Initialize environment with 3 trucks and 5 warehouses
    env = LogisticsEnv(num_trucks=3, num_warehouses=5)
    env.reset()
    
    vis = Visualizer(env.world, output_dir="outputs")
    logger = MetricsLogger(filepath="outputs/complex_metrics.csv")
    
    print("Starting complex simulation...")
    
    # Run for 50 ticks
    for i in range(50):
        # Simple Heuristic Policy:
        # If truck has cargo, move to destination (if neighbor) or random neighbor
        # If truck is empty, move to random neighbor
        
        actions = []
        trucks = [a for a in env.world.agents.values() if a.role == "truck"]
        trucks.sort(key=lambda x: x.id)
        
        for truck in trucks:
            neighbors = list(env.world.graph.neighbors(truck.location))
            if not neighbors:
                actions.append(0) # Stay
                continue
                
            # Random walk for now
            # Action 1 corresponds to neighbors[0], Action 2 to neighbors[1], etc.
            choice = random.randint(0, len(neighbors) - 1)
            actions.append(choice + 1)
            
        # Step environment
        obs, reward, terminated, truncated, info = env.step(actions)
        
        # Visualize
        vis.draw()
        
        # Log metrics
        metrics = {
            "tick": i,
            "truck_count": len(trucks),
            "total_agents": len(env.world.agents),
            "reward": reward
        }
        logger.log(i, metrics)
        
        if i % 10 == 0:
            print(f"Tick {i}: {len(env.world.agents)} agents active.")

    print("Complex simulation finished.")
    vis.close()
    logger.save()

if __name__ == "__main__":
    run_complex_demo()
