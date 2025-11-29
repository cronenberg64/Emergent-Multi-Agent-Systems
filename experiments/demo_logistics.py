import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import random
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.world import World
from src.agents.physical import TruckAgent, WarehouseAgent
from src.agents.abstract import TaskAgent
from src.vis.visualizer import Visualizer
from src.core.communication import MessageType
from src.core.metrics import MetricsLogger

def run_demo():
    world = World()
    
    # Create agents
    w1 = WarehouseAgent()
    w2 = WarehouseAgent()
    w3 = WarehouseAgent()
    
    t1 = TruckAgent()
    t2 = TruckAgent()
    
    # Add agents
    world.add_agent(w1)
    world.add_agent(w2)
    world.add_agent(w3)
    world.add_agent(t1)
    world.add_agent(t2)
    
    # Connect warehouses (Graph topology)
    world.connect_agents(w1.id, w2.id)
    world.connect_agents(w2.id, w3.id)
    world.connect_agents(w3.id, w1.id) # Cycle
    
    # Place trucks
    t1.location = w1.id
    t2.location = w2.id
    world.connect_agents(t1.id, w1.id) # Truck is at W1
    world.connect_agents(t2.id, w2.id) # Truck is at W2
    
    # Add some tasks to warehouses
    task1 = TaskAgent(origin=w1.id, destination=w3.id)
    w1.inventory.append(task1.id)
    world.add_agent(task1) 
    world.connect_agents(task1.id, w1.id)

    vis = Visualizer(world, output_dir="outputs")
    logger = MetricsLogger(filepath="outputs/demo_metrics.csv")
    
    print("Starting simulation...")
    for i in range(20):
        world.tick()
        vis.draw()
        
        # Log metrics
        metrics = {
            "agent_count": len(world.agents),
            "messages_delivered": 0 # Placeholder, would need to track in World
        }
        logger.log(i, metrics)

        # Simulate dynamic behavior
        if i == 5:
            print("Injecting new task at W2")
            task2 = TaskAgent(origin=w2.id, destination=w1.id)
            w2.inventory.append(task2.id)
            world.add_agent(task2)
            world.connect_agents(task2.id, w2.id)
            
        time.sleep(0.1)

    print("Simulation finished.")
    vis.close()
    logger.save()

if __name__ == "__main__":
    run_demo()
