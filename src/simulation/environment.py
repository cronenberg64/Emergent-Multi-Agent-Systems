import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.core.world import World
from src.agents.physical import TruckAgent, WarehouseAgent
from src.agents.abstract import TaskAgent

class LogisticsEnv(gym.Env):
    def __init__(self, num_trucks: int = 3, num_warehouses: int = 5):
        self.num_trucks = num_trucks
        self.num_warehouses = num_warehouses
        self.world = World()
        # Action space: 5 actions per truck. MultiDiscrete? 
        # For simplicity in this custom loop, we'll just expect a list of ints.
        self.action_space = spaces.MultiDiscrete([5] * num_trucks) 
        # Observation space: Flattened vector. 
        # 2 features per truck + 1 feature per warehouse (task count)
        obs_dim = (2 * num_trucks) + num_warehouses
        self.observation_space = spaces.Box(low=0, high=100, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world = World()
        
        # Create Warehouses
        warehouses = []
        for i in range(self.num_warehouses):
            w = WarehouseAgent()
            self.world.add_agent(w)
            warehouses.append(w)
            
        # Connect Warehouses (Simple Cycle + Random shortcuts)
        for i in range(len(warehouses)):
            self.world.connect_agents(warehouses[i].id, warehouses[(i+1)%len(warehouses)].id)
            
        # Create Trucks
        for i in range(self.num_trucks):
            t = TruckAgent()
            # Start at random warehouse
            start_w = np.random.choice(warehouses)
            t.location = start_w.id
            self.world.add_agent(t)
            self.world.connect_agents(t.id, start_w.id)
            
        return self._get_obs(), {}

    def step(self, actions):
        # Reward: -0.01 per tick (fuel)
        reward = -0.01

        # actions: List of integers, one per TruckAgent
        trucks = [a for a in self.world.agents.values() if a.role == "truck"]
        
        # Apply actions
        for i, truck in enumerate(trucks):
            if i < len(actions):
                action = actions[i]
                if action > 0:
                    neighbors = list(self.world.graph.neighbors(truck.location))
                    if neighbors and (action - 1) < len(neighbors):
                        # Move truck
                        # Disconnect from old
                        old_loc = truck.location
                        truck.location = neighbors[action - 1]
                        # Update graph connections (remove old edge, add new edge)
                        # In this simplified model, we assume truck is "at" a node.
                        # We might not need to strictly remove/add edges if we just track location property,
                        # but for graph vis it helps.
                        if self.world.graph.has_edge(truck.id, old_loc):
                            self.world.graph.remove_edge(truck.id, old_loc)
                        self.world.connect_agents(truck.id, truck.location)
                        
                        # Pickup Logic
                        # If truck is at a warehouse and warehouse has tasks, pick one up
                        # Simplified: Truck capacity 1
                        if not truck.cargo:
                            # Find warehouse at this location
                            # In this model, location is the ID of the warehouse agent
                            warehouse = self.world.agents.get(truck.location)
                            if warehouse and warehouse.role == "warehouse" and warehouse.inventory:
                                task_id = warehouse.inventory.pop(0)
                                truck.cargo.append(task_id)
                                # Also move task agent to truck location (conceptually)
                                # In graph, maybe connect task to truck?
                                # For now, just tracking ID is enough for logic.
                                
                        # Delivery Logic
                        # If truck has cargo, check if current location is destination
                        elif truck.cargo:
                            task_id = truck.cargo[0]
                            task = self.world.agents.get(task_id)
                            # Task might not be in world agents dict if it's just data, 
                            # but in our impl we added it.
                            # However, if it was removed or something, handle it.
                            if task and hasattr(task, 'destination') and task.destination == truck.location:
                                # Delivered!
                                truck.cargo.pop(0)
                                # Remove task agent from world? Or mark completed?
                                self.world.remove_agent(task_id)
                                reward += 10.0 # Big reward for delivery
        if np.random.random() < 0.1:
            warehouses = [a for a in self.world.agents.values() if a.role == "warehouse"]
            if len(warehouses) >= 2:
                origin, dest = np.random.choice(warehouses, 2, replace=False)
                task = TaskAgent(origin=origin.id, destination=dest.id)
                origin.inventory.append(task.id)
                self.world.add_agent(task)
                self.world.connect_agents(task.id, origin.id)

        self.world.tick()
        
        obs = self._get_obs()
        
        
        terminated = False
        truncated = False
        if self.world.tick_count >= 200:
            truncated = True
            
        info = {}
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # 2 features per truck + 1 feature per warehouse
        obs = []
        
        trucks = [a for a in self.world.agents.values() if a.role == "truck"]
        # Sort by ID to ensure consistency? Or just iteration order (which might vary)
        # Better to sort.
        trucks.sort(key=lambda x: x.id)
        
        for truck in trucks:
            obs.append(1.0 if truck.cargo else 0.0)
            obs.append(hash(truck.location) % 100 / 100.0)
            
        warehouses = [a for a in self.world.agents.values() if a.role == "warehouse"]
        warehouses.sort(key=lambda x: x.id)
        
        for w in warehouses:
            obs.append(len(w.inventory))
            
        return np.array(obs, dtype=np.float32)

    def render(self):
        print(f"Tick: {self.world.tick_count}, Agents: {len(self.world.agents)}")
