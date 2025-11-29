import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.simulation.environment import LogisticsEnv
from src.core.world import World

st.set_page_config(page_title="Emergent MAS Dashboard", layout="wide")

st.title("Autonomous World Engine Dashboard")

# Sidebar controls
st.sidebar.header("Simulation Config")
num_trucks = st.sidebar.slider("Number of Trucks", 1, 10, 3)
num_warehouses = st.sidebar.slider("Number of Warehouses", 2, 10, 5)
run_sim = st.sidebar.button("Run Simulation")

# Placeholders for live updates
col1, col2 = st.columns([2, 1])
with col1:
    graph_placeholder = st.empty()
with col2:
    metrics_placeholder = st.empty()

if run_sim:
    env = LogisticsEnv(num_trucks=num_trucks, num_warehouses=num_warehouses)
    env.reset()
    
    # Run for 50 ticks
    for i in range(50):
        # Random actions for now
        actions = [np.random.randint(0, 5) for _ in range(num_trucks)]
        env.step(actions)
        
        # Draw Graph
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(env.world.graph, seed=42)
        
        colors = []
        labels = {}
        for node in env.world.graph.nodes():
            agent = env.world.agents.get(node)
            if agent:
                labels[node] = f"{agent.role[:1]}.{agent.id[:4]}"
                if agent.role == "truck":
                    colors.append("blue")
                elif agent.role == "warehouse":
                    colors.append("red")
                elif agent.role == "task":
                    colors.append("green")
                else:
                    colors.append("gray")
            else:
                colors.append("black")
                labels[node] = "?"
                
        nx.draw(env.world.graph, pos, ax=ax, node_color=colors, with_labels=True, labels=labels)
        ax.set_title(f"Tick: {env.world.tick_count}")
        
        with col1:
            graph_placeholder.pyplot(fig)
        plt.close(fig)
        
        # Update Metrics
        with col2:
            metrics_placeholder.metric("Tick", env.world.tick_count)
            metrics_placeholder.metric("Active Agents", len(env.world.agents))
            
        time.sleep(0.1)
