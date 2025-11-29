import matplotlib.pyplot as plt
import networkx as nx
from src.core.world import World

import os

class Visualizer:
    def __init__(self, world: World, output_dir: str = "."):
        self.world = world
        self.output_dir = output_dir
        self.fig, self.ax = plt.subplots()
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def draw(self):
        self.ax.clear()
        pos = nx.spring_layout(self.world.graph, seed=42)
        
        # Draw nodes
        colors = []
        labels = {}
        for node in self.world.graph.nodes():
            agent = self.world.agents.get(node)
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
        
        nx.draw(self.world.graph, pos, ax=self.ax, node_color=colors, with_labels=True, labels=labels)
        self.ax.set_title(f"Tick: {self.world.tick_count}")
        
        filepath = os.path.join(self.output_dir, f"vis_tick_{self.world.tick_count:03d}.png")
        self.fig.savefig(filepath)

    def close(self):
        plt.close(self.fig)
