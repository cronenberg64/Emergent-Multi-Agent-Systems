import networkx as nx
import random
from typing import Dict, List
from src.core.agent import Agent
from src.core.communication import CommunicationChannel, Message

class World:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.graph = nx.Graph()
        self.comm_channel = CommunicationChannel()
        self.tick_count = 0

    def add_agent(self, agent: Agent):
        """
        Add an agent to the world.
        """
        self.agents[agent.id] = agent
        self.graph.add_node(agent.id, agent=agent)

    def remove_agent(self, agent_id: str):
        """
        Remove an agent from the world.
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.graph.remove_node(agent_id)

    def connect_agents(self, agent1_id: str, agent2_id: str, weight: float = 1.0):
        """
        Create a connection (edge) between two agents.
        """
        if agent1_id in self.agents and agent2_id in self.agents:
            self.graph.add_edge(agent1_id, agent2_id, weight=weight)

    def tick(self):
        """
        Advance the world by one step.
        """
        self.tick_count += 1
        
        # 1. Deliver messages
        messages = self.comm_channel.deliver()
        for msg in messages:
            if msg.receiver_id in self.agents:
                self.agents[msg.receiver_id].receive_message(msg)

        # 2. Agent Perception & Decision
        # Shuffle execution order to prevent bias
        agent_ids = list(self.agents.keys())
        random.shuffle(agent_ids)

        actions = {}
        for agent_id in agent_ids:
            agent = self.agents[agent_id]
            # In a real scenario, we'd pass a filtered view of the world state
            agent.perceive(self) 
            agent_actions = agent.decide()
            actions[agent_id] = agent_actions
            
            # Collect outgoing messages
            for msg in agent.get_outbox():
                msg.timestamp = self.tick_count
                self.comm_channel.send(msg)

        # 3. Resolve Actions (Placeholder)
        # Here we would handle physical interactions, conflicts, etc.
        for agent_id, agent_actions in actions.items():
            self.agents[agent_id].act()

    def get_state(self):
        """
        Return a summary of the world state.
        """
        return {
            "tick": self.tick_count,
            "agent_count": len(self.agents),
            "edges": self.graph.number_of_edges()
        }
