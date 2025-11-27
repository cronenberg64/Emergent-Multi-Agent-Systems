import unittest
from src.core.agent import Agent
from src.core.world import World
from src.core.communication import MessageType

class TestCoreEngine(unittest.TestCase):
    def test_agent_creation(self):
        agent = Agent(role="tester")
        self.assertIsNotNone(agent.id)
        self.assertEqual(agent.role, "tester")

    def test_world_tick(self):
        world = World()
        agent1 = Agent(role="sender")
        agent2 = Agent(role="receiver")
        
        world.add_agent(agent1)
        world.add_agent(agent2)
        world.connect_agents(agent1.id, agent2.id)
        
        # Agent 1 sends message to Agent 2
        agent1.send_message(agent2.id, MessageType.INFO, "Hello")
        
        # Tick 1: Message is sent to channel (collected from outbox)
        world.tick()
        
        # Tick 2: Message is delivered to Agent 2
        # (Note: In current implementation, deliver() is called at start of tick, 
        # so messages sent in Tick 1 are delivered in Tick 2)
        world.tick()
        
        self.assertEqual(len(agent2.inbox), 1)
        self.assertEqual(agent2.inbox[0].content, "Hello")
        self.assertEqual(agent2.inbox[0].sender_id, agent1.id)

if __name__ == '__main__':
    unittest.main()
