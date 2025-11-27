from typing import List, Any, Optional
from src.core.agent import Agent
from src.core.communication import Message, MessageType
from src.core.world import World

class TruckAgent(Agent):
    def __init__(self, capacity: int = 1):
        super().__init__(role="truck")
        self.capacity = capacity
        self.cargo: List[str] = [] # List of Task IDs
        self.location: Optional[str] = None # Current node ID
        self.destination: Optional[str] = None

    def perceive(self, world: World):
        # Simple perception: know current location and connected nodes
        if self.location:
            self.state["location"] = self.location
            self.state["neighbors"] = list(world.graph.neighbors(self.location))

    def decide(self) -> List[Any]:
        super().decide() # Process messages
        
        actions = []
        # Basic logic: if has cargo, move to destination. If empty, look for tasks.
        if self.cargo:
            # Move towards destination (simplified: random neighbor for now if no pathfinding)
            # In real impl, use nx.shortest_path
            pass
        else:
            # Ask current location for tasks
            if self.location:
                self.send_message(self.location, MessageType.ASK, "Any tasks?")
        
        return actions

    def act(self):
        pass

class WarehouseAgent(Agent):
    def __init__(self):
        super().__init__(role="warehouse")
        self.inventory: List[str] = [] # Task IDs waiting here

    def perceive(self, world: World):
        pass

    def decide(self) -> List[Any]:
        super().decide()
        return []

    def process_messages(self):
        for msg in self.inbox:
            if msg.msg_type == MessageType.ASK and msg.content == "Any tasks?":
                if self.inventory:
                    task_id = self.inventory[0]
                    self.send_message(msg.sender_id, MessageType.BID, task_id)
        self.inbox.clear()

    def act(self):
        pass
