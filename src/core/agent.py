import uuid
from typing import List, Dict, Any, Optional
from src.core.communication import Message, MessageType

from src.learning.policy import Policy, RandomPolicy

class Agent:
    def __init__(self, role: str = "generic", policy: Optional[Policy] = None):
        self.id = str(uuid.uuid4())
        self.role = role
        self.state: Dict[str, Any] = {}
        self.memory: List[Any] = []
        self.inbox: List[Message] = []
        self.outbox: List[Message] = []
        self.policy = policy

    def perceive(self, world_state: Any):
        """
        Update internal state based on world state.
        To be overridden by subclasses.
        """
        pass

    def receive_message(self, message: Message):
        """
        Receive a message from the world/communication channel.
        """
        self.inbox.append(message)

    def decide(self) -> List[Any]:
        """
        Decide on actions to take.
        Returns a list of actions (could be strings, objects, or dicts).
        """
        # Default behavior: process inbox
        self.process_messages()
        
        # Use policy if available
        if self.policy:
            return self.policy.decide(self.state, None) # World view passed as None for now or self.state
        
        return []

    def process_messages(self):
        """
        Process messages in the inbox.
        """
        # Placeholder for message processing logic
        self.inbox.clear()

    def act(self):
        """
        Execute actions.
        This might update internal state or prepare messages to send.
        """
        pass

    def send_message(self, receiver_id: str, msg_type: MessageType, content: Any):
        """
        Queue a message to be sent.
        """
        msg = Message(sender_id=self.id, receiver_id=receiver_id, msg_type=msg_type, content=content)
        self.outbox.append(msg)

    def get_outbox(self) -> List[Message]:
        """
        Retrieve and clear the outbox.
        """
        messages = self.outbox
        self.outbox = []
        return messages

    def __repr__(self):
        return f"Agent(id={self.id[:8]}, role={self.role})"
