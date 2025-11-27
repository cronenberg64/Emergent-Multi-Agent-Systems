from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, List
import uuid

class MessageType(Enum):
    INFO = auto()
    ASK = auto()
    BID = auto()
    REQUEST = auto()
    NEGOTIATE = auto()
    ACCEPT = auto()
    REJECT = auto()

@dataclass
class Message:
    sender_id: str
    receiver_id: str
    msg_type: MessageType
    content: Any
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: int = 0

class CommunicationChannel:
    """
    Simulates the medium through which agents communicate.
    Can be extended to include noise, delay, or cost.
    """
    def __init__(self):
        self.message_queue: List[Message] = []

    def send(self, message: Message):
        """
        Send a message. In this simple version, it's instant delivery to the queue.
        """
        self.message_queue.append(message)

    def deliver(self) -> List[Message]:
        """
        Retrieve all messages for the current tick.
        """
        messages = self.message_queue
        self.message_queue = []
        return messages
