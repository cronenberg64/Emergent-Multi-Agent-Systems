# System Architecture: Emergent Multi-Agent Systems

## Overview
This system models a world where every entity is an agent. The environment is not a static grid but a dynamic graph of interacting agents.

## Core Components

### 1. Agent (`src/core/agent.py`)
The fundamental building block.
- **Identity**: Unique ID and Role.
- **State**: Internal dictionary of attributes (location, cargo, etc.).
- **Policy**: Pluggable decision-making module (`Random`, `RuleBased`, `RL`).
- **Communication**: Inbox/Outbox for `Message` objects.

### 2. World (`src/core/world.py`)
The container and scheduler.
- **Graph**: NetworkX graph representing physical/logical connections.
- **Tick Loop**:
    1. Deliver messages.
    2. Agents perceive and decide (shuffled order).
    3. Resolve actions.
- **Communication Channel**: Handles message routing.

### 3. Simulation Environment (`src/simulation/environment.py`)
A Gym-compatible wrapper around the World.
- Allows standard RL training loops (e.g., Stable Baselines3, RLLib).
- Maps global actions to agent-specific actions.

### 4. Visualization (`src/vis/visualizer.py`)
- Real-time (or post-hoc) visualization of the agent graph.
- Color-coded agents (Trucks=Blue, Warehouses=Red).

## Design Choices

### Graph vs. Grid
We chose a **Graph** representation to allow for flexible topology (e.g., road networks, supply chains) without the sparsity of a large grid. Agents move between nodes (other agents).

### Agent-Centric Environment
Unlike standard RL where Environment >> Agent, here the Environment is just a collection of Agents. This supports "Infinite Agent" scalability and emergent behavior where the environment dynamics are defined by agent interactions.

### Communication
Explicit message passing allows for negotiation and coordination protocols to emerge or be scripted.
