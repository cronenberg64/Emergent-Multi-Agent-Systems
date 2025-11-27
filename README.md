# Emergent Multi-Agent Systems

**A research and engineering project exploring continuous-learning, multi-agent ecosystems where complex behavior emerges naturally from simple interacting agents.**

## Overview

This project explores a world-model where all entities—trucks, warehouses, tasks, and markets—are autonomous agents. Unlike traditional simulations where a central controller dictates logic, here the environment itself is composed of interacting agents forming a dynamic graph. Complex system-level behavior (coordination, optimization, resilience) emerges from local interactions and learning.

## Key Features

- **Agent-Centric Architecture**: Every entity is an `Agent` with its own state, memory, and policy.
- **Graph-Based World**: The environment is a dynamic graph of connected agents, allowing for flexible topologies.
- **Emergent Coordination**: Agents communicate via a structured message protocol (`ASK`, `BID`, `NEGOTIATE`) to solve tasks.
- **Pluggable Policies**: Support for Random, Rule-Based, and Reinforcement Learning (PPO/A2C) policies.
- **Visualization**: Real-time visualization of the agent graph and interactions.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Demo
Run the logistics demo to see a simple world with Warehouses and Trucks:

```bash
python experiments/demo_logistics.py
```

This will:
1. Initialize a world with 3 Warehouses and 2 Trucks.
2. Run a simulation for 20 ticks.
3. Generate visualization frames in `outputs/`.
4. Log metrics to `outputs/demo_metrics.csv`.

### Visualization
Check the `outputs/` directory for `.png` frames of the simulation.

## Project Structure

```
src/
  core/          # Base Agent, World, Communication, Metrics
  agents/        # Specific agent implementations (Truck, Warehouse, Task)
  simulation/    # Gym-compatible environment wrapper
  learning/      # Policy interfaces (RL hooks)
  vis/           # Visualization tools
experiments/     # Demo scripts and experiments
tests/           # Unit tests
```

## Roadmap

- [x] Core Engine & Communication
- [x] Basic Agent Types (Physical & Abstract)
- [x] Visualization & Metrics
- [ ] Deep RL Integration (PPO Training)
- [ ] LLM-based Negotiation
- [ ] Large-scale Physics Simulation (Isaac Sim integration)

## License
MIT
