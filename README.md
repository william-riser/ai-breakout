# AI Breakout

A Deep Q-Learning (DQN) implementation for the classic Atari Breakout game, using RAM state representation and advanced visualization tools.

## Overview

This project implements a Deep Q-Network (DQN) agent that learns to play the Atari Breakout game using RAM state representation. The implementation includes comprehensive visualization and analysis tools to understand the agent's decision-making process. For the full project details, reference the [Final Paper](/FinalPaper.pdf)

## Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- ALE-py
- Matplotlib
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-breakout.git
cd ai-breakout
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Atari ROMs:
```bash
python -m ale_py.roms.install
```

## Project Structure

- `visualize_agent.py`: Main visualization and analysis script
- `ram_state_representation/`: Contains the DQN agent implementation
- `evaluate.py`: Evaluation script for agent performance
- `test_environment.py`: Environment testing utilities

## Results

The project generates several visualization files:
- `q_value_evolution.png`: Shows how Q-values change over time
- `action_distribution.png`: Displays the distribution of actions taken
- `decision_confidence.png`: Visualizes agent's decision confidence
- `ram_importance.png`: Shows the importance of different RAM positions


