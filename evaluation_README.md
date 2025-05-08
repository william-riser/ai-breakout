# DQN Agent Evaluation Tools

This directory contains tools for evaluating and analyzing the performance of the DQN agent trained on the Atari Breakout game using RAM state representation.

## Files

- `evaluate.py`: Main evaluation script for measuring agent performance
- `visualize_agent.py`: Tool for visualizing agent behavior and analyzing decision making
- `evaluation_README.md`: This documentation file

## Requirements

Make sure you have all dependencies installed from the project's `requirements.txt` file.

## Evaluation Script

The `evaluate.py` script provides quantitative evaluation of a trained agent over multiple episodes.

### Usage

```bash
# Basic evaluation of the final model (100 episodes)
python evaluate.py

# Evaluate a specific model
python evaluate.py --model ram_state_representation/dqn_breakout_ram_episode_10000.pth

# Run 50 evaluation episodes and render the gameplay
python evaluate.py --episodes 50 --render

# Compare multiple model checkpoints
python evaluate.py --compare --models ram_state_representation/dqn_breakout_ram_episode_10000.pth ram_state_representation/dqn_breakout_ram_episode_20000.pth ram_state_representation/dqn_breakout_ram_final.pth
```

### Arguments

- `--model`: Path to the model weights file (default: 'ram_state_representation/dqn_breakout_ram_final.pth')
- `--episodes`: Number of episodes to evaluate (default: 100)
- `--render`: Enable rendering of the environment during evaluation
- `--delay`: Delay between frames when rendering (default: 0.01 seconds)
- `--compare`: Compare multiple models
- `--models`: List of model paths to compare (use with --compare)

### Output

- `evaluation_results.txt`: Detailed performance statistics
- `evaluation_score_distribution.png`: Histogram of scores achieved during evaluation
- `model_comparison.png`: Box plot comparing performance across different models (when using --compare)
- `model_comparison_results.txt`: Detailed comparison statistics (when using --compare)

## Visualization Script

The `visualize_agent.py` script provides tools for visualizing the agent's behavior and analyzing its decision-making process.

### Usage

```bash
# Visualize agent playing one episode with default model
python visualize_agent.py

# Visualize multiple episodes with a slower delay between frames
python visualize_agent.py --episodes 3 --delay 0.05

# Analyze RAM position importance for the agent's decisions
python visualize_agent.py --analyze_ram
```

### Arguments

- `--model`: Path to the model weights file (default: 'ram_state_representation/dqn_breakout_ram_final.pth')
- `--episodes`: Number of episodes to visualize (default: 1)
- `--delay`: Delay between frames for better visualization (default: 0.01 seconds)
- `--no_q_tracking`: Disable Q-value tracking and analysis
- `--analyze_ram`: Perform RAM importance analysis
- `--ram_samples`: Number of samples to use for RAM importance analysis (default: 1000)

### Output

When visualizing agent play:
- Real-time rendering of the game
- Real-time display of Q-values and actions
- `q_value_evolution.png`: Plot of Q-values over time
- `action_distribution.png`: Distribution of actions taken
- `decision_confidence.png`: Agent's confidence in chosen actions

When analyzing RAM importance:
- `ram_importance.png`: Visualization of which RAM positions most influence agent decisions

## Examples

### Basic Evaluation
To run a basic evaluation of the trained agent:

```bash
python evaluate.py
```

This will evaluate the default final model over 100 episodes and generate evaluation metrics.

### Model Comparison
To compare performance across training checkpoints:

```bash
python evaluate.py --compare
```

This will compare three default checkpoints (10000, 20000, and final) and generate comparison plots.

### Agent Visualization with Analysis
To visualize the agent playing and analyze its decision process:

```bash
python visualize_agent.py --delay 0.05
```

This will render one episode with a slight delay between frames and generate Q-value analysis.

### RAM Importance Analysis
To understand which RAM positions are most important for the agent's decisions:

```bash
python visualize_agent.py --analyze_ram
```

This will analyze which RAM positions have the most influence on the agent's Q-values. 