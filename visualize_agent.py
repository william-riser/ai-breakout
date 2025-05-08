import ale_py
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import argparse
import sys
import os

# Add the project root to the Python path to enable imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ram_state_representation.dqn_agent import DQNAgent

def visualize_q_values(agent, state):
    """Visualize Q-values for a given state"""
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(agent.device)
    q_values = agent.qnetwork_local(state_tensor).detach().cpu().numpy()[0]
    
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    
    plt.figure(figsize=(8, 4))
    bars = plt.bar(action_names, q_values)
    
    # Highlight the best action
    best_action = np.argmax(q_values)
    bars[best_action].set_color('red')
    
    plt.title('Q-Values by Action')
    plt.ylabel('Q-Value')
    plt.xlabel('Action')
    plt.ylim([min(q_values) - 0.1, max(q_values) + 0.1])
    
    for i, v in enumerate(q_values):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    return plt.gcf()

def visualize_agent_play(model_path, episodes=1, delay=0.01, save_video=False, track_q_values=True):
    """
    Visualize the agent playing Breakout and track its decision-making process.
    
    Args:
        model_path (str): Path to the saved model weights
        episodes (int): Number of episodes to visualize
        delay (float): Delay between frames for better visualization
        save_video (bool): Whether to save a video of the agent playing
        track_q_values (bool): Whether to track and plot Q-values
    """
    # Create environment with human rendering
    env = gym.make('ALE/Breakout-ram-v5', render_mode='human')
    
    # Initialize agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # Load trained model
    try:
        agent.load(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Action names for human-readable output
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    
    # Setup for Q-value tracking if enabled
    if track_q_values:
        q_value_history = []
        state_history = []
        action_history = []
        reward_history = []
    
    # Play episodes
    for episode in range(1, episodes + 1):
        print(f"\nStarting visualization episode {episode}/{episodes}")
        state, info = env.reset()
        episode_reward = 0
        step = 0
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            # Get Q-values and select action
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(agent.device)
            q_values = agent.qnetwork_local(state_tensor).detach().cpu().numpy()[0]
            action = agent.act(state, eps=0.0)
            
            # Display info about the current step
            print(f"Step {step}: Action={action_names[action]} (Q-values: {q_values})")
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Track data if Q-value tracking is enabled
            if track_q_values:
                q_value_history.append(q_values)
                state_history.append(state)
                action_history.append(action)
                reward_history.append(reward)
            
            # Update state
            state = next_state
            step += 1
            
            # Delay for better visualization
            time.sleep(delay)
        
        print(f"Episode {episode} finished with total reward: {episode_reward}")
    
    # Analyze Q-values if tracking was enabled
    if track_q_values and q_value_history:
        analyze_q_values(q_value_history, action_history, reward_history)
    
    env.close()

def analyze_q_values(q_values, actions, rewards):
    """
    Analyze the Q-values recorded during gameplay.
    
    Args:
        q_values (list): List of Q-value arrays for each step
        actions (list): List of actions taken at each step
        rewards (list): List of rewards received at each step
    """
    # Convert lists to numpy arrays for easier analysis
    q_values = np.array(q_values)
    actions = np.array(actions)
    rewards = np.array(rewards)
    
    # Plot Q-value evolution over time
    plt.figure(figsize=(12, 8))
    
    # Plot Q-values for each action
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    for action_idx in range(q_values.shape[1]):
        plt.plot(q_values[:, action_idx], label=f'Action: {action_names[action_idx]}')
    
    # Mark points where rewards were received
    reward_steps = np.where(rewards > 0)[0]
    if len(reward_steps) > 0:
        for step in reward_steps:
            plt.axvline(x=step, color='r', linestyle='--', alpha=0.3)
    
    plt.title('Q-Value Evolution During Gameplay')
    plt.xlabel('Step')
    plt.ylabel('Q-Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('q_value_evolution.png')
    
    # Plot action distribution
    plt.figure(figsize=(10, 6))
    unique_actions, action_counts = np.unique(actions, return_counts=True)
    action_labels = [action_names[int(a)] for a in unique_actions]
    
    plt.bar(action_labels, action_counts)
    plt.title('Action Distribution')
    plt.xlabel('Action')
    plt.ylabel('Count')
    for i, count in enumerate(action_counts):
        plt.text(i, count + 1, str(count), ha='center')
    plt.savefig('action_distribution.png')
    
    # Calculate and display statistics
    print("\nQ-Value Analysis:")
    print(f"Total steps: {len(actions)}")
    print(f"Total rewards: {sum(rewards)}")
    
    print("\nAction Distribution:")
    for action_idx, count in zip(unique_actions, action_counts):
        action_name = action_names[int(action_idx)]
        percentage = (count / len(actions)) * 100
        print(f"{action_name}: {count} times ({percentage:.1f}%)")
    
    print("\nAverage Q-values by action:")
    for action_idx in range(q_values.shape[1]):
        mean_q = np.mean(q_values[:, action_idx])
        std_q = np.std(q_values[:, action_idx])
        print(f"{action_names[action_idx]}: Mean={mean_q:.4f}, Std={std_q:.4f}")
    
    # Confidence analysis: how confident was the agent in its chosen actions?
    chosen_q_values = np.array([q_values[i, actions[i]] for i in range(len(actions))])
    confidence_scores = chosen_q_values - np.mean(q_values, axis=1)
    
    print("\nDecision Confidence Analysis:")
    print(f"Average confidence in chosen actions: {np.mean(confidence_scores):.4f}")
    print(f"Min confidence: {np.min(confidence_scores):.4f}")
    print(f"Max confidence: {np.max(confidence_scores):.4f}")
    
    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(confidence_scores, bins=20, alpha=0.7)
    plt.axvline(np.mean(confidence_scores), color='r', linestyle='dashed', linewidth=2, 
                label=f'Mean: {np.mean(confidence_scores):.4f}')
    plt.title('Agent Decision Confidence Distribution')
    plt.xlabel('Confidence Score (Q_chosen - Q_avg)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('decision_confidence.png')
    
    print("\nAnalysis results saved as image files.")

def analyze_ram_importance(model_path, samples=1000):
    """
    Analyze which RAM positions are most important for the agent's decisions.
    
    Args:
        model_path (str): Path to the saved model weights
        samples (int): Number of environment samples to collect for analysis
    """
    # Create environment without rendering
    env = gym.make('ALE/Breakout-ram-v5', render_mode=None)
    
    # Initialize agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # Load trained model
    try:
        agent.load(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Collect state samples
    print(f"Collecting {samples} state samples for RAM importance analysis...")
    states = []
    
    state, _ = env.reset()
    for _ in range(samples):
        states.append(state.copy())
        action = agent.act(state, eps=0.1)  # Use some exploration to get diverse states
        next_state, _, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state
    
    states = np.array(states)
    
    # Calculate importance of each RAM position based on weight magnitudes
    # For each RAM value, we perturb it and see how much that affects Q-values
    print("Analyzing RAM position importance...")
    importance_scores = np.zeros(state_size)
    
    for i in range(state_size):
        # Sample a subset of states for efficiency
        sample_indices = np.random.choice(len(states), min(100, len(states)), replace=False)
        sample_states = states[sample_indices]
        
        # Calculate baseline Q-values
        baseline_q = []
        for state in sample_states:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(agent.device)
            q_values = agent.qnetwork_local(state_tensor).detach().cpu().numpy()[0]
            baseline_q.append(q_values)
        baseline_q = np.array(baseline_q)
        
        # Perturb the i-th RAM position and recalculate Q-values
        perturbed_states = sample_states.copy()
        perturbed_states[:, i] = (perturbed_states[:, i] + 10) % 256  # Add 10, wrap around 0-255
        
        perturbed_q = []
        for state in perturbed_states:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(agent.device)
            q_values = agent.qnetwork_local(state_tensor).detach().cpu().numpy()[0]
            perturbed_q.append(q_values)
        perturbed_q = np.array(perturbed_q)
        
        # Calculate how much the perturbation affected Q-values
        q_diff = np.abs(perturbed_q - baseline_q)
        importance_scores[i] = np.mean(q_diff)
    
    # Normalize importance scores
    importance_scores = importance_scores / np.max(importance_scores)
    
    # Plot importance scores
    plt.figure(figsize=(15, 6))
    plt.bar(range(state_size), importance_scores)
    plt.title('RAM Position Importance for Agent Decisions')
    plt.xlabel('RAM Position')
    plt.ylabel('Relative Importance')
    plt.grid(True, alpha=0.3)
    
    # Highlight the most important positions
    top_positions = np.argsort(importance_scores)[-10:]
    for pos in top_positions:
        plt.text(pos, importance_scores[pos] + 0.02, str(pos), ha='center', fontweight='bold')
    
    plt.savefig('ram_importance.png')
    
    # Print the most important RAM positions
    print("\nTop 10 most important RAM positions:")
    for i, pos in enumerate(top_positions[::-1]):
        print(f"{i+1}. Position {pos}: {importance_scores[pos]:.4f}")
    
    env.close()
    print("RAM importance analysis complete. Results saved to ram_importance.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize and analyze trained DQN agent on Breakout')
    parser.add_argument('--model', type=str, default='ram_state_representation/dqn_breakout_ram_final.pth',
                      help='Path to the saved model weights')
    parser.add_argument('--episodes', type=int, default=1, 
                      help='Number of episodes to visualize')
    parser.add_argument('--delay', type=float, default=0.01,
                      help='Delay between frames when visualizing (seconds)')
    parser.add_argument('--no_q_tracking', action='store_true',
                      help='Disable Q-value tracking and analysis')
    parser.add_argument('--analyze_ram', action='store_true',
                      help='Perform RAM importance analysis')
    parser.add_argument('--ram_samples', type=int, default=1000,
                      help='Number of samples to use for RAM importance analysis')
    
    args = parser.parse_args()
    
    if args.analyze_ram:
        analyze_ram_importance(args.model, samples=args.ram_samples)
    else:
        visualize_agent_play(args.model, episodes=args.episodes, 
                            delay=args.delay, track_q_values=not args.no_q_tracking) 