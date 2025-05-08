import ale_py
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import sys
import os

# Add the project root to the Python path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ram_state_representation.dqn_agent import DQNAgent

def evaluate_agent(model_path, episodes=100, render=False, delay=0.0, save_results=True):
    """
    Evaluate a trained DQN agent on the Breakout game.
    
    Args:
        model_path (str): Path to the saved model weights
        episodes (int): Number of episodes to evaluate
        render (bool): Whether to render the game
        delay (float): Delay between frames when rendering (for better visualization)
        save_results (bool): Whether to save the evaluation results to a file
    
    Returns:
        dict: Dictionary with evaluation metrics
    """
    # Create environment
    render_mode = 'human' if render else None
    env = gym.make('ALE/Breakout-ram-v5', render_mode=render_mode)
    
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
        return None
    
    # Evaluation loop
    scores = []
    max_score = 0
    min_score = float('inf')
    total_steps = 0
    lives_data = []
    
    print(f"\nEvaluating agent over {episodes} episodes...")
    for i in range(1, episodes + 1):
        state, info = env.reset()
        score = 0
        terminated = False
        truncated = False
        steps = 0
        if 'lives' in info:
            initial_lives = info['lives']
            lives_lost = 0
        
        while not terminated and not truncated:
            action = agent.act(state, eps=0.0)  # No exploration during evaluation
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Track lives lost (if available in info)
            if 'lives' in info and info['lives'] < initial_lives:
                lives_lost += (initial_lives - info['lives'])
                initial_lives = info['lives']
            
            state = next_state
            score += reward
            steps += 1
            
            if render and delay > 0:
                time.sleep(delay)
        
        # Collect statistics
        scores.append(score)
        max_score = max(max_score, score)
        min_score = min(min_score, score)
        total_steps += steps
        
        if 'lives' in info:
            lives_data.append(lives_lost)
        
        if i % 10 == 0 or i == 1:
            print(f"Episode {i}/{episodes} - Score: {score}")
    
    # Calculate statistics
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    std_score = np.std(scores)
    avg_steps = total_steps / episodes
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Number of episodes: {episodes}")
    print(f"Mean score: {mean_score:.2f}")
    print(f"Median score: {median_score:.2f}")
    print(f"Standard deviation: {std_score:.2f}")
    print(f"Min score: {min_score}")
    print(f"Max score: {max_score}")
    print(f"Average steps per episode: {avg_steps:.2f}")
    
    if lives_data:
        print(f"Average lives lost per episode: {np.mean(lives_data):.2f}")
    
    # Plotting the score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, alpha=0.7, color='blue')
    plt.axvline(mean_score, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_score:.2f}')
    plt.axvline(median_score, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_score:.2f}')
    plt.title('Score Distribution during Evaluation')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('evaluation_score_distribution.png')
    
    # Save evaluation results to file
    if save_results:
        results = {
            'model_path': model_path,
            'episodes': episodes,
            'mean_score': mean_score,
            'median_score': median_score,
            'std_score': std_score,
            'min_score': min_score,
            'max_score': max_score,
            'avg_steps': avg_steps,
            'scores': scores
        }
        
        if lives_data:
            results['avg_lives_lost'] = np.mean(lives_data)
        
        # Save detailed results to text file
        with open('evaluation_results.txt', 'w') as f:
            f.write("Evaluation Results\n")
            f.write("=================\n\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Episodes: {episodes}\n\n")
            f.write(f"Mean score: {mean_score:.2f}\n")
            f.write(f"Median score: {median_score:.2f}\n")
            f.write(f"Standard deviation: {std_score:.2f}\n")
            f.write(f"Min score: {min_score}\n")
            f.write(f"Max score: {max_score}\n")
            f.write(f"Average steps per episode: {avg_steps:.2f}\n")
            
            if lives_data:
                f.write(f"Average lives lost per episode: {np.mean(lives_data):.2f}\n\n")
            
            f.write("Individual episode scores:\n")
            for i, score in enumerate(scores, 1):
                f.write(f"Episode {i}: {score}\n")
        
        print(f"Evaluation results saved to evaluation_results.txt")
        print(f"Score distribution plot saved to evaluation_score_distribution.png")
    
    env.close()
    return results

def compare_models(model_paths, episodes=50, save_results=True):
    """
    Compare multiple trained models by evaluating each one.
    
    Args:
        model_paths (list): List of paths to different model weights
        episodes (int): Number of episodes to evaluate each model
        save_results (bool): Whether to save the comparison results
    """
    if not model_paths:
        print("No models provided for comparison")
        return
    
    results = {}
    for model_path in model_paths:
        print(f"\nEvaluating model: {model_path}")
        model_result = evaluate_agent(model_path, episodes=episodes, render=False, save_results=False)
        if model_result:
            results[model_path] = model_result
    
    if not results:
        print("No valid results to compare")
        return
    
    # Compare results
    print("\nModel Comparison:")
    print("================")
    
    # Create a table format
    headers = ["Model", "Mean Score", "Median Score", "Std Dev", "Max Score"]
    row_format = "{:<30} {:<12} {:<12} {:<12} {:<12}"
    
    print(row_format.format(*headers))
    print("-" * 80)
    
    for model_path, result in results.items():
        model_name = model_path.split('/')[-1]
        print(row_format.format(
            model_name,
            f"{result['mean_score']:.2f}",
            f"{result['median_score']:.2f}",
            f"{result['std_score']:.2f}",
            f"{result['max_score']}"
        ))
    
    # Plotting comparison
    plt.figure(figsize=(12, 6))
    
    # Box plot for score distribution
    data = [result['scores'] for result in results.values()]
    labels = [path.split('/')[-1] for path in results.keys()]
    
    plt.boxplot(data, labels=labels, patch_artist=True)
    plt.title('Score Distribution Comparison Between Models')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    
    # Save comparison results if requested
    if save_results:
        with open('model_comparison_results.txt', 'w') as f:
            f.write("Model Comparison Results\n")
            f.write("=======================\n\n")
            f.write(f"Episodes per model: {episodes}\n\n")
            
            for model_path, result in results.items():
                model_name = model_path.split('/')[-1]
                f.write(f"Model: {model_name}\n")
                f.write(f"Mean score: {result['mean_score']:.2f}\n")
                f.write(f"Median score: {result['median_score']:.2f}\n")
                f.write(f"Standard deviation: {result['std_score']:.2f}\n")
                f.write(f"Min score: {result['min_score']}\n")
                f.write(f"Max score: {result['max_score']}\n")
                f.write(f"Average steps per episode: {result['avg_steps']:.2f}\n")
                f.write("\n")
        
        print(f"Comparison results saved to model_comparison_results.txt")
        print(f"Comparison plot saved to model_comparison.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained DQN agent on Breakout')
    parser.add_argument('--model', type=str, default='ram_state_representation/dqn_breakout_ram_final.pth',
                        help='Path to the saved model weights')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during evaluation')
    parser.add_argument('--delay', type=float, default=0.01,
                        help='Delay between frames when rendering (seconds)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare multiple models')
    parser.add_argument('--models', nargs='+', 
                        help='List of model paths to compare (use with --compare)')
    
    args = parser.parse_args()
    
    if args.compare:
        if not args.models:
            # Default models to compare if none specified
            args.models = [
                'ram_state_representation/dqn_breakout_ram_episode_10000.pth',
                'ram_state_representation/dqn_breakout_ram_episode_20000.pth',
                'ram_state_representation/dqn_breakout_ram_final.pth'
            ]
        compare_models(args.models, episodes=args.episodes)
    else:
        evaluate_agent(args.model, episodes=args.episodes, render=args.render, delay=args.delay) 