# train.py

import ale_py
import gymnasium as gym
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
from dqn_agent import DQNAgent

N_EPISODES = 20000
MAX_T = 1000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.999
SCORE_SOLVED = 30.0
PRINT_EVERY = 100
SAVE_EVERY = 500


env = gym.make('ALE/Breakout-ram-v5', render_mode=None)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print(f'State size: {state_size}, Action size: {action_size}')

agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0)
gi
scores = []
scores_window = deque(maxlen=100)
eps = EPS_START

print("\nStarting Training...")
for i_episode in tqdm(range(1, N_EPISODES + 1), desc="Training Progress"):
    state, info = env.reset()
    score = 0
    terminated = False
    truncated = False
    timestep = 0

    while not terminated and not truncated and timestep < MAX_T:
        action = agent.act(state, eps)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.step(state, action, reward, next_state, terminated or truncated)
        state = next_state
        score += reward
        timestep += 1
        if terminated or truncated:
            break

    scores_window.append(score)
    scores.append(score)
    eps = max(EPS_END, EPS_DECAY * eps)

    if i_episode % PRINT_EVERY == 0:
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps:.4f}')

    if i_episode % SAVE_EVERY == 0:
        agent.save(f"dqn_breakout_ram_episode_{i_episode}.pth")

    if np.mean(scores_window) >= SCORE_SOLVED:
        print(f'\nEnvironment solved in {i_episode:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
        agent.save("dqn_breakout_ram_solved.pth")
        break
    if i_episode % 100 == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        rolling_avg = np.convolve(scores, np.ones(100) / 100, mode='valid')
        plt.plot(np.arange(len(rolling_avg)) + 99, rolling_avg,
                 label='Rolling Avg (100 episodes)')
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.title('DQN Training Scores for Breakout (RAM)')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_scores_breakout_ram.png')
        plt.show()

# Save final model
agent.save("dqn_breakout_ram_final.pth")
env.close()
print("Training finished.")


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
rolling_avg = np.convolve(scores, np.ones(100)/100, mode='valid')
plt.plot(np.arange(len(rolling_avg)) + 99, rolling_avg, label='Rolling Avg (100 episodes)')
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.title('DQN Training Scores for Breakout (RAM)')
plt.legend()
plt.grid(True)
plt.savefig('training_scores_breakout_ram.png')
print("Score plot saved as training_scores_breakout_ram.png")
