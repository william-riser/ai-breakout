import ale_py
import gymnasium as gym
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tqdm import tqdm
from dqn_agent import DQNAgent

N_EPISODES = 1_000_000
MAX_T = 500
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.999
SCORE_SOLVED = 100.0
PRINT_EVERY = 100
SAVE_EVERY = 500

env = gym.make('ALE/Breakout-ram-v5', render_mode=None)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print(f'State size: {state_size}, Action size: {action_size}')

agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0)
scores = []
scores_window = deque(maxlen=100)
episode_losses = []  # to store the average loss per episode
eps = EPS_START

print("\nStarting Training...")
for i_episode in tqdm(range(1, N_EPISODES + 1), desc="Training Progress"):
    state, info = env.reset()
    score = 0
    terminated = False
    truncated = False
    timestep = 0
    losses_episode = []  # losses for this episode

    while not terminated and not truncated and timestep < MAX_T:
        action = agent.act(state, eps)
        next_state, reward, terminated, truncated, info = env.step(action)
        loss = agent.step(state, action, reward, next_state, terminated or truncated)
        if loss is not None:
            losses_episode.append(loss)
        state = next_state
        score += reward
        timestep += 1
        if terminated or truncated:
            break

    scores_window.append(score)
    scores.append(score)
    # Record the average loss for the episode (or NaN if no loss was recorded)
    avg_loss = np.mean(losses_episode) if len(losses_episode) > 0 else np.nan
    episode_losses.append(avg_loss)
    eps = max(EPS_END, EPS_DECAY * eps)

    if i_episode % PRINT_EVERY == 0:
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\t'
              f'Epsilon: {eps:.4f}\tAvg Loss: {avg_loss if not np.isnan(avg_loss) else "N/A"}')

    if i_episode % SAVE_EVERY == 0:
        agent.save(f"dqn_breakout_ram_episode_{i_episode}.pth")

    if np.mean(scores_window) >= SCORE_SOLVED:
        print(f'\nEnvironment solved in {i_episode:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
        agent.save("dqn_breakout_ram_solved.pth")
        break

    # Plot scores and loss every 100 episodes for monitoring
    if i_episode % 100 == 0:
        # Plot loss
        fig_loss = plt.figure()
        ax_loss = fig_loss.add_subplot(111)
        # Note: some episodes might not have a training loss, so they appear as NaN.
        plt.plot(np.arange(len(episode_losses)), episode_losses, label='Average Loss per Episode')
        plt.ylabel('Loss')
        plt.xlabel('Episode #')
        plt.title('DQN Training Loss for Breakout (RAM)')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_loss_breakout_ram.png')
        plt.show()
        # Plot scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores, label='Score per Episode')
        rolling_avg = np.convolve(scores, np.ones(100) / 100, mode='valid')
        plt.plot(np.arange(len(rolling_avg)) + 99, rolling_avg, label='Rolling Avg (100 episodes)')
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

# Final score plot
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores, label='Score per Episode')
rolling_avg = np.convolve(scores, np.ones(100)/100, mode='valid')
plt.plot(np.arange(len(rolling_avg)) + 99, rolling_avg, label='Rolling Avg (100 episodes)')
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.title('DQN Training Scores for Breakout (RAM)')
plt.legend()
plt.grid(True)
plt.savefig('training_scores_breakout_ram.png')
print("Score plot saved as training_scores_breakout_ram.png")

# Final loss plot
fig_loss = plt.figure()
ax_loss = fig_loss.add_subplot(111)
plt.plot(np.arange(len(episode_losses)), episode_losses, label='Average Loss per Episode')
plt.ylabel('Loss')
plt.xlabel('Episode #')
plt.title('DQN Training Loss for Breakout (RAM)')
plt.legend()
plt.grid(True)
plt.savefig('training_loss_breakout_ram.png')
print("Loss plot saved as training_loss_breakout_ram.png")
