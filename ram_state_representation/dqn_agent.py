import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

from ram_state_representation.dqn_model import QNetwork
from ram_state_representation.replay_buffer import ReplayBuffer

BUFFER_SIZE = 50_000
BATCH_SIZE = 64
GAMMA = 0.999
LR = 1e-4
TARGET_UPDATE_EVERY = 4

class DQNAgent():
    def __init__(self, state_size, action_size, seed=0):
        """Initialize the DQN Agent with networks, replay buffer and hyperparameters."""
        self.state_size = state_size
        self.action_size = action_size
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.qnetwork_local = QNetwork(state_size, action_size).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """Store experience in replay memory and trigger learning when appropriate."""
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % TARGET_UPDATE_EVERY
        loss = None
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                loss = self.learn(experiences, GAMMA)
        return loss

    def act(self, state, eps=0.):
        """Select action using epsilon-greedy policy based on current Q-network."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update Q-network parameters using batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return loss.item()

    def soft_update(self, local_model, target_model, tau=1e-3):
        """Gradually update target network parameters from local network."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename="dqn_breakout_ram.pth"):
        """Save trained model parameters to file."""
        torch.save(self.qnetwork_local.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load(self, filename="dqn_breakout_ram.pth"):
        """Load model parameters from file."""
        try:
            self.qnetwork_local.load_state_dict(torch.load(filename, map_location=self.device))
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
            self.qnetwork_local.eval()
            self.qnetwork_target.eval()
            print(f"Model loaded from {filename}")
        except FileNotFoundError:
             print(f"Error: Could not find model file {filename}")
        except Exception as e:
             print(f"Error loading model: {e}")
