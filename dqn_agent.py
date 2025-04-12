# dqn_agent.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

from dqn_model import QNetwork
from replay_buffer import ReplayBuffer

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-4
TARGET_UPDATE_EVERY = 100

class DQNAgent():
    def __init__(self, state_size, action_size, seed=0):
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
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % TARGET_UPDATE_EVERY
        loss = None
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                # Capture the loss returned by learn()
                loss = self.learn(experiences, GAMMA)
        return loss

    def act(self, state, eps=0.):
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
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        # Return the loss as a scalar value (using .item())
        return loss.item()

    def soft_update(self, local_model, target_model, tau=1e-3):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename="dqn_breakout_ram.pth"):
        torch.save(self.qnetwork_local.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load(self, filename="dqn_breakout_ram.pth"):
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
