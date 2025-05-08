import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Neural network model for approximating Q-values."""
    def __init__(self, state_size, output_size, hidden_size=128):
        """Initialize neural network with three fully connected layers."""
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        """Build network that maps state to action values."""
        x = F.relu(self.fc1(state))
        h = F.relu(self.fc2(x))
        return self.fc3(h)