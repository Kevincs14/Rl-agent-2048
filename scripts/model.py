import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, action_dim=4):
        super(PolicyNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=2, padding=1)
        
        self.flat_size = 128 * 5 * 5
        
        self.fc1 = nn.Linear(self.flat_size, 512)
        self.fc2 = nn.Linear(512, 256)
        
        self.actor = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)
        
    def forward(self, state):
        x = state.view(-1, 1, 4, 4)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.actor(x)
        value = self.critic(x)
        return value.squeeze(-1), logits.squeeze(0)
