import torch
import torch.nn as nn

class DRQN_GRU(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(DRQN_GRU, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
        )
        self.gru1 = nn.GRU(256, 256, batch_first=True)
        self.value_hidden = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
        )
        self.advantage_hidden = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
        )
        self.value = nn.Linear(32, 1)
        self.advantage = nn.Linear(32, output_size)

    def forward(self, input):
        self.gru1.flatten_parameters()
        x = self.layer1(input)
        x, hidden = self.gru1(x)
        value_hidden = self.value_hidden(x[:, -1, :])
        advantage_hidden = self.advantage_hidden(x[:, -1, :])
        value = self.value(value_hidden)
        advantage = self.advantage(advantage_hidden)
        output = value + (advantage - torch.mean(advantage, dim=1, keepdim=True))
        return output
    