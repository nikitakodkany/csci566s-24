import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_outputs):
        super(MLPModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # MLP layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        for i in range(num_layers - 1):
            setattr(self, f'fc{i+2}', nn.Linear(hidden_size, hidden_size))
            setattr(self, f'relu{i+2}', nn.ReLU())
        self.fc_out = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        for i in range(self.num_layers - 1):
            x = getattr(self, f'fc{i+2}')(x)
            x = getattr(self, f'relu{i+2}')(x)
        x = self.fc_out(x)
        return x
        