import pandas as pd
import numpy as np

# models
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

class HouseValuePredictor(nn.Module):
    def __init__(self, input_dim):
        super(HouseValuePredictor, self).__init__()
        
        # defining the number of layers and their size
        self.fc1 = nn.Linear(input_dim, 128).type(torch.float32)
        self.bn1 = nn.BatchNorm1d(128).type(torch.float32)
        
        self.fc2 = nn.Linear(128, 32).type(torch.float32)
        self.bn2 = nn.BatchNorm1d(32).type(torch.float32)
        
        self.fc3 = nn.Linear(32, 1).type(torch.float32)
        
        # defining dropouts
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # using ReLU as the activation function
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x