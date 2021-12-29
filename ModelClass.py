import torch.nn as nn
from torch.nn.modules.activation import Sigmoid

class LinearRegretion(nn.Module):
    def __init__(self, num_feature):
        super.__init__()
        self.linear = nn.Linear(num_feature, 1)

    def forward(self, X):
        out = self.linear(X)
        return out

class LogisticRegression(nn.Module):
    def __init__(self, num_feature):
        super.__init__()
        self.linear = nn.Linear(num_feature, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, X):
        out = self.linear(X)
        out = self.sigmoid(out)
        return out 

class NeuralNetwork(nn.Module):
    def __init__(self, num_features):
        super.__init__()
        self.linear1 = nn.Linear(num_features, 4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, X):
        out = self.linear1(X)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out