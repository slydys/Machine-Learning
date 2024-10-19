import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

class MyCNN(nn.Module):
        def __init__(self):
            #create CNN module
            super(MyCNN, self).__init__()
            self.Conv1 = nn.Conv2d(1, 32, 3, 1)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.Conv2 = nn.Conv2d(32, 64, 3, 1, 1)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.Conv3 = nn.Conv2d(64, 128, 3, 1, 1)
            self.pool3 = nn.MaxPool2d(2, 2)

            self.fc1 = nn.Linear(128 * 3 * 3, 768)
            self.fc2 = nn.Linear(768, 10)

        def forward(self, x):
            x = self.pool1(func.relu(self.Conv1(x)))
            x = self.pool2(func.relu(self.Conv2(x)))
            x = self.pool3(func.relu(self.Conv3(x)))
            x = x.view(-1, 128 * 3 * 3)
            x = func.relu(self.fc1(x))
            x = self.fc2(x)

            x = func.softmax(x, dim = 1)
            return x

