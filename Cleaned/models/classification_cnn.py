import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassificationCNN(nn.Module):
    def __init__(self, shape_in=[1, 28, 28], num_classes=10, configuration=[32, 64, 128]):
        super(ClassificationCNN, self).__init__()
        self.shape_in = shape_in
        self.num_classes = num_classes
        self.configuration = configuration

        self.conv1 = nn.Conv2d(1, self.configuration[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.configuration[0], self.configuration[1], kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.fc1 = nn.Linear(self.configuration[1]*int(shape_in[1]/4)*int(shape_in[2]/4), self.configuration[2])
        self.fc2 = nn.Linear(self.configuration[2], num_classes)

        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, int(self.configuration[1])*int(self.shape_in[1]/4)*int(self.shape_in[2]/4))

        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout(x)
        out = self.fc2(x)

        return out
    

if __name__ == "__main__":
    model = ClassificationCNN()
    input_data = torch.randn(1, 28, 28)
    print(model.forward(input_data).shape)


