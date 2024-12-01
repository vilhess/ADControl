import torch 
import torch.nn as nn 
import torch.nn.functional as F

class MNIST_LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.rep_dim = 64
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(16, eps=1e-04, affine=False)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

        self.fc = nn.Linear(64 * 3 * 3, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn3(x)))

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x