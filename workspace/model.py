from torch import nn
import torch

class SimpleMNISTClassifier(nn.Module):
    def __init__(self, depth: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, depth, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(28 * 28 * depth, 10),
        )

    def forward(self, x):
        return self.net(x)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)