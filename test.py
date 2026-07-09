import torch
from torch import nn

print(torch.__version__)
print(torch.cuda.is_available())


class PilotModule(nn.Module):
    def __init__(self, tokens: int, **kwargs):
        super(PilotModule, self).__init__(**kwargs)
        self.tokens = tokens
        self.linear = nn.Linear(tokens, tokens, True)
        self.relu = nn.ReLU(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.shape[-1] == self.tokens
        ), f"Expected input with {self.tokens} tokens, but got {x.shape[-1]}"

        return self.relu(self.linear(x))


class AutoPilot(nn.Module):

    def __init__(self, depth: int, tokens: int, **kwargs):

        super(AutoPilot, self).__init__(**kwargs)
        self.layers = nn.ModuleList([PilotModule(tokens) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


net = AutoPilot(3, 2)
print(net)

result = net(torch.rand(3, 2))
print(result)
