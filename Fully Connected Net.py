import torch
import tqdm
from torch import nn

print(torch.__version__)
print(torch.cuda.is_available())

EPOCHS = 10000
SAMPLES = 5
TOKENS = 768
DEPTH = 3
F_IN = 5
F_OUT = 2


inputs = torch.rand(SAMPLES, F_IN).cuda()
labels = torch.rand(SAMPLES, F_OUT).cuda()


class PilotModule(nn.Module):
    def __init__(self, tokens: int, **kwargs):
        super(PilotModule, self).__init__(**kwargs)
        self.tokens = tokens
        self.linear = nn.Linear(tokens, tokens, True)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.shape[-1] == self.tokens
        ), f"Expected input with {self.tokens} tokens, but got {x.shape[-1]}"

        return self.relu(self.linear(x))


class AutoPilot(nn.Module):

    def __init__(self, depth: int, f_in: int, f_out: int, tokens: int, **kwargs):

        super(AutoPilot, self).__init__(**kwargs)
        self.layers = nn.Sequential(
            nn.Linear(f_in, tokens, True),
            *[PilotModule(tokens) for _ in range(depth)],
            nn.Linear(tokens, f_out, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Tracker:
    def __init__(self):
        self.minv = float("inf")
        self.maxv = -float("inf")
        self.last = 0.0
        self.avgv = 0.0
        self.ticks = 0

    def log(self, value: float):
        self.last = value
        self.minv = min(self.minv, value)
        self.maxv = max(self.maxv, value)
        self.avgv = (self.avgv * self.ticks + value) / (self.ticks + 1)
        self.ticks += 1

    def __str__(self):
        return f"Min: {self.minv:.4f}, Max: {self.maxv:.4f}, Avg: {self.avgv:.4f}"


class Trainer:
    def __init__(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.tracker = Tracker()

    def step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        self.tracker.log(loss.item())
        return self.tracker.last


class Evaluator:
    def __init__(self, model: nn.Module, criterion: nn.Module):
        self.model = model
        self.criterion = criterion
        self.tracker = Tracker()

    def eval(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            output = self.model(x)
            loss = self.criterion(output, y)
            self.tracker.log(loss.item())
            return self.tracker.last


net = AutoPilot(DEPTH, F_IN, F_OUT, TOKENS).cuda()
opt = torch.optim.Adam(net.parameters(), lr=0.0005)
loss = nn.MSELoss().cuda()

trainer = Trainer(net, opt, loss)
validator = Evaluator(net, loss)

epochs = tqdm.trange(EPOCHS, desc="Training")

for epoch in epochs:

    train_loss = trainer.step(inputs, labels)
    val_loss = validator.eval(inputs, labels)

    epochs.set_postfix(
        {"Trainer": str(trainer.tracker), "Validator": str(validator.tracker)}
    )

Y = net(inputs)
print(torch.allclose(Y, labels, atol=1e-2))
