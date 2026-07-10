import torch
import tqdm
from torch import nn

print(torch.__version__)
print(torch.cuda.is_available())

EPOCHS = 100
SAMPLES = 10000
IMG_SIZE = 32
NUM_CLASSES = 100
MODEL_DEPTH = 1
KERNEL_DEPTH = 1
KERNEL_SIZE = 3


inputs = torch.rand(SAMPLES, 3, IMG_SIZE, IMG_SIZE).cuda()
labels = torch.randint(0, NUM_CLASSES, (SAMPLES,)).cuda()


class PilotModule(nn.Module):
    def __init__(self, cin: int, cout: int, size: int, **kwargs):
        super(PilotModule, self).__init__(**kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=cin,
            out_channels=cout,
            kernel_size=(size, size),
            stride=1,
            padding="same",
            bias=True,
        )
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2d(x))


class AutoPilot(nn.Module):

    def __init__(self, depth: int, f_in: int, f_out: int, **kwargs):

        super(AutoPilot, self).__init__(**kwargs)
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=f_in,
                out_channels=KERNEL_DEPTH,
                kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
                stride=1,
                padding="same",
                bias=True,
            ),
            *[
                PilotModule(KERNEL_DEPTH, KERNEL_DEPTH, KERNEL_SIZE)
                for _ in range(depth)
            ],
            nn.Flatten(),
            nn.Linear(KERNEL_DEPTH * IMG_SIZE * IMG_SIZE, f_out, True),
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


net = AutoPilot(MODEL_DEPTH, 3, NUM_CLASSES).cuda()
opt = torch.optim.Adam(net.parameters(), lr=0.0005)
loss = nn.CrossEntropyLoss().cuda()

trainer = Trainer(net, opt, loss)
validator = Evaluator(net, loss)

epochs = tqdm.trange(EPOCHS, desc="Training")

for epoch in epochs:

    train_loss = trainer.step(inputs, labels)
    val_loss = validator.eval(inputs, labels)

    epochs.set_postfix(
        {"Trainer": str(trainer.tracker), "Validator": str(validator.tracker)}
    )

with torch.no_grad():
    logits = net(inputs)
    preds = logits.argmax(dim=1)

    accuracy = (preds == labels).float().mean()
    print(f"Accuracy: {accuracy:.3f}")
