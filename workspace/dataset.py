from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

from .pl_dataset import DataModule


class MNISTDataset(DataModule):

    def _build_trainset(self) -> MNIST:
        return MNIST(
            root=self.config.data.rootdir,
            train=True,
            download=True,
            transform=ToTensor(),
        )

    def _build_valset(self) -> MNIST:
        return MNIST(
            root=self.config.data.rootdir,
            train=False,
            download=True,
            transform=ToTensor(),
        )
