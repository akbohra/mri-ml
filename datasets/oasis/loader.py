import torch
from core.base import BaseDataset


class DummyOASISDataset(BaseDataset):
    """
    Dummy dataset to validate the ML pipeline.
    Returns random images and valid labels.
    """

    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Fake MRI image: 1 channel, 64x64
        image = torch.randn(1, 64, 64)

        # Fake label: 0 (NC) or 1 (AD)
        label = torch.randint(0, 2, (1,)).item()

        return image, label
