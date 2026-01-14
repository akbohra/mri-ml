# for abstract base classes to define the behaviour of the classes
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """
    Dataset template.
    Dataset implementations should handle:
    - loading data from disk
    - returning samples in a consistent format
    """

    @abstractmethod
    def __len__(self):
        """Return number of samples"""
        pass

    @abstractmethod
    def __getitem__(self, index):
        """
        Return one sample.
        Expected to return (image, label).
        """
        pass


class BaseModel(ABC):
    """
    Abstract interface for all models.
    Models should implement forward inference.
    """

    @abstractmethod
    def forward(self, x):
        """Forward pass"""
        pass


class BaseTask(ABC):
    """
    Defines what problem is being solved.
    Responsible for:
    - loss computation
    - metric computation
    - interpreting model outputs
    """

    @abstractmethod
    def compute_loss(self, predictions, targets):
        pass

    @abstractmethod
    def compute_metrics(self, predictions, targets):
        pass
