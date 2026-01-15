import torch
import torch.optim as optim

from datasets.oasis.loader import DummyOASISDataset
from models.cnn.simple_cnn import SimpleCNN
from tasks.classification.binary_nc_ad import BinaryNCADTask
from core.engine import train_one_batch


def main():
    dataset = DummyOASISDataset(num_samples=20)
    model = SimpleCNN()
    task = BinaryNCADTask()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # create a fake batch
    images = []
    labels = []

    for i in range(8):
        img, lbl = dataset[i]
        images.append(img)
        labels.append(lbl)

    images = torch.stack(images)          # (8, 1, 64, 64)
    labels = torch.tensor(labels)          # (8,)

    loss, metrics = train_one_batch(
        model, task, optimizer, images, labels
    )

    print("Loss:", loss)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
