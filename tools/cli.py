import argparse
import torch
import torch.optim as optim

from datasets.oasis.loader import DummyOASISDataset
from models.cnn.simple_cnn import SimpleCNN
from tasks.classification.binary_nc_ad import BinaryNCADTask
from core.engine import train_one_batch


def get_model(name, activation):
    # For now, only one model is implemented
    if name == "simple_cnn":
        model = SimpleCNN()
    else:
        raise ValueError(f"Unknown model: {name}")

    # Activation is currently fixed (intentional extensibility point)
    if activation != "relu":
        print(f"[INFO] Activation '{activation}' not yet implemented, using ReLU")

    return model


def get_task(name):
    if name == "binary_nc_ad":
        return BinaryNCADTask()
    else:
        raise ValueError(f"Unknown task: {name}")


def get_optimizer(name, model, lr):
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif name == "sgd":
        return optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def main():
    parser = argparse.ArgumentParser(description="MRI ML Pipeline CLI")

    parser.add_argument("--task", default="binary_nc_ad",
                        help="Task name (e.g. binary_nc_ad)")
    parser.add_argument("--model", default="simple_cnn",
                        help="Model architecture")
    parser.add_argument("--activation", default="relu",
                        help="Activation function")
    parser.add_argument("--optimizer", default="adam",
                        help="Optimizer (adam, sgd)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--mode", default="train",
                        help="train or test")

    args = parser.parse_args()

    # Dataset (dummy for now)
    dataset = DummyOASISDataset(num_samples=20)

    # Model, task, optimizer
    model = get_model(args.model, args.activation)
    task = get_task(args.task)
    optimizer = get_optimizer(args.optimizer, model, args.lr)

    # Create one batch
    images, labels = [], []
    for i in range(8):
        img, lbl = dataset[i]
        images.append(img)
        labels.append(lbl)

    images = torch.stack(images)
    labels = torch.tensor(labels)

    if args.mode == "train":
        loss, metrics = train_one_batch(
            model, task, optimizer, images, labels
        )
        print("Loss:", loss)
        print("Metrics:", metrics)
    else:
        print("[INFO] Test mode not yet implemented")


if __name__ == "__main__":
    main()
