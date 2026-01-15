import torch


def train_one_batch(model, task, optimizer, images, labels):
    model.train()

    optimizer.zero_grad()

    predictions = model(images)
    loss = task.compute_loss(predictions, labels)

    loss.backward()
    optimizer.step()

    metrics = task.compute_metrics(predictions, labels)

    return loss.item(), metrics
