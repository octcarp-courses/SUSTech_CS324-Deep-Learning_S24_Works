import argparse
import csv

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from utils import AverageMeter, accuracy


CONFIG_DEFAULT: dict[str, int | float | bool] = {
    "input_length": 19,
    "input_dim": 1,
    "num_classes": 10,
    "num_hidden": 128,
    "batch_size": 128,
    "learning_rate": 0.001,
    "max_epoch": 1000,
    "max_norm": 10,
    "data_size": 1000000,
    "portion_train": 0.8,
    "use_scheduler": False,
}


def train(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
) -> tuple[float, float]:
    model.train()
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()

        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["max_norm"])

        optimizer.step()

        acc = accuracy(outputs, batch_targets)
        losses.update(loss.item())
        accuracies.update(acc)

        # if step % 10 == 0:
        #     print(f'[{step}/{len(data_loader)}]', losses, accuracies)
    return losses.avg, accuracies.avg


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
) -> tuple[float, float]:
    model.eval()
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        acc = accuracy(outputs, batch_targets)

        losses.update(loss.item())
        accuracies.update(acc)

        # if step % 10 == 0:
        #     print(f'[{step}/{len(data_loader)}]', losses, accuracies)
    return losses.avg, accuracies.avg


def main(config: dict) -> tuple[list[float], list[float], list[float], list[float]]:
    if config is None:
        config = CONFIG_DEFAULT

    input_length: int = config["input_length"]
    input_dim: int = config["input_dim"]
    num_classes: int = config["num_classes"]
    num_hidden: int = config["num_hidden"]
    batch_size: int = config["batch_size"]
    learning_rate: float = config["learning_rate"]
    max_epoch: int = config["max_epoch"]
    max_norm: float = config["max_norm"]
    data_size: int = config["data_size"]
    portion_train: float = config["portion_train"]
    use_scheduler: bool = config["use_scheduler"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {torch.cuda.get_device_name(device)}")

    # Initialize the model that we are going to use
    model = VanillaRNN(input_length, input_dim, num_hidden, num_classes, device)
    model.to(device)

    # Initialize the dataset and data loader
    dataset = PalindromeDataset(input_length, data_size)

    # Split dataset into train and validation sets
    train_dataset, val_dataset = random_split(
        dataset, [portion_train, 1 - portion_train]
    )

    # Create data loaders for training and validation
    train_dloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    train_acc_l: list[float] = []
    val_acc_l: list[float] = []
    train_loss_l: list[float] = []
    val_loss_l: list[float] = []
    for epoch in tqdm(range(max_epoch)):
        # Train the model for one epoch
        train_loss, train_acc = train(
            model, train_dloader, optimizer, criterion, device, config
        )

        if use_scheduler:
            scheduler.step()

        # Evaluate the trained model on the validation set
        val_loss, val_acc = evaluate(model, val_dloader, criterion, device, config)

        train_acc_l.append(train_acc)
        val_acc_l.append(val_acc)
        train_loss_l.append(train_loss)
        val_loss_l.append(val_loss)

    print("Done training.")

    # save data to .csv
    filename = "results/latest.csv"
    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["Epoch", "Train Acc", "Val Acc", "Train Loss", "Val Loss"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for epoch, train_acc, val_acc, train_loss, val_loss in zip(
            range(max_epoch), train_acc_l, val_acc_l, train_loss_l, val_loss_l
        ):
            writer.writerow(
                {
                    "Epoch": epoch,
                    "Train Acc": train_acc,
                    "Val Acc": val_acc,
                    "Train Loss": train_loss,
                    "Val Loss": val_loss,
                }
            )
    print(f"Save data to {filename}")

    print(f"After {max_epoch} Epochs:")
    print(
        f"Train Acc: {train_acc_l[-1] * 100:.4f}%, Validate Acc: {val_acc_l[-1] * 100:.4f}%"
    )
    print(f"Train Loss: {train_loss_l[-1]:.6f}, Validate Loss: {val_loss_l[-1]:.6f}")

    return train_acc_l, val_acc_l, train_loss_l, val_loss_l


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument(
        "--input_length", type=int, default=19, help="Length of an input sequence"
    )
    parser.add_argument(
        "--input_dim", type=int, default=1, help="Dimensionality of input sequence"
    )
    parser.add_argument(
        "--num_classes", type=int, default=10, help="Dimensionality of output sequence"
    )
    parser.add_argument(
        "--num_hidden",
        type=int,
        default=128,
        help="Number of hidden units in the model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of examples to process in a batch",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--max_epoch", type=int, default=1000, help="Number of epochs to run for"
    )
    parser.add_argument("--max_norm", type=float, default=10.0)
    parser.add_argument(
        "--data_size", type=int, default=1000000, help="Size of the total dataset"
    )
    parser.add_argument(
        "--portion_train",
        type=float,
        default=0.8,
        help="Portion of the total dataset used for training",
    )

    config = parser.parse_args()
    # Train the model
    main(None)
