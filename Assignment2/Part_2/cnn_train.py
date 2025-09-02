import argparse
import numpy as np
import os
import csv

from tqdm import tqdm

from cnn_model import CNN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Default constants
LEARNING_RATE_DEFAULT: float = 1e-4
BATCH_SIZE_DEFAULT: int = 32
MAX_EPOCHS_DEFAULT: int = 200
EVAL_FREQ_DEFAULT: int = 1
OPTIMIZER_DEFAULT: str = "ADAM"
DATA_DIR_DEFAULT: str = "../data"
MODEL_PARA_PATH_DEFAULT: str = ""
EXP_FILENAME: str = "results/latest.csv"

FLAGS = None


def accuracy(predictions, targets) -> None:
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    pass


def train(
    train_loader: DataLoader,
    test_loader: DataLoader,
    model_para_path: str = MODEL_PARA_PATH_DEFAULT,
    max_epoch: list[int] = MAX_EPOCHS_DEFAULT,
    eval_freq: list[int] = EVAL_FREQ_DEFAULT,
    learning_rate: float = LEARNING_RATE_DEFAULT,
    optimizer_type: str = OPTIMIZER_DEFAULT,
) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name(device)}")

    cnn = CNN(3, 10).to(device)

    if model_para_path != "":
        cnn.load_state_dict(torch.load(model_para_path))
        print(f"Loaded model from {model_para_path}")

    loss_fn = nn.CrossEntropyLoss().to(device)
    if optimizer_type == "ADAM":
        optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
    elif optimizer_type == "SGD":
        optimizer = optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_type == "RMS":
        optimizer = optim.RMSprop(cnn.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

    eval_epochs: list[int] = []
    train_acc_l: list[float] = []
    test_acc_l: list[float] = []
    train_loss_l: list[float] = []
    test_loss_l: list[float] = []

    for epoch in tqdm(range(max_epoch)):
        train_total: int = 0
        train_correct: int = 0
        train_loss: float = 0.0

        cnn.train()
        for data in train_loader:
            train_x, train_y = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = cnn(train_x)
            loss = loss_fn(outputs, train_y)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            train_total += train_y.size(0)
            train_correct += (predicted == train_y).sum().item()
            train_loss += loss.item()

        if epoch % eval_freq == 0 or epoch == max_epoch - 1:
            test_correct: int = 0
            test_total: int = 0
            test_loss: float = 0.0

            cnn.eval()
            with torch.no_grad():
                for data in test_loader:
                    test_x, test_y = data[0].to(device), data[1].to(device)
                    outputs = cnn(test_x)
                    loss = loss_fn(outputs, test_y)

                    _, predicted = torch.max(outputs, 1)
                    test_total += test_y.size(0)
                    test_correct += (predicted == test_y).sum().item()
                    test_loss += loss.item()

            train_acc = train_correct / train_total
            test_acc = test_correct / test_total
            train_loss = train_loss / len(train_loader)
            test_loss = test_loss / len(test_loader)

            eval_epochs.append(epoch)
            train_acc_l.append(train_acc)
            test_acc_l.append(test_acc)
            train_loss_l.append(train_loss)
            test_loss_l.append(test_loss)

    cnn.eval()
    with torch.no_grad():
        class_correct = torch.zeros(10).to(device)
        class_total = torch.zeros(10).to(device)
        for data in train_loader:
            train_x, train_y = data[0].to(device), data[1].to(device)
            outputs = cnn(train_x)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == train_y).squeeze()
            for i in range(train_y.size(0)):
                label = train_y[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
        train_classes_acc = (class_correct / class_total).tolist()

        class_correct = torch.zeros(10).to(device)
        class_total = torch.zeros(10).to(device)
        for data in test_loader:
            test_x, test_y = data[0].to(device), data[1].to(device)
            outputs = cnn(test_x)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == test_y).squeeze()
            for i in range(test_y.size(0)):
                label = test_y[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        test_classes_acc = (class_correct / class_total).tolist()

    if model_para_path != "":
        torch.save(cnn.state_dict(), model_para_path)
        print(f"Saved model to {model_para_path}")

    # save data to .csv

    with open(EXP_FILENAME, "w", newline="") as csvfile:
        fieldnames = ["Epoch", "Train Acc", "Test Acc", "Train Loss", "Test Loss"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for epoch, train_acc, test_acc, train_loss, test_loss in zip(
            eval_epochs, train_acc_l, test_acc_l, train_loss_l, test_loss_l
        ):
            writer.writerow(
                {
                    "Epoch": epoch,
                    "Train Acc": train_acc,
                    "Test Acc": test_acc,
                    "Train Loss": train_loss,
                    "Test Loss": test_loss,
                }
            )
    print(f"Save data to {EXP_FILENAME}")

    print(f"After {max_epoch} Epochs:")
    print(
        f"Train Acc: {train_acc_l[-1] * 100:.4f}%, Test Acc: {test_acc_l[-1] * 100:.4f}%"
    )
    print(f"Train Loss: {train_loss_l[-1]:.6f}, Test Loss: {test_loss_l[-1]:.6f}")

    return (
        eval_epochs,
        train_acc_l,
        test_acc_l,
        train_loss_l,
        test_loss_l,
        train_classes_acc,
        test_classes_acc,
    )


# def main():
#     """
#     Main function
#     """
#     train()
#
#
# if __name__ == '__main__':
#     # Command line arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
#                         help='Learning rate')
#     parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
#                         help='Number of steps to run trainer.')
#     parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
#                         help='Batch size to run trainer.')
#     parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
#                         help='Frequency of evaluation on the test set')
#     parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
#                         help='Directory for storing input data')
#     FLAGS, unparsed = parser.parse_known_args()
#
#     main()
