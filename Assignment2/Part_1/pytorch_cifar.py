import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from pytorch_mlp import MLP
from tqdm import tqdm

DEFAULT_MAX_EPOCH = 150
DEFAULT_EVAL_FREQ = 1
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_OPTIMIZER = "ADAM"


def train(
    train_loader=DataLoader,
    test_loader=DataLoader,
    max_epoch=DEFAULT_MAX_EPOCH,
    eval_freq=DEFAULT_EVAL_FREQ,
    learning_rate=DEFAULT_LEARNING_RATE,
    optimizer_type=DEFAULT_OPTIMIZER,
    validate_loader=None,
) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(n_inputs=32 * 32 * 3, n_hidden=[512, 64], n_classes=10)
    model = model.to(device)

    if optimizer_type == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = nn.CrossEntropyLoss().to(device)

    eval_epochs_l: list[int] = []
    train_acc_l: list[float] = []
    test_acc_l: list[float] = []
    train_loss_l: list[float] = []
    test_loss_l: list[float] = []

    for epoch in tqdm(range(max_epoch)):
        train_correct = 0
        train_total = 0
        train_loss = 0.0
        model.train()
        for data in train_loader:
            train_x, train_y = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(train_x)
            loss = loss_fn(outputs, train_y)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            train_total += train_y.size(0)
            train_correct += (predicted == train_y).sum().item()
            train_loss += loss.item()

        train_acc = train_correct / train_total
        train_loss /= len(train_loader)

        if epoch % eval_freq == 0 or epoch == (max_epoch - 1):
            model.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0.0
            with torch.no_grad():
                for data in test_loader:
                    test_x, test_y = data[0].to(device), data[1].to(device)
                    outputs = model(test_x)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += test_y.size(0)
                    test_correct += (predicted == test_y).sum().item()
                    test_loss += loss_fn(outputs, test_y).item()

            test_acc = test_correct / test_total
            test_loss /= len(test_loader)

            eval_epochs_l.append(epoch)
            train_acc_l.append(train_acc)
            test_acc_l.append(test_acc)
            train_loss_l.append(train_loss)
            test_loss_l.append(test_loss)

    print(f"After {max_epoch} Epochs: ")
    print(
        f"Train Acc: {train_acc_l[-1] * 100:.4f}%, Test Acc: {test_acc_l[-1] * 100:.4f}%"
    )
    print(f"Train Loss: {train_loss_l[-1]:.6f}, Test Loss: {test_loss_l[-1]:.6f}")

    return eval_epochs_l, train_acc_l, test_acc_l, train_loss_l, test_loss_l
