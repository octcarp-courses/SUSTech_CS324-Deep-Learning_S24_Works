import torch


class AverageMeter:
    def __init__(self, name: str, fmt: str = ":f") -> None:
        self.name: str = name
        self.fmt: str = fmt
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        return f"{self.name}: {self.val:{self.fmt}} ({self.avg:{self.fmt}})"


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    acc = correct / target.size(0)
    return acc.item()
