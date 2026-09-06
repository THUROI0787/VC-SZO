"""Train ANN VGG9 on CIFAR-10 for TTA baseline comparison.

Trains a standard (non-spiking) VGG9 with BatchNorm on CIFAR-10.
Supports multiple training configs (learning rate, weight decay, scheduler).

Reference configs (from literature survey):
  Config A (default): lr=0.1, wd=5e-4, cosine 160 epochs  — standard VGG recipe
  Config B:           lr=0.3, wd=5e-5, cosine 160 epochs  — user's SNN-style recipe
  Config C:           lr=0.05, wd=5e-4, step decay       — Torch VGG recipe
  Config D:           lr=0.01, wd=1e-4, cosine 200 epochs — conservative

Usage:
    python phase0/train_ann_vgg9.py --config A --epochs 160 --batch_size 128
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tta_snn_zo.utils import accuracy, set_seed, setup_logger, save_checkpoint


# --------------------------------------------------------------------------- #
#  ANN VGG9 (standard BatchNorm, no spiking)
# --------------------------------------------------------------------------- #
class ANN_VGG9(nn.Module):
    """Standard VGG-9 with BatchNorm for CIFAR-10.

    Architecture matches SNN_VGG9 but uses ReLU activations instead of LIF.
    """
    def __init__(self, num_classes: int = 10, width_scale: float = 1.0):
        super().__init__()
        w = width_scale
        c1, c2, c3 = int(64 * w), int(128 * w), int(256 * w)

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, c1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            # Block 2
            nn.Conv2d(c1, c2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            # Block 3
            nn.Conv2d(c2, c3, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(c3 * 4 * 4, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes, bias=False),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


# --------------------------------------------------------------------------- #
#  Training configs
# --------------------------------------------------------------------------- #
TRAINING_CONFIGS = {
    "A": {
        "lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4,
        "scheduler": "cosine", "epochs": 160,
        "description": "Standard VGG cosine (lr=0.1, wd=5e-4)",
    },
    "B": {
        "lr": 0.3, "momentum": 0.9, "weight_decay": 5e-5,
        "scheduler": "cosine", "epochs": 160,
        "description": "SNN-style cosine (lr=0.3, wd=5e-5)",
    },
    "C": {
        "lr": 0.05, "momentum": 0.9, "weight_decay": 5e-4,
        "scheduler": "step", "epochs": 200,
        "description": "Step decay (lr=0.05, milestones 100/150)",
    },
    "D": {
        "lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4,
        "scheduler": "cosine", "epochs": 200,
        "description": "Conservative cosine (lr=0.01, wd=1e-4)",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="B", choices=list(TRAINING_CONFIGS.keys()))
    p.add_argument("--epochs", type=int, default=0, help="override config epochs")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0, help="override config lr")
    p.add_argument("--weight_decay", type=float, default=0, help="override config wd")
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--out_dir", type=str, default="outputs/phase0_final/ANN")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = dict(TRAINING_CONFIGS[args.config])
    if args.epochs > 0:
        cfg["epochs"] = args.epochs
    if args.lr > 0:
        cfg["lr"] = args.lr
    if args.weight_decay > 0:
        cfg["weight_decay"] = args.weight_decay

    set_seed(args.seed)
    run_name = args.run_name or f"ann_vgg9_config{args.config}_lr{cfg['lr']}_wd{cfg['weight_decay']}_e{cfg['epochs']}_s{args.seed}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("train_ann", str(out_dir), run_name)
    logger.info(f"Config {args.config}: {cfg['description']}")
    logger.info(f"Hyperparameters: {cfg}")
    logger.info(f"args: {vars(args)}")

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {device}")

    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_root, train=True,
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=args.data_root, train=False,
                                           download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers,
                                              pin_memory=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers,
                                             pin_memory=True)

    # Model
    model = ANN_VGG9(num_classes=10).to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg["lr"],
                          momentum=cfg["momentum"],
                          weight_decay=cfg["weight_decay"])

    if cfg["scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    elif cfg["scheduler"] == "step":
        scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    else:
        scheduler = None

    # Training loop
    best_acc = 0.0
    best_ckpt = None
    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        # Evaluate
        model.eval()
        test_acc = 0.0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                test_acc += accuracy(outputs, targets)[0] * inputs.size(0)
        test_acc /= len(testset)

        if test_acc > best_acc:
            best_acc = test_acc
            best_ckpt = {
                "state_dict": model.state_dict(),
                "test_acc": best_acc,
                "epoch": epoch,
                "config": cfg,
                "args": vars(args),
            }

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:3d}/{cfg['epochs']} | "
                        f"Train Loss: {train_loss/len(trainloader):.4f} | "
                        f"Test Acc: {test_acc:.2f}% | "
                        f"Best: {best_acc:.2f}% | "
                        f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                        f"Time: {time.time()-t0:.0f}s")

    # Save best checkpoint
    ckpt_path = str(out_dir / f"{run_name}_best.pt")
    save_checkpoint(ckpt_path, best_ckpt["state_dict"], extra={
        "test_acc": best_acc, "epoch": best_ckpt["epoch"],
        "config": cfg, "args": vars(args),
    })
    logger.info(f"Best test accuracy: {best_acc:.2f}%")
    logger.info(f"Checkpoint saved: {ckpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())