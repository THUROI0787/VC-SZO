"""Phase 0 - source SNN pretraining with TET loss.

Trains ``SNN_VGG9`` (SpikingJelly LIF + BNTT) on CIFAR-10 with the TET objective
(Temporal Efficient Training, ICLR'22):

    L = (1 - lambda) * mean_t CE(O_t, y) + lambda * mean_t MSE(O_t, phi)

with ``lambda = 5e-2`` and ``phi = v_threshold`` (the paper's default for CIFAR).
Hyperparameters follow the BNTT / PAR reference recipe (SGD lr=0.3, momentum 0.9,
wd 5e-4, step LR decay at 50/70/90% of epochs, T=25, leak->tau=20).

Outputs (into --out_dir):
  * best checkpoint (highest clean test accuracy) ``{run}_best.pt``
  * last checkpoint                          ``{run}_last.pt``
  * full training log                        ``{run}.log``
  * per-epoch metrics CSV                    ``{run}_metrics.csv``
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from spikingjelly.activation_based import functional

cudnn.benchmark = True

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tta_snn_zo.data import cifar10_transform, get_cifar10_loaders  # noqa: E402
from tta_snn_zo.losses import tet_loss  # noqa: E402
from tta_snn_zo.models import build_model  # noqa: E402
from tta_snn_zo.utils import (  # noqa: E402
    accuracy, dump_json, load_checkpoint, resolve_device, save_checkpoint,
    set_seed, setup_logger, timestamp, write_csv,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--out_dir", type=str, default="outputs/phase0")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--model", type=str, default="vgg9", choices=["vgg9", "resnet19"])
    p.add_argument("--num_steps", type=int, default=25)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--tau", type=float, default=20.0)
    p.add_argument("--v_threshold", type=float, default=1.0)
    p.add_argument("--width_scale", type=float, default=1.0,
                   help="channel width multiplier (0.5 = half channels, saves memory)")
    p.add_argument("--use_checkpoint", action="store_true",
                   help="gradient checkpointing on conv segments (saves ~T* memory)")
    p.add_argument("--lambda_mse", type=float, default=0.05)
    p.add_argument("--mse_target", type=str, default="const", choices=["const", "onehot"])
    p.add_argument("--lr_decay", type=str, default="step", choices=["step", "cosine", "cosine_step", "none"])
    p.add_argument("--checkpoint_every", type=int, default=0, help="0 = off")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (each fork ~0.5-1GB; keep 0 on small RAM)")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--max_batches_per_epoch", type=int, default=0, help="0 = all")
    return p.parse_args()


def adjust_lr(optimizer, epoch, max_epoch):
    if epoch == int(max_epoch * 0.5) or epoch == int(max_epoch * 0.7) or epoch == int(max_epoch * 0.9):
        for pg in optimizer.param_groups:
            pg["lr"] /= 10
            print(f"  [lr decay] epoch {epoch}: lr -> {pg['lr']:.5f}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    run_name = args.run_name or f"{args.model}_T{args.num_steps}_lr{args.lr}_e{args.epochs}_s{args.seed}"
    logger = setup_logger("pretrain", args.out_dir, run_name)
    logger.info(f"args: {vars(args)}")

    train_loader, test_loader = get_cifar10_loaders(
        args.data_root, args.batch_size, args.num_workers,
        train_transform=cifar10_transform(train=True),
        test_transform=cifar10_transform(train=False),
    )
    model = build_model(args.model, num_steps=args.num_steps,
                        num_classes=args.num_classes, tau=args.tau,
                        v_threshold=args.v_threshold,
                        width_scale=args.width_scale,
                        use_checkpoint=args.use_checkpoint).to(device)
    logger.info(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    start_epoch = 1
    if args.resume:
        ckpt = load_checkpoint(args.resume, device)
        model.load_state_dict(ckpt["state_dict"])
        start_epoch = ckpt.get("epoch", 1) + 1
        logger.info(f"resumed from {args.resume} (epoch {start_epoch})")

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    best_acc = -1.0
    metrics = []

    for epoch in range(start_epoch, args.epochs + 1):
        if args.lr_decay == "step":
            adjust_lr(optimizer, epoch, args.epochs)
        elif args.lr_decay == "cosine":
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * 0.5 * (1 + torch.cos(torch.tensor(
                    (epoch - 1) / args.epochs * 3.14159)).item())
        elif args.lr_decay == "cosine_step":
            # cosine 主体 + 在 60% 和 85% 处叠加 step 跳降(×0.5)
            base = args.lr * 0.5 * (1 + torch.cos(torch.tensor(
                (epoch - 1) / args.epochs * 3.14159)).item())
            factor = 1.0
            if epoch >= int(args.epochs * 0.60):
                factor *= 0.5
            if epoch >= int(args.epochs * 0.85):
                factor *= 0.5
            for pg in optimizer.param_groups:
                pg["lr"] = base * factor

        model.train()
        t0 = time.time()
        train_loss, train_acc, n_b = 0.0, 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            if args.max_batches_per_epoch and i >= args.max_batches_per_epoch:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            functional.reset_net(model)
            logits, logits_t = model(x, return_logits_t=True)
            loss = tet_loss(logits_t, y, lambda_mse=args.lambda_mse,
                            v_threshold=args.v_threshold, mse_target=args.mse_target)
            loss.backward()
            optimizer.step()
            functional.reset_net(model)
            acc = accuracy(logits, y)[0]
            train_loss += loss.item()
            train_acc += acc
            n_b += 1
        train_loss /= max(n_b, 1)
        train_acc /= max(n_b, 1)

        # clean test accuracy
        model.eval()
        test_loss, test_acc, n_b = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                functional.reset_net(model)
                logits = model(x)
                test_loss += float(nn.functional.cross_entropy(logits, y).item())
                test_acc += accuracy(logits, y)[0]
                n_b += 1
                functional.reset_net(model)
        test_loss /= max(n_b, 1)
        test_acc /= max(n_b, 1)

        lr_now = optimizer.param_groups[0]["lr"]
        metrics.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "train_acc": round(train_acc, 3), "test_loss": round(test_loss, 4),
                        "test_acc": round(test_acc, 3), "lr": lr_now})
        logger.info(
            f"epoch {epoch:3d}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.2f}% | "
            f"test loss {test_loss:.4f} acc {test_acc:.2f}% | lr {lr_now:.5f} | {time.time()-t0:.1f}s"
        )

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
        ckpt_extra = {"args": vars(args), "epoch": epoch, "test_acc": test_acc}
        save_checkpoint(f"{args.out_dir}/{run_name}_last.pt", model.state_dict(), ckpt_extra)
        if is_best:
            save_checkpoint(f"{args.out_dir}/{run_name}_best.pt", model.state_dict(), ckpt_extra)
        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            save_checkpoint(f"{args.out_dir}/{run_name}_epoch{epoch}.pt",
                            model.state_dict(), ckpt_extra)

    write_csv(f"{args.out_dir}/{run_name}_metrics.csv", metrics)
    dump_json(f"{args.out_dir}/{run_name}_config.json", vars(args))
    logger.info(f"best clean test acc: {best_acc:.2f}%  "
                f"checkpoints in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
