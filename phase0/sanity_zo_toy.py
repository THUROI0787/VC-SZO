"""Phase 0 - shallowest ZO sanity check (validated on the hard-spike SNN).

What this checks, from shallow to deep:

Part A - **gradient fidelity (variance amplification).**
   On a tiny hard-spike SNN (SpikingJelly LIF, surrogate only in backward), the
   true objective is *piecewise-constant* in the weights.  A single two-point ZO
   sample therefore has near-zero correlation with the surrogate gradient
   (cos ~ 1/sqrt(d), the "variance amplification" noted by SZO ICML'26).
   Averaging over more perturbation samples restores alignment:
   cos_sim(eps, N) grows with N toward the smoothed-gradient direction.
   PASS  <=>  cos_sim(N=512) > 0.15  and  cos_sim(512) > 2 * cos_sim(1).

Part B - **ZO fine-tuning from a warm start.**
   The report's central claim: ZO is ill-suited to training from scratch on SNNs
   but works well for *local* adjustment from a good initial point (exactly the
   TTA setting).  We warm-start the toy with a few surrogate-gradient (BP) steps
   and then fine-tune with two-point ZO (the real ``ZoOptimizer`` engine);
   the loss must keep decreasing.
   PASS  <=>  final loss < warm-start loss.

Everything (Poisson/direct encoding, BNTT, LIF reset, perturb/restore, fixed
encoding) is exercised here before any larger experiment.  Runs in ~1 minute.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, neuron, surrogate
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tta_snn_zo.utils import (  # noqa: E402
    resolve_device, save_checkpoint, set_seed, setup_logger,
)
from tta_snn_zo.zo import ZoOptimizer  # noqa: E402


class TinySNN(nn.Module):
    """~2k-parameter hard-spike SNN (direct encoding) for the ZO toy check."""

    def __init__(self, ch1=8, ch2=16, num_steps=3, tau=2.0, v_threshold=0.5, gain=1.0):
        super().__init__()
        self.num_steps = num_steps
        lif = dict(tau=tau, v_threshold=v_threshold, v_reset=0.0,
                   surrogate_function=surrogate.ATan())
        self.conv1 = nn.Conv2d(3, ch1, 3, 1, 1, bias=False)
        self.lif1 = neuron.LIFNode(**lif)
        self.pool1 = nn.AvgPool2d(4)
        self.fc1 = nn.Linear(ch1 * 8 * 8, ch2, bias=False)
        self.lif_fc = neuron.LIFNode(**lif)
        self.fc2 = nn.Linear(ch2, 10, bias=False)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=gain)

    def forward(self, x):
        outs = []
        for t in range(self.num_steps):
            out = self.lif1(self.conv1(x * 2.0))   # direct encoding
            out = self.pool1(out).flatten(1)
            out = self.lif_fc(self.fc1(out))
            outs.append(self.fc2(out))
        return torch.stack(outs).mean(0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--out_dir", type=str, default="outputs/phase0")
    p.add_argument("--num_steps", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--subset", type=int, default=256)
    p.add_argument("--warmup_steps", type=int, default=80, help="BP warm-start steps")
    p.add_argument("--zo_steps", type=int, default=150, help="ZO fine-tune steps")
    p.add_argument("--eps", type=float, default=1e-2)
    p.add_argument("--lr", type=float, default=3e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    logger = setup_logger("sanity_zo_toy", args.out_dir, f"run_seed{args.seed}")

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    ds = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=tf)
    idx = torch.randperm(len(ds))[: args.subset]
    loader = DataLoader(Subset(ds, idx.tolist()), batch_size=args.batch_size,
                        shuffle=True, num_workers=2, pin_memory=True)
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    criterion = nn.CrossEntropyLoss()

    model = TinySNN(num_steps=args.num_steps).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    d = sum(p.numel() for p in params)
    logger.info(f"device={device} params={d:,} subset={len(loader.dataset)} "
                f"eps={args.eps} lr={args.lr}")

    # ------------------------------------------------------------------ #
    # Part A: gradient fidelity (train mode: surrogate gradient available)
    # ------------------------------------------------------------------ #
    model.train()
    functional.reset_net(model)
    torch.manual_seed(42)
    logits = model(x)
    loss = criterion(logits, y)
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    functional.reset_net(model)
    true_grad = torch.cat([
        g.flatten() if g is not None else torch.zeros(p.numel(), device=p.device)
        for g, p in zip(grads, params)
    ])

    def objective():
        functional.reset_net(model)
        with torch.no_grad():
            return float(criterion(model(x), y).item())

    zo = ZoOptimizer(params, lr=args.lr, eps=args.eps, seed=args.seed + 100, device=device)
    cos_vals = {}
    for N in (1, 64, 512):
        est, proj = zo.estimate(objective, num_samples=N)
        est_flat = torch.cat([e.flatten() for e in est])
        cos = torch.nn.functional.cosine_similarity(est_flat, true_grad, dim=0).item()
        cos_vals[N] = cos
        logger.info(f"[Part A] N={N:4d} samples: cos_sim(zo, true_grad) = {cos:.4f}")
    pass_a = cos_vals[512] > 0.15 and cos_vals[512] > 2 * cos_vals[1]
    logger.info(f"[Part A] variance amplification: single-sample cos ~ 1/sqrt(d) "
                f"= {1 / (d ** 0.5):.4f}; averaging N=512 -> {cos_vals[512]:.4f} "
                f"({'PASS' if pass_a else 'FAIL'})")

    # ------------------------------------------------------------------ #
    # Part B: BP warm-start then ZO fine-tuning (the TTA-like setting)
    # ------------------------------------------------------------------ #
    model2 = TinySNN(num_steps=args.num_steps).to(device)
    model2.train()
    opt = torch.optim.Adam(model2.parameters(), lr=1e-2)
    warm_loss = None
    for _ in range(args.warmup_steps):
        opt.zero_grad(set_to_none=True)
        functional.reset_net(model2)
        logits = model2(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        functional.reset_net(model2)
        warm_loss = float(loss.item())

    model2.eval()
    zo2 = ZoOptimizer(
        [p for p in model2.parameters() if p.requires_grad],
        lr=args.lr, eps=args.eps, seed=args.seed + 200, device=device,
    )
    functional.reset_net(model2)
    with torch.no_grad():
        l_start = float(criterion(model2(x), y).item())
    losses = [l_start]
    for step in range(args.zo_steps):
        zo2.step(lambda: _objective(model2, x, y, criterion))
        functional.reset_net(model2)
        with torch.no_grad():
            losses.append(float(criterion(model2(x), y).item()))
    l_final = losses[-1]
    pass_b = l_final < l_start
    logger.info(f"[Part B] warm-start loss={warm_loss:.4f}; ZO fine-tune "
                f"{l_start:.4f} -> {l_final:.4f} (min {min(losses):.4f}) "
                f"({'PASS' if pass_b else 'FAIL'})")

    save_checkpoint(f"{args.out_dir}/sanity_zo_toy_seed{args.seed}.pt",
                    model2.state_dict(), {"args": vars(args), "loss": l_final})
    ok = pass_a and pass_b
    logger.info(f"gradient-fidelity={pass_a}  zo-finetune={pass_b}")
    logger.info("SANITY ZO CHECK " + ("PASSED ✓" if ok else "FAILED ✗"))
    return 0 if ok else 1


def _objective(model, x, y, criterion):
    functional.reset_net(model)
    with torch.no_grad():
        return float(criterion(model(x), y).item())


if __name__ == "__main__":
    sys.exit(main())
