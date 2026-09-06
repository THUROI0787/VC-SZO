"""Empirical VC-SZO estimator diagnostic against a surrogate-BP reference."""
from __future__ import annotations

import argparse
import csv
import gc
import os
from pathlib import Path

_gpu_parser = argparse.ArgumentParser(add_help=False)
_gpu_parser.add_argument("--gpu", type=int, default=-1)
_gpu_args, _ = _gpu_parser.parse_known_args()
if _gpu_args.gpu >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_args.gpu)

import torch
import torch.nn.functional as F

from phase3.candidates_phase3 import PHASE3_CANDIDATES
from phase4.engines import AdapterBPEngine
from phase4.run_phase4_snn import build_model
from tta_snn_zo.data import get_cifar10c_loader
from tta_snn_zo.launch import resolve_cifar10c
from tta_snn_zo.tta import ZOTTAEngine
from tta_snn_zo.utils import set_seed
from types import SimpleNamespace

BASE = next(c["engine"] for c in PHASE3_CANDIDATES if c["id"] == "m301")
CONFIGS = (("MD-K1", 1, 1.0), ("MD-K2", 2, 1.0), ("MD-K4", 4, 1.0),
           ("SD-q50", 1, 0.5), ("SD-q25", 1, 0.25))

def flatten(tensors):
    return torch.cat([tensor.detach().reshape(-1) for tensor in tensors])

def bp_reference(x, time_steps, device, seed):
    model = build_model(time_steps, device, logger=None)
    engine = AdapterBPEngine(model, objective="margin", seed=seed, device=device)
    torch.manual_seed(seed + 1_000_000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 1_000_000)
    engine._begin_forward()
    loss = engine._loss(engine.model(x))
    grads = torch.autograd.grad(loss, engine.params)
    vector = flatten(grads)
    del engine, model
    gc.collect()
    torch.cuda.empty_cache()
    return vector

def zo_replicates(x, time_steps, device, seed, samples, sparsity, repetitions):
    model = build_model(time_steps, device, logger=None)
    config = dict(BASE)
    config.update(num_samples=samples, block_sparsity=sparsity, lr=0.0,
                  weight_decay=0.0, rollback_collapse=False)
    engine = ZOTTAEngine(model, device=device, seed=seed, **config)
    enc_seed = seed + 1_000_000
    estimates, projected = [], []
    objective = lambda: engine._objective(x, enc_seed)
    for _ in range(repetitions):
        estimate, proj = engine.zo.estimate(objective, num_samples=samples)
        estimates.append(flatten(estimate).cpu())
        projected.append(proj)
    del engine, model
    gc.collect()
    torch.cuda.empty_cache()
    return torch.stack(estimates), torch.tensor(projected)

def parse_csv(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]

def main():
    parser = argparse.ArgumentParser(parents=[_gpu_parser])
    parser.add_argument("--T", type=int, default=12)
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repetitions", type=int, default=24)
    parser.add_argument("--offsets", default="0,240")
    parser.add_argument("--corruptions", default="defocus_blur,glass_blur")
    parser.add_argument("--out", default="outputs/phase4/variance_diagnostic/gpu.csv")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root, _ = resolve_cifar10c(SimpleNamespace(data_root="data", data_mode="auto", level=args.level))
    rows = []
    for offset in parse_csv(args.offsets, int):
        os.environ["P4_SAMPLE_OFFSET"] = str(offset)
        for corruption in parse_csv(args.corruptions):
            set_seed(args.seed)
            loader = get_cifar10c_loader(root, corruption, args.batch, 0, args.level)
            x, _ = next(iter(loader))
            x = x.to(device)
            reference = bp_reference(x, args.T, device, args.seed).cpu()
            reference_norm = reference.norm().clamp_min(1e-12)
            for name, samples, sparsity in CONFIGS:
                estimates, projected = zo_replicates(
                    x, args.T, device, args.seed, samples, sparsity, args.repetitions)
                mean = estimates.mean(0)
                residual = estimates - mean
                cosines = F.cosine_similarity(estimates, reference.unsqueeze(0), dim=1)
                rows.append({"T": args.T, "level": args.level, "batch": args.batch,
                    "seed": args.seed, "offset": offset, "corruption": corruption,
                    "config": name, "directions": samples, "sparsity": sparsity,
                    "repetitions": args.repetitions, "dimension": reference.numel(),
                    "bp_norm": reference_norm.item(), "mean_cosine": cosines.mean().item(),
                    "std_cosine": cosines.std().item(),
                    "mean_estimate_cosine": F.cosine_similarity(mean, reference, dim=0).item(),
                    "mean_norm_ratio": (mean.norm() / reference_norm).item(),
                    "trace_variance": residual.square().sum(1).mean().item(),
                    "snr": (mean.norm() / residual.square().sum(1).mean().sqrt().clamp_min(1e-12)).item(),
                    "projected_std": projected.std().item()})
                print(rows[-1], flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(out)

if __name__ == "__main__":
    main()
