# VC-SZO: Forward-Only Test-Time Adaptation for Spiking Neural Networks

Research code for forward-only, small-batch test-time adaptation (TTA) of pretrained spiking neural networks. VC-SZO freezes the SNN backbone and updates a 512-parameter late channel-wise affine adapter using only symmetric zeroth-order (ZO) probes.

This private research repository contains code and pretrained checkpoints. The submission source, datasets, and bulk experiment outputs are intentionally excluded.

## Method map

- **VC-SZO-SD (`m307`)**: one Bernoulli-masked Gaussian direction, density `q=0.5`; two perturbation forwards per update.
- **VC-SZO-MD (`m301`)**: average of `K=2` dense Gaussian directions; four perturbation forwards per update.
- Both use the same top-two margin objective, common Poisson encoding randomness for positive/negative probes, and the same channel adapter after VGG9 `pool3`.

The repository also retains dense ZO, one-point, momentum, reset, filtered-objective, temporal-adapter, BP, TENT, MEMO, and BN candidates used in the Phase-1--3 elimination chain.

## Layout

```text
tta_snn_zo/        core SNN/ANN models, adapters, ZO optimizer, TTA baselines
phase0/             ANN/SNN pretraining and data preparation
phase1/             initial baselines and broad candidate definitions
phase2/             objective, variance, subspace, and ANN/SNN sweeps
phase3/             selected candidate sweeps across T/batch/severity/seed
phase4/             fixed-sample protocol, final runners, BP and diagnostics
checkpoints/        ANN and SNN-VGG9 checkpoints (Git LFS)
scripts/            concise reproduction entry points
```

## Checkpoints

| Backbone | Horizon | Clean CIFAR-10 accuracy | File |
|---|---:|---:|---|
| ANN-VGG9 | -- | 92.78 | `checkpoints/ann/vgg9_cifar10_best.pt` |
| SNN-VGG9+BNTT+LIF | 4 | 86.47 | `checkpoints/snn/vgg9_t4_best.pt` |
| SNN-VGG9+BNTT+LIF | 6 | 87.62 | `checkpoints/snn/vgg9_t6_best.pt` |
| SNN-VGG9+BNTT+LIF | 12 | 89.39 | `checkpoints/snn/vgg9_t12_best.pt` |
| SNN-VGG9+BNTT+LIF | 25 | 89.81 | `checkpoints/snn/vgg9_t25_best.pt` |

Exact epochs, byte sizes, and SHA-256 hashes are in `checkpoints/manifest.json`.

## Setup

```bash
conda create -n vcszo python=3.11 -y
conda activate vcszo
pip install -r requirements.txt
git lfs pull
python scripts/smoke_test.py
```

Download the official CIFAR-10-C archive from [Zenodo](https://zenodo.org/records/2535967) and extract it as:

```text
data/CIFAR-10-C/gaussian_noise.npy
...
data/CIFAR-10-C/labels.npy
```

CIFAR-10 clean data is loaded through `torchvision` under `data/`.

## Reproduce the main protocol

The paper-facing protocol uses prequential prediction, fixed sample counts, and common Poisson random numbers. To launch the complete T=12, batch=2, severity-5 grid:

```bash
GPU=0 DATA_ROOT=data bash scripts/run_main_snn.sh
```

ANN transfer comparison:

```bash
GPU=1 DATA_ROOT=data bash scripts/run_main_ann.sh
```

A smaller check can be run directly:

```bash
python phase4/run_phase4_snn.py \\
  --gpu 0 --tag smoke --T 12 --levels 5 --batches 2 --seeds 42 \\
  --methods source,m307,m301 --corruptions defocus_blur \\
  --max-samples 8 --data-root data
```

Outputs are written atomically under `outputs/` and are ignored by Git.

## Experimental phases

1. **Phase 0** trains matched ANN/SNN backbones.
2. **Phase 1** establishes Source/TENT/vanilla-ZO baselines and a broad candidate pool.
3. **Phase 2** tests objectives, perturbation radii, sparse/multi-direction estimators, drift controls, and ANN transfer.
4. **Phase 3** scans selected candidates across temporal horizon, batch size, corruption severity, and seed.
5. **Phase 4** uses fixed samples and common encoding randomness, separates local adapter BP from full temporal-graph adaptation, and records forward/backward/memory budgets.

## Notes

- Predictions are made before each online update; labels are never used for adaptation.
- `m307` and `m301` are alternative operating points, not simultaneous components.
- The local BP control detaches the adapter input and is not a full-BPTT memory baseline.
- No dataset or paper source is distributed in this repository.
