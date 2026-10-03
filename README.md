# VC-SZO: Forward-Only Test-Time Adaptation for Spiking Neural Networks

> **Accepted as a poster at the NeurIPS 2026 Workshop on On-Device Intelligence: Foundation Models under Real-World Constraints.**
> Camera-ready paper: coming soon.

VC-SZO adapts a pretrained spiking neural network (SNN) at test time using forward evaluations only. It freezes the backbone and updates a 512-parameter channel-wise affine adapter through symmetric zeroth-order probes with shared spike-encoding randomness.

## Status

The accepted paper establishes a backward-free, memory-efficient update mechanism; it does not claim a statistically clear accuracy gain over the frozen source model on held-out streams.

- VC-SZO matches the frozen model and a comparable local-BP update in the tested CIFAR-10-C streams.
- Peak memory is 17x lower than the full-temporal-graph gradient baseline and 127x lower than the augmentation-based baseline in the reported audit.
- Sparse and multi-direction variants expose different variance/query operating points.

## Authors

- Ruoyu Zhao — City University of Hong Kong (corresponding author)
- Yuting Chen — Georgia Institute of Technology
- Jiaqi Wu — City University of Hong Kong
- Luziwei Leng — BrainGalaxy

## Method

`vcszo_sd` uses one unrescaled Bernoulli-masked Gaussian direction with density 0.5, requiring two perturbation forwards per update.

`vcszo_md` averages two dense Gaussian directions, requiring four perturbation forwards per update.

Both variants use the negative top-two probability margin, the same late channel adapter, prequential prediction, and common Poisson randomness for each symmetric comparison.

## Repository layout

```text
tta_snn_zo/       core models, adapters, ZO optimizer, and TTA baselines
experiments/      paper configurations and fixed-sample evaluation protocol
scripts/          checkpoint validation and main experiment launcher
checkpoints/snn/  four paper SNN checkpoints tracked with Git LFS
```

## Setup

```bash
git clone https://github.com/THUROI0787/VC-SZO.git
cd VC-SZO
conda create -n vcszo python=3.11 -y
conda activate vcszo
pip install -r requirements.txt
git lfs pull
python scripts/smoke_test.py
```

Download the official CIFAR-10-C archive from [Zenodo](https://zenodo.org/records/2535967) and extract it under `data/`:

```text
data/CIFAR-10-C/gaussian_noise.npy
...
data/CIFAR-10-C/labels.npy
```

## Run the paper protocol

The main T=12, batch=2, severity-5 grid is available through:

```bash
GPU=0 DATA_ROOT=data bash scripts/run_main_snn.sh
```

For a small end-to-end check:

```bash
python experiments/run_snn.py \
  --gpu 0 --tag smoke --T 12 --levels 5 --batches 2 --seeds 42 \
  --methods source,vcszo_sd,vcszo_md --corruptions defocus_blur \
  --max-samples 8 --data-root data
```

Results are written atomically under `outputs/`, which is ignored by Git.

## Checkpoints

| Horizon | Clean CIFAR-10 accuracy | File |
|---:|---:|---|
| 4 | 86.47 | `checkpoints/snn/vgg9_t4_best.pt` |
| 6 | 87.62 | `checkpoints/snn/vgg9_t6_best.pt` |
| 12 | 89.39 | `checkpoints/snn/vgg9_t12_best.pt` |
| 25 | 89.81 | `checkpoints/snn/vgg9_t25_best.pt` |

Exact byte sizes and SHA-256 hashes are recorded in `checkpoints/manifest.json`.

## Scope

- Predictions are made before each online update; labels are never used for adaptation.
- Held-out experiments support accuracy preservation, not a consistent improvement over Source.
- The dense-ZO ablation is query-matched, not matched for perturbation energy or empirical update norm.
- Dataset files, generated outputs, and manuscript source are not distributed in this repository.

## Citation

The camera-ready citation will be added when the workshop proceedings entry is available. Author metadata is provided in `CITATION.cff`.

## License

No software license has been selected yet. Please contact the authors before reuse beyond inspection and reproducibility review.
