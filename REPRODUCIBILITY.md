# Reproducibility notes

## Paper default

- Backbone: VGG9 with BNTT and LIF neurons, pretrained at each temporal horizon
- Dataset: CIFAR-10-C, all 15 corruptions
- Main stress test: T=12, batch size 2, severity 5
- Online stream: 120 samples per corruption, 60 updates at batch size 2
- Adapter: channel-wise affine layer after `pool3` (256 scales and 256 biases)
- Objective: negative top-two probability margin
- ZO radius: 1e-2; learning rate: 1e-3
- SD: density 0.5, unrescaled
- MD: two dense directions

## Randomness

For every batch, prediction and all positive/negative objective probes reuse the same Poisson encoding seed. ZO direction randomness uses an independent generator. This common-random-number protocol prevents encoding noise from dominating the finite difference.

## Method identifiers

| Identifier | Description |
|---|---|
| `vcszo_sd` | sparse-direction VC-SZO |
| `vcszo_md` | two-direction averaged VC-SZO |
| `dense_margin` | one dense direction with the margin objective |
| `dense_entropy` | one dense direction with entropy minimization |
| `source` | frozen pretrained model |

## Evidence boundary

The held-out evaluations do not show a statistically clear improvement over the frozen Source baseline. The supported result is a forward-only update mechanism that preserves accuracy in the tested streams and avoids the collapse of several small-batch TTA baselines.

## Compute accounting

- `prediction_forwards` counts prequential predictions.
- `objective_forwards` counts additional update probes.
- `backward_calls` is reported separately and is zero for VC-SZO.
