# Reproducibility notes

## Final default

- Dataset: CIFAR-10-C, all 15 corruptions
- SNN: VGG9 + BNTT + LIF, separately pretrained for each temporal horizon
- Default stress test: `T=12`, batch `2`, severity `5`
- Online stream: 120 samples/corruption, 60 updates at batch 2
- Adapter: channel-wise affine after `pool3`, 256 scales + 256 biases
- Objective: negative top-two probability margin
- ZO radius: `1e-2`; learning rate: `1e-3`
- SD: density `0.5`, unrescaled; MD: two dense directions

## Randomness

For every batch, prediction and all positive/negative objective probes reuse the same Poisson encoding seed. ZO direction randomness remains independent. This common-random-number protocol prevents encoding noise from dominating the finite difference.

## Evidence tiers

Phase 1--3 are candidate screening. Phase 4 is the paper-valid fixed-sample/common-randomness protocol. Do not treat candidate-selection streams as independent confirmatory evidence.
