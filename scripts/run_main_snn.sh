#!/usr/bin/env bash
set -euo pipefail
python experiments/run_snn.py \
  --gpu "${GPU:-0}" --tag main_t12_b2_l5 \
  --T 12 --levels 5 --batches 2 --seeds 42,215,3407 \
  --methods source,zo_noop,vcszo_sd,vcszo_md,tent,memo,bn_stats,bp_margin,bp_entropy \
  --corruptions all --max-samples 120 --data-root "${DATA_ROOT:-data}"
