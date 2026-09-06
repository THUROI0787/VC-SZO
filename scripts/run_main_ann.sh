#!/usr/bin/env bash
set -euo pipefail
python phase4/run_phase4_ann.py \
  --gpu "${GPU:-0}" --tag main_ann_b2_l5 \
  --T 0 --levels 5 --batches 2 --seeds 42,215,3407 \
  --methods source,zo_noop,m307,m301,tent,memo,bn_stats,bp_margin,bp_entropy \
  --corruptions all --max-samples 120 --data-root "${DATA_ROOT:-data}"
