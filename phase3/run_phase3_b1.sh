#!/bin/bash
# =========================================================================== #
# Phase 3: SNN+ZO Candidate — batch=1 only
# =========================================================================== #
# 只跑 batch=1,遍历所有 T (4/6/12/25) × level (1/3/5) × seed (42/3407/215)
# 共 4 × 3 × 3 = 36 个 setting
#
# Usage:
#   bash phase3/run_phase3.sh --gpu 0              # 所有 T
#   bash phase3/run_phase3.sh --gpu 0 --T 6        # 仅 T=6
#   bash phase3/run_phase3.sh --gpu 0 --quick       # 快速测试
# =========================================================================== #

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

DATA_ROOT="${DATA_ROOT:-data}"
BASE_OUT_DIR="${BASE_OUT_DIR:-outputs/phase3}"
GPU="${GPU:--1}"
QUICK="${QUICK:-}"
T_VALUES="${T_VALUES:-4,6,12,25}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_root) DATA_ROOT="$2"; shift 2 ;;
        --base_out_dir) BASE_OUT_DIR="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --T) T_VALUES="$2"; shift 2 ;;
        --quick) QUICK="--quick"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

declare -A CKPT_MAP
CKPT_MAP[4]="checkpoints/snn/vgg9_t4_best.pt"
CKPT_MAP[6]="checkpoints/snn/vgg9_t6_best.pt"
CKPT_MAP[12]="checkpoints/snn/vgg9_t12_best.pt"
CKPT_MAP[25]="checkpoints/snn/vgg9_t25_best.pt"

IFS=',' read -ra T_ARRAY <<< "$T_VALUES"

echo "=========================================="
echo "Phase 3: SNN+ZO — batch=1 only"
echo "  T values:   ${T_ARRAY[*]}"
echo "  Data root:  $DATA_ROOT"
echo "  Base out:   $BASE_OUT_DIR"
echo "  GPU:        $GPU"
echo "  Quick:      ${QUICK:-no}"
echo "=========================================="

GPU_ARG=""
if [ "$GPU" -ge 0 ]; then
    GPU_ARG="--gpu $GPU"
fi

for T in "${T_ARRAY[@]}"; do
    CKPT="${CKPT_MAP[$T]}"
    OUT_DIR="${BASE_OUT_DIR}/T${T}"

    echo ""
    echo "=========================================="
    echo "  T=${T}  (batch=1 only)"
    echo "  Checkpoint: $CKPT"
    echo "  Output:     $OUT_DIR"
    echo "=========================================="

    if [ ! -f "$CKPT" ]; then
        echo "  WARNING: Checkpoint not found, skipping T=${T}"
        echo "  Expected: $CKPT"
        continue
    fi

    mkdir -p "$OUT_DIR/logs"

    python phase3/run_phase3_sweep.py \
        --pretrained "$CKPT" \
        --data_root "$DATA_ROOT" \
        --out_dir "$OUT_DIR" \
        --T "$T" \
        --batch 1 \
        $GPU_ARG \
        $QUICK \
        2>&1 | tee "${OUT_DIR}/run_batch1_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "  T=${T} batch=1 done! Results in $OUT_DIR"
done

echo ""
echo "=========================================="
echo "Phase 3 SNN batch=1 Complete!"
echo "=========================================="
for T in "${T_ARRAY[@]}"; do
    OUT_DIR="${BASE_OUT_DIR}/T${T}"
    if [ -d "$OUT_DIR" ]; then
        N=$(ls -d "$OUT_DIR"/level1_batch1_*/ 2>/dev/null | wc -l)
        echo "  T=${T}: $N settings completed"
    fi
done