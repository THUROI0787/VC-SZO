#!/bin/bash
# =========================================================================== #
# Phase 3: ANN Baseline (T=0) — 全量 sweep
# =========================================================================== #
# 遍历所有 level (1/3/5) × batch (1/2/4/8/32/128) × seed (42/3407/215)
# 共 3 × 6 × 3 = 54 个 setting
# 评测 4 baselines + 7 ZO candidates (跳过 SNN 特有的 m604)
#
# Usage:
#   bash phase3/run_phase3_ann.sh --gpu 0              # 完整运行
#   bash phase3/run_phase3_ann.sh --gpu 0 --quick       # 快速测试
# =========================================================================== #

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

DATA_ROOT="${DATA_ROOT:-data}"
BASE_OUT_DIR="${BASE_OUT_DIR:-outputs/phase3_ann}"
GPU="${GPU:--1}"
QUICK="${QUICK:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_root) DATA_ROOT="$2"; shift 2 ;;
        --base_out_dir) BASE_OUT_DIR="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --quick) QUICK="--quick"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ANN checkpoint path (Config B: lr=0.3, wd=5e-5, cosine 160 epochs)
CKPT="checkpoints/ann/vgg9_cifar10_best.pt"

echo "=========================================="
echo "Phase 3: ANN Baseline (T=0) — 全量 sweep"
echo "  Checkpoint: $CKPT"
echo "  Data root:  $DATA_ROOT"
echo "  Base out:   $BASE_OUT_DIR"
echo "  GPU:        $GPU"
echo "  Quick:      ${QUICK:-no}"
echo "=========================================="

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    echo "Please train ANN VGG9 first with:"
    echo "  python phase0/train_ann_vgg9.py --config B --epochs 160 --batch_size 128"
    exit 1
fi

mkdir -p "$BASE_OUT_DIR/logs"

GPU_ARG=""
if [ "$GPU" -ge 0 ]; then
    GPU_ARG="--gpu $GPU"
fi

python phase3/run_phase3_sweep_ann.py \
    --pretrained "$CKPT" \
    --data_root "$DATA_ROOT" \
    --out_dir "$BASE_OUT_DIR" \
    $GPU_ARG \
    $QUICK \
    2>&1 | tee "${BASE_OUT_DIR}/run_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=========================================="
echo "Phase 3 ANN Complete!"
echo "=========================================="
echo "Results in: $BASE_OUT_DIR"
echo "Settings: $(ls -d $BASE_OUT_DIR/level*/ 2>/dev/null | wc -l) completed"