#!/bin/bash
# =========================================================================== #
# Phase 3: SNN+ZO Candidate 大规模验证
# =========================================================================== #
# 在多种 setting(level×batch×seed)下评测 4 baselines + 8 SNN+ZO candidates
#
# 运行策略:
#   1. 按 T 值拆分(每个 T 独立运行)
#   2. 每个 T 内按难度排序: level 1→3→5, batch 1→4→8→32→128
#   3. 单点故障隔离: 一个 setting 失败不影响后续
#
# Usage:
#   bash phase3/run_phase3.sh --gpu 0                    # 完整运行(所有T)
#   bash phase3/run_phase3.sh --gpu 0 --T 6              # 仅T=6
#   bash phase3/run_phase3.sh --gpu 0 --T 4,6,12,25      # 指定多个T
#   bash phase3/run_phase3.sh --gpu 0 --quick             # 快速测试
#   bash phase3/run_phase3.sh --gpu 1 --T 25 --reverse    # 反向顺序(第二张卡)
# =========================================================================== #

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

DATA_ROOT="${DATA_ROOT:-data}"
BASE_OUT_DIR="${BASE_OUT_DIR:-outputs/phase3}"
GPU="${GPU:--1}"
QUICK="${QUICK:-}"
REVERSE="${REVERSE:-}"
T_VALUES="${T_VALUES:-4,6,12,25}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_root) DATA_ROOT="$2"; shift 2 ;;
        --base_out_dir) BASE_OUT_DIR="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --T) T_VALUES="$2"; shift 2 ;;
        --quick) QUICK="--quick"; shift ;;
        --reverse) REVERSE="--reverse"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Checkpoint paths (same as Phase 2)
declare -A CKPT_MAP
CKPT_MAP[4]="checkpoints/snn/vgg9_t4_best.pt"
CKPT_MAP[6]="checkpoints/snn/vgg9_t6_best.pt"
CKPT_MAP[12]="checkpoints/snn/vgg9_t12_best.pt"
CKPT_MAP[25]="checkpoints/snn/vgg9_t25_best.pt"

IFS=',' read -ra T_ARRAY <<< "$T_VALUES"

echo "=========================================="
echo "Phase 3: SNN+ZO Candidate Validation"
echo "  T values:   ${T_ARRAY[*]}"
echo "  Data root:  $DATA_ROOT"
echo "  Base out:   $BASE_OUT_DIR"
echo "  GPU:        $GPU"
echo "  Quick:      ${QUICK:-no}"
echo "  Reverse:    ${REVERSE:-no}"
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
    echo "  T=${T}"
    echo "  Checkpoint: $CKPT"
    echo "  Output:     $OUT_DIR"
    echo "=========================================="

    if [ ! -f "$CKPT" ]; then
        echo "  WARNING: Checkpoint not found, skipping T=${T}"
        echo "  Expected: $CKPT"
        continue
    fi

    mkdir -p "$OUT_DIR/logs"

    # Run phase3 sweep for this T
    python phase3/run_phase3_sweep.py \
        --pretrained "$CKPT" \
        --data_root "$DATA_ROOT" \
        --out_dir "$OUT_DIR" \
        --T "$T" \
        $GPU_ARG \
        $QUICK \
        $REVERSE \
        2>&1 | tee "${OUT_DIR}/run_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "  T=${T} done! Results in $OUT_DIR"
    echo "  Settings: $(ls -d $OUT_DIR/level*/ 2>/dev/null | wc -l) completed"
done

echo ""
echo "=========================================="
echo "Phase 3 Complete!"
echo "=========================================="
for T in "${T_ARRAY[@]}"; do
    OUT_DIR="${BASE_OUT_DIR}/T${T}"
    if [ -d "$OUT_DIR" ]; then
        N=$(ls -d "$OUT_DIR"/level*/ 2>/dev/null | wc -l)
        echo "  T=${T}: $N settings completed"
    fi
done