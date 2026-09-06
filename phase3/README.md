# Phase 3: SNN+ZO Candidate 大规模验证

## 概述

Phase 3 基于 Phase 2 的结果,选取 8 个最有希望的 SNN+ZO candidates,在多种 setting 和 seed 下进行系统验证。

## 选取的 Candidates (8个)

| ID | 方法 | 选择理由 |
|----|------|---------|
| m604 | Temporal adapter | SNN特有,在T25上表现最好(+1.11%) |
| m301 | Multi-sample ns=2 | 跨T一致性好,平均Δ最高 |
| m307 | Subspace 50% | 在T6上表现最好(+1.14%),方法多样性 |
| m306 | Momentum+ns=4 | 在T6/T25上都强 |
| m407 | Episodic reset | 不同机制,在T6上强 |
| m502 | Filtered entropy | 不同objective,在T6上强 |
| m709 | One-point eps=5e-2 | 计算高效,在T25上强 |
| m202 | Medium eps=3e-2 | 在T25上强 |

## 实验 Setting

### 维度
- **Level**: 1, 3, 5 (3种)
- **Batch size**: 1, 4, 8, 32, 128 (5种)
- **Seed**: 0, 1, 2, 3 (4种)
- **Total**: 3 × 5 × 4 = 60 个实验 setting

### 每个实验评测
- 4个 baselines: B0 Source-only, B1 TENT-BP, B2 MEMO, B3 BN Stats
- 8个 SNN+ZO candidates
- 15个 corruption tasks
- 每个 corruption 60 batches (max_batches=60)

### 随机性来源说明
1. **数据加载顺序**: CIFAR-10-C 的 DataLoader 默认 shuffle=False,所以数据顺序是固定的
2. **ZO 随机扰动**: ZoOptimizer 使用 `torch.Generator` + `seed` 参数控制,不同 seed 产生不同扰动方向
3. **MEMO 增强**: 随机裁剪/翻转/颜色抖动,受 `torch.manual_seed` 影响
4. **模型初始化**: 预训练 checkpoint 固定,不受 seed 影响
5. **Dropout/BN**: 模型 eval 模式,无随机性

**结论**: 不同 seed 主要影响 ZO 扰动方向和 MEMO 增强,这正是我们想要评估的统计变异性。

## 文件结构

```
phase3/
├── README.md              # 本文件
├── run_phase3_sweep.py    # 主评测脚本
├── candidates_phase3.py   # 8个候选的定义
└── run_phase3.sh          # 一键运行脚本
```

## 使用方法

### 1. 将文件复制到项目根目录

```bash
cp phase3/run_phase3_sweep.py ./
cp phase3/candidates_phase3.py ./
cp phase3/run_phase3.sh ./
```

### 2. 运行

```bash
# 完整运行(60个setting,按难度排序)
bash run_phase3.sh --gpu 0

# 指定T值
bash run_phase3.sh --gpu 0 --T 6

# 快速测试(仅level=3, batch=8, seed=0)
bash run_phase3.sh --gpu 0 --quick

# 指定多个T
bash run_phase3.sh --gpu 0 --T 4,6,12,25
```

### 3. 输出

```
outputs/phase3/
├── T4/
│   ├── level1_batch1_seed0/
│   │   ├── summary.csv
│   │   └── summary.json
│   ├── level1_batch1_seed1/
│   │   ├── summary.csv
│   │   └── summary.json
│   └── ...
├── T6/
├── T12/
└── T25/
```

每个 setting 只保留 summary.csv 和 summary.json,没有单个 corruption 的 log。

## 鲁棒性设计

1. **OOM 保护**: 每个方法单独 try-except,OOM 时 skip 并记录
2. **MEMO OOM 保护**: batch>=32 时自动降低 n_aug
3. **按难度排序**: batch=1 最先跑,batch=128 最后跑
4. **断点续跑**: 检测已有结果文件,跳过已完成 setting
5. **单点故障隔离**: 一个方法失败不影响其他方法