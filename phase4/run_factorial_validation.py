#!/usr/bin/env python3
"""Wait for the frozen factorial scan, select once, then validate out of sample.

The development data use offset 1560 and seeds 17/53/97.  Validation uses four
disjoint offsets and seeds 23/61/109.  This launcher intentionally has no code
path for changing the selection after validation results are observed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
TAGS = (
    "factor_dev_T12_lr3e4", "factor_dev_T12_lr1e3",
    "factor_dev_T25_lr3e4", "factor_dev_T25_lr1e3",
)
OFFSETS_BY_GPU = {0: (2160, 2400), 1: (2280, 2520)}
SEEDS = "23,61,109"
METHODS = "source,zo_noop,m307,m301,bp_margin,tent,memo,bn_stats"


def complete() -> bool:
    return all(len(list((ROOT / "outputs" / "phase4" / tag).glob(
        "T*/level*_batch*_seed*/*/*.json"))) == 96 for tag in TAGS)


def main() -> None:
    while not complete():
        counts = {
            tag: len(list((ROOT / "outputs" / "phase4" / tag).glob(
                "T*/level*_batch*_seed*/*/*.json"))) for tag in TAGS
        }
        print(f"waiting for development scan: {counts}", flush=True)
        time.sleep(30)

    subprocess.run([str(PYTHON), "phase4/select_factorial.py"], cwd=ROOT, check=True)
    selection = json.loads((ROOT / "outputs" / "phase4" / "analysis" /
                            "factorial_selection.json").read_text())["selected"]
    T = int(selection["T"])
    batch = int(selection["batch"])
    level = int(selection["level"])
    lr = float(selection["lr"])
    lr_key = "lr3e4" if abs(lr - 3e-4) < 1e-12 else "lr1e3"
    setting_key = f"T{T}_b{batch}_L{level}_{lr_key}"
    print(f"frozen setting: {setting_key}; beginning fresh validation", flush=True)

    log_dir = ROOT / "outputs" / "phase4"
    processes: list[tuple[subprocess.Popen, object, str]] = []
    for gpu, offsets in OFFSETS_BY_GPU.items():
        log_path = log_dir / f"factor_validation_gpu{gpu}.log"
        log = log_path.open("w")
        commands = []
        for offset in offsets:
            tag = f"factor_val_{setting_key}_w{offset}"
            commands.append([
                str(PYTHON), "phase4/run_phase4_snn.py",
                "--gpu", str(gpu), "--tag", tag,
                "--T", str(T), "--levels", str(level),
                "--batches", str(batch), "--seeds", SEEDS,
                "--methods", METHODS, "--corruptions", "blur",
                "--max-samples", "120",
            ])
        # One sequential worker per GPU; the two workers run concurrently.
        code = (
            "import os,subprocess,sys; cmds=" + repr(commands) + "; "
            "[subprocess.run(c,cwd=" + repr(str(ROOT)) + ",env=os.environ,check=True) for c in cmds]"
        )
        env = os.environ.copy()
        env["P4_ZO_LR"] = str(lr)
        # Each command sets its own held-out sample offset through this wrapper.
        wrapper_code = (
            "import os,subprocess; items=" + repr(list(zip(offsets, commands))) + "; "
            "[(os.environ.__setitem__('P4_SAMPLE_OFFSET',str(o)),"
            "subprocess.run(c,cwd=" + repr(str(ROOT)) + ",env=os.environ,check=True)) "
            "for o,c in items]"
        )
        proc = subprocess.Popen([sys.executable, "-c", wrapper_code], cwd=ROOT,
                                env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append((proc, log, str(log_path)))

    failed = []
    for proc, log, path in processes:
        rc = proc.wait()
        log.close()
        if rc:
            failed.append((path, rc))
    if failed:
        raise SystemExit(f"validation worker failure(s): {failed}")
    print("factorial held-out validation complete", flush=True)


if __name__ == "__main__":
    main()
