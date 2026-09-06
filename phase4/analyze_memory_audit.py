"""Aggregate the isolated T=25, B=8 memory audit and activation-volume model."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path("outputs/phase4")
OUT = ROOT / "analysis"
METHODS = ["source", "m307", "bp_margin", "tent", "memo"]
LABELS = {"source": "Source", "m307": "VC-SZO-SD", "bp_margin": "Local BP",
          "tent": "TENT", "memo": "MEMO"}
rows=[]
for method in METHODS:
    tag=ROOT/f"memory_audit_T25_b8_{method}"
    files=list(tag.rglob("glass_blur.json"))
    if len(files)!=1:
        raise SystemExit(f"expected one warm glass_blur cell for {method}, found {len(files)}")
    r=json.loads(files[0].read_text())
    if r.get("status")!="ok": raise SystemExit(f"non-ok: {files[0]}")
    rows.append({"method":LABELS[method], "T":25, "batch":8,
                 "peak_allocated_mib":r["peak_allocated_mb"],
                 "peak_reserved_mib":r["peak_reserved_mb"], "time_s":r["time_s"],
                 "prediction_forwards":r.get("prediction_forwards") if method in {"source","m307","bp_margin"} else None,
                 "objective_forwards":r.get("objective_forwards") if method in {"source","m307","bp_margin"} else None,
                 "backward_calls":r.get("backward_calls") if method in {"source","m307","bp_margin"} else None})
df=pd.DataFrame(rows)
base=float(df.loc[df.method.eq("VC-SZO-SD"),"peak_allocated_mib"].iloc[0])
df["memory_vs_sd"]=df.peak_allocated_mib/base
# One float32 tensor at each post-BN/LIF resolution per time step.
per_sample_step=2*(64*32*32)+2*(128*16*16)+3*(256*8*8)+1024
trace=[]
for batch in (2,8):
    elements=25*batch*per_sample_step
    trace.append({"T":25,"batch":batch,"elements":elements,
                  "one_copy_trace_mib":elements*4/2**20,
                  "definition":"one float32 tensor at each of seven conv and one FC spike stages per time step"})
OUT.mkdir(parents=True,exist_ok=True)
df.to_csv(OUT/"memory_audit_T25_b8.csv",index=False,float_format="%.6f")
pd.DataFrame(trace).to_csv(OUT/"bptt_activation_volume.csv",index=False,float_format="%.6f")
print(df.to_string(index=False))
print(pd.DataFrame(trace).to_string(index=False))
