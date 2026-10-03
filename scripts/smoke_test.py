from pathlib import Path
import hashlib
import json

import torch
from tta_snn_zo.models import build_model

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'checkpoints/manifest.json').read_text())
for t in (4,6,12,25):
    path=root/f'checkpoints/snn/vgg9_t{t}_best.pt'
    record=manifest[f'snn/vgg9_t{t}_best.pt']
    assert path.stat().st_size == record['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    model=build_model('vgg9',num_steps=t,num_classes=10)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in ckpt['state_dict'].items()})
    print(f'SNN T={t}: checksum and load passed; clean_acc={record["clean_test_acc"]:.2f}%')
print('all checkpoint smoke tests passed')
