from pathlib import Path
import json, torch
from tta_snn_zo.models import build_model
from phase0.train_ann_vgg9 import ANN_VGG9

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'checkpoints/manifest.json').read_text())
for t in (4,6,12,25):
    path=root/f'checkpoints/snn/vgg9_t{t}_best.pt'
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    model=build_model('vgg9',num_steps=t,num_classes=10)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in ckpt['state_dict'].items()})
    print(f'SNN T={t}: loaded, clean_acc={manifest[f"snn/vgg9_t{t}_best.pt"]["clean_test_acc"]:.2f}%')
path=root/'checkpoints/ann/vgg9_cifar10_best.pt'
ckpt=torch.load(path,map_location='cpu',weights_only=False)
model=ANN_VGG9(num_classes=10); model.load_state_dict({k.removeprefix('module.'):v for k,v in ckpt['state_dict'].items()})
print(f'ANN: loaded, clean_acc={manifest["ann/vgg9_cifar10_best.pt"]["clean_test_acc"]:.2f}%')
print('checkpoint smoke test passed')
