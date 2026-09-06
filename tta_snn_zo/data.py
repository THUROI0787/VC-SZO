"""Data loading: CIFAR-10 (clean) and CIFAR-10-C (official npy or synthetic)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .corruptions import CORRUPTIONS

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def cifar10_transform(train: bool = True):
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_cifar10_loaders(
    root: str,
    batch_size: int = 64,
    num_workers: int = 0,
    train_transform=None,
    test_transform=None,
    download: bool = True,
):
    train_transform = train_transform or cifar10_transform(train=True)
    test_transform = test_transform or cifar10_transform(train=False)
    train_set = datasets.CIFAR10(root=root, train=True, transform=train_transform, download=download)
    test_set = datasets.CIFAR10(root=root, train=False, transform=test_transform, download=download)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def get_cifar10_test_set(root: str, transform=None, download: bool = True):
    transform = transform or cifar10_transform(train=False)
    return datasets.CIFAR10(root=root, train=False, transform=transform, download=download)


class CIFAR10C(Dataset):
    """CIFAR-10-C from per-corruption .npy files (official layout).

    Expected files: ``root/<corruption>.npy`` (uint8 [N,32,32,3]) and
    ``root/labels.npy`` (int64 [N]). ``level`` selects the severity block
    (1-indexed, each block has 10k samples in the official 50k dataset).

    Robustness: if the file contains exactly one severity level (N == 10000,
    e.g. our synthetic single-level files) the whole file is used regardless of
    ``level``; if N is a multiple of 5 (official 50k layout) the level block is
    sliced; otherwise the whole file is used.
    """

    def __init__(self, root: str, corruption: str, transform=None, level: int = 5):
        super().__init__()
        self.root = Path(root)
        self.corruption = corruption
        self.transform = transform
        self.level = level
        data_path = self.root / f"{corruption}.npy"
        label_path = self.root / "labels.npy"
        if not data_path.exists():
            raise FileNotFoundError(f"CIFAR-10-C data missing: {data_path}")
        data = np.load(str(data_path))
        targets = np.load(str(label_path))
        n = len(targets)
        if n == 10000 and len(data) == 10000:
            # single-level file (synthetic or pre-sliced) - use all
            self.data = data
            self.targets = targets
        elif n % 5 == 0:
            block = n // 5
            start = min((level - 1) * block, n - block)
            self.data = data[start : start + block]
            self.targets = targets[start : start + block]
        else:
            self.data = data
            self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def get_cifar10c_loader(
    root: str,
    corruption: str,
    batch_size: int = 64,
    num_workers: int = 0,
    level: int = 5,
    transform=None,
):
    transform = transform or cifar10_transform(train=False)
    ds = CIFAR10C(root=root, corruption=corruption, transform=transform, level=level)
    sample_offset = int(os.environ.get("P4_SAMPLE_OFFSET", "0"))
    if sample_offset < 0 or sample_offset >= len(ds):
        raise ValueError(f"P4_SAMPLE_OFFSET={sample_offset} outside dataset of size {len(ds)}")
    if sample_offset:
        ds = Subset(ds, range(sample_offset, len(ds)))
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def iter_cifar10c(
    root: str,
    corruptions: list = None,
    batch_size: int = 64,
    num_workers: int = 0,
    level: int = 5,
    transform=None,
):
    """Yield (corruption_name, loader) for each corruption."""
    corruptions = corruptions or CORRUPTIONS
    for name in corruptions:
        yield name, get_cifar10c_loader(root, name, batch_size, num_workers, level, transform)


def auto_resolve_cifar10c_root(
    data_root: str,
    use_synthetic: bool = False,
    prepare_synthetic: bool = True,
    level: int = 5,
    download_official: bool = False,
    clean_test_set=None,
) -> tuple:
    """Resolve a usable CIFAR-10-C root.

    Returns (root_path, data_mode) where data_mode in {'official','synthetic'}.
    - official: root contains <corruption>.npy files (from CIFAR-10-C.tar).
    - synthetic: generated on the fly via corruptions.py into root+'/synthetic_level<level>'.
    """
    root = Path(data_root)
    official_candidate = root / "CIFAR-10-C"
    if official_candidate.is_dir() and (official_candidate / "gaussian_noise.npy").exists():
        return str(official_candidate), "official"
    if download_official:
        raise RuntimeError(
            "Official CIFAR-10-C download not performed by code; run setup_env.sh or "
            "download CIFAR-10-C.tar from https://zenodo.org/record/2535967 and extract "
            "to <data_root>/CIFAR-10-C/"
        )
    # synthetic
    if use_synthetic or prepare_synthetic:
        syn_root = root / f"synthetic_level{level}"
        if not (syn_root / "gaussian_noise.npy").exists():
            if clean_test_set is None:
                raise ValueError(
                    "synthetic CIFAR-10-C requested but no clean test set provided; "
                    "pass clean_test_set or run phase0/prepare_cifar10c.py first"
                )
            from .corruptions import prepare_level5_dir

            images = np.asarray(clean_test_set.data)
            labels = np.asarray(clean_test_set.targets)
            prepare_level5_dir(images, labels, str(syn_root), severities=(level,),
                               n_per_severity=10000, seed=0)
        return str(syn_root), "synthetic"
    raise FileNotFoundError(f"no CIFAR-10-C data found under {root}")


def get_clean_test_accuracy_loader(root: str, batch_size: int = 128, num_workers: int = 2):
    ds = get_cifar10_test_set(root)
    sample_offset = int(os.environ.get("P4_SAMPLE_OFFSET", "0"))
    if sample_offset < 0 or sample_offset >= len(ds):
        raise ValueError(f"P4_SAMPLE_OFFSET={sample_offset} outside dataset of size {len(ds)}")
    if sample_offset:
        ds = Subset(ds, range(sample_offset, len(ds)))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


# --------------------------------------------------------------------------- #
#  CIFAR-100 helpers (phase 3)
# --------------------------------------------------------------------------- #
def cifar100_transform(train: bool = True):
    return cifar10_transform(train)  # same normalization target values, fine for phase-3 demo


def get_cifar100_test_set(root: str, transform=None, download: bool = True):
    transform = transform or cifar100_transform(train=False)
    return datasets.CIFAR100(root=root, train=False, transform=transform, download=download)
