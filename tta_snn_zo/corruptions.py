"""Synthetic CIFAR-10-C style corruptions (Hendrycks & Dietterich 2019, official algorithm).

Faithful numpy + OpenCV reimplementation of the 15 CIFAR-10-C corruptions at any
severity 1..5.  Used when the official ``CIFAR-10-C.tar`` (2.7 GB) cannot be
downloaded (small disk / offline).  The official .npy loader in ``data.py`` is
the preferred path on machines with enough disk; this module is the fallback and
is also used by ``phase0/prepare_cifar10c.py`` to pre-generate level-5 files.

Reference: https://github.com/hendrycks/robustness (MIT license).
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]

# frost texture assets (small PNGs, mirror of hendrycks/robustness/cifar10_c/frost)
_FROST_URLS = [
    "https://raw.githubusercontent.com/hendrycks/robustness/master/cifar10_c/frost/frost1.png",
    "https://raw.githubusercontent.com/hendrycks/robustness/master/cifar10_c/frost/frost2.png",
    "https://raw.githubusercontent.com/hendrycks/robustness/master/cifar10_c/frost/frost3.png",
    "https://raw.githubusercontent.com/hendrycks/robustness/master/cifar10_c/frost/frost4.png",
    "https://raw.githubusercontent.com/hendrycks/robustness/master/cifar10_c/frost/frost5.png",
]


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _clip_0_1(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 1)


def _gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(x.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)


def _disk(radius: int, alias_blur: float = 0.1, dtype: type = np.float32) -> np.ndarray:
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X**2 + Y**2) <= radius**2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def _line(length: int, angle: float) -> np.ndarray:
    mask = np.zeros((length, length), dtype=np.float32)
    half = max(1, int(length / 2))
    angle = np.deg2rad(angle)
    x, y = np.ogrid[: mask.shape[0], : mask.shape[1]]
    cx, cy = mask.shape[0] // 2, mask.shape[1] // 2
    mask[((x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)) ** 2 < 1] = 1
    return mask / np.sum(mask)


def _value_noise(size: int, octave: int) -> np.ndarray:
    """Simple value noise used for fog / snow textures."""
    coarse = 2 ** max(octave, 0)
    grid = np.random.rand(coarse + 1, coarse + 1)
    xs = np.linspace(0, coarse, size, endpoint=False)
    x0 = np.floor(xs).astype(int)
    fx = xs - x0
    x0 = np.clip(x0, 0, coarse)
    x1 = np.clip(x0 + 1, 0, coarse)
    return _bilinear(grid, x0, x1, fx, size)


def _bilinear(grid: np.ndarray, x0: np.ndarray, x1: np.ndarray, fx: np.ndarray, size: int) -> np.ndarray:
    g = grid
    y0 = np.clip(np.floor(np.linspace(0, g.shape[0] - 1, size)).astype(int), 0, g.shape[0] - 1)
    y1 = np.clip(y0 + 1, 0, g.shape[0] - 1)
    fy = np.linspace(0, 1, size)
    fy = (np.linspace(0, g.shape[0] - 1, size) - y0).astype(float)
    # gather
    v00 = g[y0][:, x0]
    v01 = g[y0][:, x1]
    v10 = g[y1][:, x0]
    v11 = g[y1][:, x1]
    fx = fx[None, :]
    fy = fy[:, None]
    out = (v00 * (1 - fx) + v01 * fx) * (1 - fy) + (v10 * (1 - fx) + v11 * fx) * fy
    return out


def _plasma_fractal(mudims: int = 32, mudepth: int = 3) -> np.ndarray:
    x = np.zeros((mudims, mudims), dtype=np.float32)
    for i in range(int(mudepth)):
        size = mudims // (2**i)
        n = _value_noise(max(size, 2), i)
        x += cv2.resize(n, (mudims, mudims), interpolation=cv2.INTER_CUBIC)
    x /= max(int(mudepth), 1)
    return _clip_0_1(x)


def _frost_textures(frost_dir: str) -> list:
    """Load frost textures; if missing, try to download; else synthetic."""
    frost_dir = Path(frost_dir)
    textures = []
    if frost_dir.is_dir():
        for i in range(1, 6):
            p = frost_dir / f"frost{i}.png"
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    textures.append(cv2.resize(img, (32, 32)))
    if not textures:
        try:
            import urllib.request

            frost_dir.mkdir(parents=True, exist_ok=True)
            for url in _FROST_URLS:
                dst = frost_dir / url.rsplit("/", 1)[-1]
                if not dst.exists():
                    urllib.request.urlretrieve(url, str(dst))
                img = cv2.imread(str(dst), cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    textures.append(cv2.resize(img, (32, 32)))
        except Exception:  # pragma: no cover - network failure fallback
            pass
    if not textures:
        # synthetic frost: white noise + blur, looks like frost on average
        rng = np.random.RandomState(0)
        for _ in range(5):
            t = rng.rand(32, 32, 3).astype(np.float32)
            t = cv2.GaussianBlur(t, (7, 7), 2.0)
            t = np.clip(t * 1.5 - 0.25, 0, 1)
            textures.append((t * 255).astype(np.uint8))
    return textures


# --------------------------------------------------------------------------- #
#  corruptions (input x: uint8 [H,W,3] in [0,255]; output same)
# --------------------------------------------------------------------------- #
def gaussian_noise(x: np.ndarray, severity: int = 1) -> np.ndarray:
    c = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.0
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x: np.ndarray, severity: int = 1) -> np.ndarray:
    c = [60, 25, 12, 5, 3][severity - 1]
    x = np.array(x) / 255.0
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x: np.ndarray, severity: int = 1) -> np.ndarray:
    c = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
    x = np.array(x, dtype=np.uint8)
    noise = np.random.uniform(size=x.shape)
    x[noise < c / 2] = 0
    x[noise > 1 - c / 2] = 255
    return np.clip(x, 0, 255)


def speckle_noise(x: np.ndarray, severity: int = 1) -> np.ndarray:
    c = [0.15, 0.2, 0.35, 0.45, 0.6][severity - 1]
    x = np.array(x) / 255.0
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x: np.ndarray, severity: int = 1) -> np.ndarray:
    c = [1, 2, 4, 8, 12][severity - 1]
    x = _gaussian(np.array(x) / 255.0, sigma=c)
    return np.clip(x, 0, 1) * 255


def glass_blur(x: np.ndarray, severity: int = 1) -> np.ndarray:
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
    x = np.uint8(_gaussian(np.array(x) / 255.0, sigma=c[0]) * 255)
    for _ in range(c[2]):
        for h in range(32 - c[1], c[1], -1):
            for w in range(32 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                hh, ww = h + dy, w + dx
                x[h, w], x[hh, ww] = x[hh, ww], x[h, w]
    return np.clip(_gaussian(x / 255.0, sigma=c[0]), 0, 1) * 255


def defocus_blur(x: np.ndarray, severity: int = 1) -> np.ndarray:
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
    x = np.array(x) / 255.0
    kernel = _disk(radius=c[0])
    kernel = np.uint8(255 * kernel / np.sum(kernel))
    x = cv2.filter2D(x, -1, kernel)
    return np.clip(x, 0, 1) * 255


def motion_blur(x: np.ndarray, severity: int = 1) -> np.ndarray:
    c = [(6, 1), (8, 1), (10, 1), (12, 1), (14, 1)][severity - 1]
    x = np.array(x) / 255.0
    kernel = _line(c[0], c[1])
    kernel = np.uint8(255 * kernel / np.sum(kernel))
    x = cv2.filter2D(x, -1, kernel)
    return np.clip(x, 0, 1) * 255


def zoom_blur(x: np.ndarray, severity: int = 1) -> np.ndarray:
    c = [
        np.arange(1, 1.11, 0.01),
        np.arange(1, 1.16, 0.01),
        np.arange(1, 1.21, 0.02),
        np.arange(1, 1.26, 0.02),
        np.arange(1, 1.31, 0.03),
    ][severity - 1]
    x = (np.array(x) / 255.0).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += cv2.resize(
            cv2.resize(x, None, fx=zoom_factor, fy=zoom_factor), (32, 32)
        )
    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def fog(x: np.ndarray, severity: int = 1) -> np.ndarray:
    c = [(1.5, 2), (2.0, 2), (2.5, 1.7), (2.5, 1.5), (3.0, 1.4)][severity - 1]
    x = np.array(x) / 255.0
    max_val = x.max()
    f = _plasma_fractal(mudims=32, mudepth=c[1])
    if f.ndim == 2:  # broadcast grayscale fractal over RGB channels
        f = f[..., None] * np.ones((1, 1, x.shape[2]), dtype=np.float32)
    x += c[0] * f
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x: np.ndarray, severity: int = 1, frost_dir: str = "data/frost") -> np.ndarray:
    c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]
    textures = _frost_textures(frost_dir)
    idx = np.random.randint(len(textures))
    frost_img = textures[idx].astype(np.float32) / 255.0
    x = np.array(x) / 255.0
    return np.clip(c[0] * x + c[1] * frost_img, 0, 1) * 255


def snow(x: np.ndarray, severity: int = 1) -> np.ndarray:
    c = [
        (0.1, 0.3, 3, 0.5, 10, 4, 0.8),
        (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
        (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
        (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
        (0.55, 0.3, 2.5, 0.75, 12, 8, 0.65),
    ][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.0
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1]).astype(np.float32)
    snow_layer = cv2.GaussianBlur(snow_layer, (3, 3), 0)
    snow_layer[snow_layer < c[2]] = 0
    snow_layer = np.repeat(snow_layer[..., np.newaxis], 3, axis=-1)
    ks = int(c[4]) | 1  # ensure odd kernel size
    x = c[3] * x + (1 - c[3]) * np.maximum(
        x, cv2.GaussianBlur(x, (ks, ks), 0) * c[5]
    ) + snow_layer * c[6]
    return np.clip(x, 0, 1) * 255


def brightness(x: np.ndarray, severity: int = 1) -> np.ndarray:
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    x = np.array(x) / 255.0
    x = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2HSV)
    x[..., 2] = np.clip(x[..., 2] + c, 0, 1)
    x = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)
    return np.clip(x, 0, 1) * 255


def contrast(x: np.ndarray, severity: int = 1) -> np.ndarray:
    c = [0.4, 0.3, 0.2, 0.1, 0.05][severity - 1]
    x = np.array(x) / 255.0
    mean = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - mean) * c + mean, 0, 1) * 255


def elastic_transform(x: np.ndarray, severity: int = 1) -> np.ndarray:
    c = [
        (244 * 2, 720 * 2, 0.12),
        (244 * 2, 720 * 2, 0.12),
        (244 * 2, 720 * 2, 0.12),
        (244 * 2, 720 * 2, 0.12),
        (244 * 2, 720 * 2, 0.12),
    ][severity - 1]
    image = np.array(x, dtype=np.float32) / 255.0
    shape = image.shape
    shape_size = shape[:2]
    center_square = np.float32((shape_size[0] // 2, shape_size[0] // 2))
    square_size = min(shape_size) // 3
    pts1 = np.float32(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
        ]
    )
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    dx = (_gaussian(np.random.uniform(-1, 1, size=shape[:2]), c[1]) * c[0]).astype(np.float32)
    dy = (_gaussian(np.random.uniform(-1, 1, size=shape[:2]), c[1]) * c[0]).astype(np.float32)
    mapx = (np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))[0] + dx).astype(np.float32)
    mapy = (np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))[1] + dy).astype(np.float32)
    out = np.zeros_like(image)
    for ch in range(shape[2]):
        out[..., ch] = cv2.remap(
            image[..., ch], mapx, mapy, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    return np.clip(out, 0, 1) * 255


def pixelate(x: np.ndarray, severity: int = 1) -> np.ndarray:
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    x = np.array(x) / 255.0
    shape = x.shape
    x = cv2.resize(x, (int(32 * c), int(32 * c)), interpolation=cv2.INTER_AREA)
    x = cv2.resize(x, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.clip(x, 0, 1) * 255


def jpeg_compression(x: np.ndarray, severity: int = 1) -> np.ndarray:
    c = [25, 18, 15, 10, 7][severity - 1]
    x = np.array(x, dtype=np.uint8)
    ok, encimg = cv2.imencode(
        ".jpg", cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), c]
    )
    if not ok:  # pragma: no cover
        return x
    return cv2.cvtColor(cv2.imdecode(encimg, 1), cv2.COLOR_BGR2RGB)


_CORRUPTION_FNS = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    "defocus_blur": defocus_blur,
    "glass_blur": glass_blur,
    "motion_blur": motion_blur,
    "zoom_blur": zoom_blur,
    "snow": snow,
    "frost": frost,
    "fog": fog,
    "brightness": brightness,
    "contrast": contrast,
    "elastic_transform": elastic_transform,
    "pixelate": pixelate,
    "jpeg_compression": jpeg_compression,
}


def apply_corruption(x: np.ndarray, name: str, severity: int = 1, **kwargs) -> np.ndarray:
    """Apply one corruption to a uint8 [H,W,3] image (or [N,H,W,3] batch)."""
    if name not in _CORRUPTION_FNS:
        raise ValueError(f"unknown corruption {name}; available: {list(_CORRUPTION_FNS)}")
    fn = _CORRUPTION_FNS[name]
    # only pass kwargs the specific corruption accepts (e.g. frost_dir for frost)
    import inspect

    sig = inspect.signature(fn)
    fn_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    single = x.ndim == 3
    arr = x[None] if single else x
    out = np.stack([fn(img, severity, **fn_kwargs) for img in arr])
    return out[0] if single else out


def generate_corrupted_set(
    clean_images: np.ndarray,
    name: str,
    severity: int = 5,
    n_per_severity: int = 10000,
    frost_dir: str = "data/frost",
    seed: int = 0,
) -> np.ndarray:
    """Generate a corrupted subset (uint8 [N,32,32,3]) from clean test images.

    To reproduce the official CIFAR-10-C protocol we take a random subset of the
    clean set of size ``n_per_severity`` (official uses the first 10k test images,
    which is equivalent under i.i.d. sampling for benchmarking).
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)
    clean = np.asarray(clean_images)
    if clean.shape[0] > n_per_severity:
        # official protocol: use the first n_per_severity test images (aligned
        # with labels[:n_per_severity])
        clean = clean[:n_per_severity]
    out = np.stack([apply_corruption(img, name, severity, frost_dir=frost_dir) for img in clean])
    out = np.clip(out, 0, 255).astype(np.uint8)
    np.random.set_state(rng_state)
    return out


def prepare_level5_dir(
    clean_test: np.ndarray,
    labels: np.ndarray,
    out_dir: str,
    severities=(5,),
    n_per_severity: int = 10000,
    seed: int = 0,
) -> dict:
    """Generate CIFAR-10-C level files (official .npy layout) into out_dir.

    Writes ``<out_dir>/<corruption>.npy`` (uint8 [N,32,32,3]) and labels.npy.
    Returns a summary dict.
    """
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for sev in severities:
        for name in CORRUPTIONS:
            dst = out_dir / f"{name}.npy"
            if dst.exists():
                summary[name] = "cached"
                continue
            data = generate_corrupted_set(
                clean_test, name, severity=sev, n_per_severity=n_per_severity,
                frost_dir=str(out_dir.parent / "frost"), seed=seed,
            )
            np.save(str(dst), data)
            summary[name] = f"generated severity={sev}"
    np.save(str(out_dir / "labels.npy"), np.asarray(labels)[:n_per_severity])
    return summary


if __name__ == "__main__":
    # quick self-test: apply all corruptions at severity 5 and show shapes
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    for name in CORRUPTIONS:
        out = np.clip(np.asarray(apply_corruption(img, name, severity=5)), 0, 255).astype(np.uint8)
        assert out.shape == img.shape, (name, out.shape)
        assert out.dtype == np.uint8, (name, out.dtype)
    print("corruptions.py self-test OK:", len(CORRUPTIONS), "corruptions")
