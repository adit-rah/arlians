import numpy as np
from .config import WorldConfig


def _fbm_fft(width: int, height: int, seed: int, hurst: float = 0.75) -> np.ndarray:
    """
    Spectral fBm synthesis via FFT power-law filter. O(N log N).
    hurst in (0, 1): lower = rougher, higher = smoother terrain.
    Returns float32 (H, W) in [0, 1].
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((height, width))

    f = np.fft.rfft2(noise)

    fy = np.fft.fftfreq(height)[:, np.newaxis]
    fx = np.fft.rfftfreq(width)[np.newaxis, :]
    power = (fy ** 2 + fx ** 2) ** ((hurst + 1) / 2)
    power[0, 0] = 1.0  # avoid DC singularity

    f_filtered = f / power
    result = np.fft.irfft2(f_filtered, s=(height, width)).astype(np.float32)

    lo, hi = result.min(), result.max()
    return ((result - lo) / (hi - lo + 1e-9)).astype(np.float32)


def make_elevation(cfg: WorldConfig) -> np.ndarray:
    """
    Elevation with radial island falloff (land rises in center, ocean at edges).
    Returns float32 (H, W) in [0, 1].
    """
    raw = _fbm_fft(cfg.width, cfg.height, seed=cfg.seed, hurst=0.80)

    cx = np.linspace(-1.0, 1.0, cfg.width, dtype=np.float32)
    cy = np.linspace(-1.0, 1.0, cfg.height, dtype=np.float32)
    gx, gy = np.meshgrid(cx, cy)
    dist = np.sqrt(gx ** 2 + gy ** 2)
    falloff = (1.0 - np.clip(dist * 0.85, 0.0, 1.0) ** 1.5).astype(np.float32)

    raw = raw * falloff
    lo, hi = raw.min(), raw.max()
    return ((raw - lo) / (hi - lo + 1e-9)).astype(np.float32)


def make_moisture(cfg: WorldConfig) -> np.ndarray:
    """Independent moisture field. Returns float32 (H, W) in [0, 1]."""
    return _fbm_fft(cfg.width, cfg.height, seed=cfg.seed + 1000, hurst=0.70)


def make_temperature_baseline(cfg: WorldConfig) -> np.ndarray:
    """
    Latitudinal gradient (hot equator, cold poles) + regional noise.
    Returns float32 (H, W) in [0, 1].
    """
    lat = np.abs(np.linspace(-1.0, 1.0, cfg.height, dtype=np.float32))
    base_temp = (1.0 - lat) ** 1.5
    base_2d = np.tile(base_temp[:, np.newaxis], (1, cfg.width))

    noise = _fbm_fft(cfg.width, cfg.height, seed=cfg.seed + 2000, hurst=0.60)
    combined = base_2d * 0.75 + noise * 0.25
    lo, hi = combined.min(), combined.max()
    return ((combined - lo) / (hi - lo + 1e-9)).astype(np.float32)


def make_resource_noise(width: int, height: int, seed: int) -> np.ndarray:
    """Single-layer noise for resource variation. Returns float32 (H, W) in [0, 1]."""
    return _fbm_fft(width, height, seed=seed, hurst=0.55)
