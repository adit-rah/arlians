"""
Renderer — top-down RGB visualization for debugging emergence (build-spec §4.2).

Phase 0 tooling: biome basemap + overlays for agents, predators, structures, and
crop tiles.  Emit PNG per step and mp4/gif per year via imageio.

save_mp4 attempts to write an mp4 via imageio.  If no suitable backend (ffmpeg /
pyav) is installed, it falls back to writing a .gif at the same stem and returns
the gif path so callers can check what was actually written.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Biome colours — one (R, G, B) uint8 triple per biome index (0-9)
# Order matches world.biome.Biome: OCEAN BEACH DESERT GRASSLAND FOREST JUNGLE
#                                  TUNDRA MOUNTAIN SNOW_PEAK WETLAND
# ---------------------------------------------------------------------------
BIOME_COLORS: np.ndarray = np.array([
    [ 30,  90, 180],   # 0  OCEAN       — deep blue
    [215, 195, 140],   # 1  BEACH       — tan / sand
    [220, 190,  80],   # 2  DESERT      — sandy yellow
    [110, 175,  60],   # 3  GRASSLAND   — medium green
    [ 34, 120,  34],   # 4  FOREST      — dark green
    [  0,  80,  30],   # 5  JUNGLE      — deep green
    [200, 215, 210],   # 6  TUNDRA      — pale blue-grey
    [130, 120, 110],   # 7  MOUNTAIN    — grey
    [245, 248, 255],   # 8  SNOW_PEAK   — near-white
    [ 50, 160, 150],   # 9  WETLAND     — teal
], dtype=np.uint8)

# Overlay colours (RGB uint8)
_AGENT_COLOR    = np.array([255, 220,   0], dtype=np.uint8)   # bright yellow
_PREDATOR_COLOR = np.array([255,  40,  40], dtype=np.uint8)   # bright red

# Structure colours by type index (0 = none; 1 = shelter; 2 = storage; 3 = wall)
_STRUCTURE_COLORS: dict[int, np.ndarray] = {
    1: np.array([255, 165,   0], dtype=np.uint8),   # shelter  — orange
    2: np.array([160, 100, 220], dtype=np.uint8),   # storage  — purple
    3: np.array([180, 180, 180], dtype=np.uint8),   # wall     — light grey
}

_CROP_COLOR = np.array([50, 210, 90], dtype=np.uint8)   # bright lime green


def biome_basemap(world) -> np.ndarray:
    """Return an (H, W, 3) uint8 RGB image with one fixed colour per biome tile.

    Parameters
    ----------
    world : world.World
        The read-only world object.  Uses ``world.biome_map`` (H, W) int8.

    Returns
    -------
    np.ndarray
        Shape ``(H, W, 3)``, dtype ``uint8``.
    """
    biome = world.biome_map  # (H, W) int8, values 0-9
    # Clamp to valid range in case of unexpected values
    idx = np.clip(biome.astype(np.intp), 0, len(BIOME_COLORS) - 1)
    return BIOME_COLORS[idx]  # fancy-index into (10,3) -> (H,W,3)


def render_frame(world, state, store) -> np.ndarray:
    """Render a single simulation frame as an (H, W, 3) uint8 RGB array.

    Layers painted in order (each overwrites previous):
      1. Biome basemap.
      2. Crop tiles  — ``state.crop_stage > 0`` tinted lime green.
      3. Structures  — ``state.structure_type > 0``, colour per type.
      4. Living non-predator agents — bright yellow dots at (y, x).
      5. Living predators           — bright red dots at (y, x).

    Safe with zero agents, zero crops, or zero structures.

    Parameters
    ----------
    world : world.World
    state : sim.state.WorldState
    store : sim.state.EntityStore

    Returns
    -------
    np.ndarray
        Shape ``(H, W, 3)``, dtype ``uint8``.
    """
    canvas = biome_basemap(world).copy()

    # --- crop overlay ---
    crop_mask = state.crop_stage > 0
    if crop_mask.any():
        canvas[crop_mask] = _CROP_COLOR

    # --- structure overlay ---
    struct_ys, struct_xs = np.where(state.structure_type > 0)
    for y, x in zip(struct_ys, struct_xs):
        stype = int(state.structure_type[y, x])
        color = _STRUCTURE_COLORS.get(stype)
        if color is not None:
            canvas[y, x] = color

    # --- agent overlays ---
    alive = store.alive
    if alive.any():
        is_pred = store.is_predator
        # Non-predator agents first (so predators are drawn on top)
        agent_mask = alive & ~is_pred
        if agent_mask.any():
            ys = store.y[agent_mask]
            xs = store.x[agent_mask]
            canvas[ys, xs] = _AGENT_COLOR

        # Predators
        pred_mask = alive & is_pred
        if pred_mask.any():
            ys = store.y[pred_mask]
            xs = store.x[pred_mask]
            canvas[ys, xs] = _PREDATOR_COLOR

    return canvas


def save_mp4(frames: Sequence[np.ndarray], path: str, fps: int = 15) -> str:
    """Write a sequence of RGB frames to an mp4 (or gif fallback) file.

    Attempts to write an mp4 using imageio v3 (requires the ``imageio[ffmpeg]``
    or ``imageio[pyav]`` backend).  If no suitable backend is available, falls
    back to writing a ``.gif`` at the same path stem and returns the gif path.

    Parameters
    ----------
    frames : sequence of (H, W, 3) uint8 ndarrays
        All frames must have the same shape.
    path : str
        Desired output path.  Extension should be ``.mp4``; if fallback is
        triggered the written file will have a ``.gif`` extension instead.
    fps : int
        Frames per second (default 15).

    Returns
    -------
    str
        Actual path written (may differ from ``path`` if gif fallback was used).
    """
    import imageio.v3 as iio

    volume = np.stack(frames, axis=0)  # (T, H, W, 3)

    try:
        iio.imwrite(path, volume, fps=fps)
        return path
    except Exception:
        # Fall back to gif
        gif_path = str(Path(path).with_suffix(".gif"))
        iio.imwrite(gif_path, volume, fps=fps)
        return gif_path
