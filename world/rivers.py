import numpy as np
from scipy.ndimage import distance_transform_edt
from .config import WorldConfig

_D8_OFFSETS = np.array([
    (0, 1), (-1, 1), (-1, 0), (-1, -1),
    (0, -1), (1, -1), (1, 0), (1, 1),
], dtype=np.int32)

_D8_DIST = np.array([1.0, np.sqrt(2), 1.0, np.sqrt(2),
                     1.0, np.sqrt(2), 1.0, np.sqrt(2)], dtype=np.float32)


def _steepest_descent_flat(elevation: np.ndarray, cfg: WorldConfig) -> np.ndarray:
    """
    For each cell, compute the flat index of its steepest downhill neighbor.
    Returns int32 (H*W,). Value -1 = no downhill or ocean.
    """
    H, W = elevation.shape
    elev_pad = np.pad(elevation.astype(np.float32), 1, mode="edge")

    best_slope = np.zeros((H, W), dtype=np.float32)
    best_dir = np.full((H, W), -1, dtype=np.int32)

    for d, (dy, dx) in enumerate(_D8_OFFSETS):
        nbr = elev_pad[1 + dy: H + 1 + dy, 1 + dx: W + 1 + dx]
        slope = (elevation - nbr) / _D8_DIST[d]
        improved = slope > best_slope
        np.copyto(best_slope, slope, where=improved)
        np.copyto(best_dir, np.int32(d), where=improved)

    ocean = elevation < cfg.sea_level
    rows_g, cols_g = np.mgrid[0:H, 0:W]
    downstream = np.full(H * W, -1, dtype=np.int32)

    has_dir = (best_dir >= 0) & ~ocean
    if has_dir.any():
        dy_map = _D8_OFFSETS[best_dir.clip(0, 7), 0]
        dx_map = _D8_OFFSETS[best_dir.clip(0, 7), 1]
        nr = (rows_g + dy_map).ravel()
        nc = (cols_g + dx_map).ravel()
        valid = has_dir.ravel()
        in_bounds = valid & (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
        src_idx = np.where(in_bounds)[0]
        downstream[src_idx] = nr[src_idx] * W + nc[src_idx]

    return downstream


def _trace_paths_vectorized(
    downstream: np.ndarray,
    sources: np.ndarray,
    n_cells: int,
    max_steps: int,
) -> np.ndarray:
    """
    Simultaneously trace all source paths downstream. Vectorized over sources.
    Returns int32 (n_cells,) accumulation: paths passing through each cell.
    """
    accum = np.zeros(n_cells, dtype=np.int32)
    current = sources.astype(np.int64).copy()  # (N,)
    active = np.ones(len(current), dtype=bool)

    for _ in range(max_steps):
        if not active.any():
            break
        live = current[active]
        np.add.at(accum, live, 1)
        nxt = downstream[live]
        alive_mask = nxt >= 0
        new_active = np.zeros(len(current), dtype=bool)
        active_indices = np.where(active)[0]
        new_active[active_indices[alive_mask]] = True
        current[active_indices[alive_mask]] = nxt[alive_mask]
        active = new_active

    return accum


def compute_river_mask(
    elevation: np.ndarray,
    cfg: WorldConfig,
    rng: np.random.Generator,
) -> tuple:
    """
    Returns (river_mask bool (H,W), flow_accum int32 (H,W)).
    """
    H, W = elevation.shape
    land = elevation >= cfg.sea_level
    land_flat = land.ravel()

    downstream = _steepest_descent_flat(elevation, cfg)

    land_indices = np.where(land_flat)[0]
    if len(land_indices) == 0:
        return np.zeros((H, W), dtype=bool), np.zeros((H, W), dtype=np.int32)

    elev_flat = elevation.ravel()
    elev_land = elev_flat[land_indices]
    threshold = np.percentile(elev_land, 90)
    high_land = land_indices[elev_land >= threshold]

    n_sources = min(len(high_land), cfg.river_max_sources)
    sources = rng.choice(high_land, size=n_sources, replace=False)

    max_steps = H + W
    accum_flat = _trace_paths_vectorized(downstream, sources, H * W, max_steps)

    flow_accum = accum_flat.reshape(H, W).astype(np.int32)

    # Adaptive threshold: top river_density fraction of land accumulation values
    land_accum = flow_accum[land]
    if land_accum.max() > 0:
        threshold = np.percentile(land_accum, (1.0 - cfg.river_density) * 100)
        threshold = max(threshold, 1)  # must have at least 1 path through it
    else:
        threshold = 1
    river_mask = (flow_accum >= threshold) & land

    return river_mask, flow_accum


def compute_river_distance(river_mask: np.ndarray) -> np.ndarray:
    """Normalized distance to nearest river. Returns float32 (H, W) in [0, 1]."""
    if river_mask.any():
        dist = distance_transform_edt(~river_mask).astype(np.float32)
        max_dist = dist.max()
        if max_dist > 0:
            dist /= max_dist
    else:
        dist = np.ones(river_mask.shape, dtype=np.float32)
    return dist
