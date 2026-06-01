from enum import IntEnum
import numpy as np
from scipy.ndimage import median_filter
from .config import WorldConfig


class Biome(IntEnum):
    OCEAN = 0
    BEACH = 1
    DESERT = 2
    GRASSLAND = 3
    FOREST = 4
    JUNGLE = 5
    TUNDRA = 6
    MOUNTAIN = 7
    SNOW_PEAK = 8
    WETLAND = 9


BIOME_NAMES = {b.value: b.name for b in Biome}


def classify_biomes(
    elevation: np.ndarray,
    moisture: np.ndarray,
    temperature: np.ndarray,
    cfg: WorldConfig,
) -> np.ndarray:
    """
    Returns int8 (H, W) with Biome enum values.
    Priority order: ocean -> beach -> snow -> mountain -> tundra ->
    jungle -> wetland -> forest -> grassland -> desert.
    """
    H, W = elevation.shape
    result = np.full((H, W), Biome.DESERT, dtype=np.int8)

    # Work from lowest to highest priority (last write wins)
    result[moisture > 0.30] = Biome.GRASSLAND
    result[moisture > 0.50] = Biome.FOREST
    # Tundra: cold, low-to-moderate moisture
    result[(temperature < 0.25) & (moisture <= 0.65)] = Biome.TUNDRA
    # Jungle: high moisture, warm-to-hot (lower threshold to 0.45 so equatorial bands qualify)
    result[(moisture > 0.70) & (temperature > 0.45)] = Biome.JUNGLE
    # Wetland: high moisture, any temperature (cold bogs are real); beats tundra
    wetland_mask = (moisture > 0.75) & (elevation < 0.65)
    result[wetland_mask] = Biome.WETLAND
    result[elevation > 0.72] = Biome.MOUNTAIN
    result[elevation > 0.88] = Biome.SNOW_PEAK
    # Beach: thin strip just above sea level
    result[(elevation >= cfg.sea_level) & (elevation < cfg.sea_level + 0.04)] = Biome.BEACH
    result[elevation < cfg.sea_level] = Biome.OCEAN

    # Smooth internal transitions while preserving hard ocean/land edge
    smoothed = median_filter(result.astype(np.int32), size=3, mode="nearest").astype(np.int8)
    # Re-apply ocean/beach mask — never let smoothing create land from ocean
    smoothed[elevation < cfg.sea_level] = Biome.OCEAN
    smoothed[(elevation >= cfg.sea_level) & (elevation < cfg.sea_level + 0.04)] = Biome.BEACH

    return smoothed
