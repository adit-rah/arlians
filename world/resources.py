import numpy as np
from .biome import Biome
from .config import WorldConfig
from .noise_layers import make_resource_noise


def _normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return ((arr - lo) / (hi - lo + 1e-9)).astype(np.float32)


# Base resource values per biome
# columns: soil_fertility, water_proximity_base, wood, stone, wild_food, minerals
_BIOME_BASE = {
    Biome.OCEAN:     (0.00, 1.00, 0.00, 0.00, 0.10, 0.00),
    Biome.BEACH:     (0.10, 0.80, 0.00, 0.20, 0.10, 0.10),
    Biome.DESERT:    (0.05, 0.05, 0.00, 0.30, 0.05, 0.30),
    Biome.GRASSLAND: (0.55, 0.30, 0.10, 0.10, 0.30, 0.10),
    Biome.FOREST:    (0.35, 0.40, 0.80, 0.10, 0.50, 0.10),
    Biome.JUNGLE:    (0.40, 0.50, 0.90, 0.05, 0.70, 0.05),
    Biome.TUNDRA:    (0.10, 0.35, 0.05, 0.20, 0.15, 0.20),
    Biome.MOUNTAIN:  (0.05, 0.20, 0.10, 0.90, 0.10, 0.80),
    Biome.SNOW_PEAK: (0.00, 0.30, 0.00, 0.70, 0.00, 0.60),
    Biome.WETLAND:   (0.90, 0.85, 0.30, 0.05, 0.40, 0.05),
}

_RESOURCE_KEYS = ["soil_fertility", "water_proximity", "wood", "stone", "wild_food", "minerals"]


def _build_base_layer(biome_map: np.ndarray, resource_index: int) -> np.ndarray:
    H, W = biome_map.shape
    out = np.zeros((H, W), dtype=np.float32)
    for biome, values in _BIOME_BASE.items():
        out[biome_map == biome] = values[resource_index]
    return out


def _river_fertility_bonus(river_distance: np.ndarray) -> np.ndarray:
    """Sharp fertility bonus near rivers. Exponent 1.5 = steep dropoff."""
    bonus = np.maximum(0.0, 1.0 - river_distance) ** 1.5
    return bonus.astype(np.float32)


# Harsh biomes: river water does not create cropland (scale 0–1 on bonus only).
_RIVER_FERT_BIOME_SCALE = {
    Biome.DESERT: 0.0,
    Biome.TUNDRA: 0.15,
    Biome.MOUNTAIN: 0.0,
    Biome.SNOW_PEAK: 0.0,
    Biome.BEACH: 0.2,
}


def _river_fertility_scale_map(biome_map: np.ndarray) -> np.ndarray:
    H, W = biome_map.shape
    scale = np.ones((H, W), dtype=np.float32)
    for biome, s in _RIVER_FERT_BIOME_SCALE.items():
        scale[biome_map == biome] = s
    return scale


def _sparsify_wild_food(wild_food: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Multiply wild food by a sparse mask so most tiles have near-zero
    wild food and only occasional patches are high. Makes hunting
    require travel, encouraging farmers to stay put.
    """
    H, W = wild_food.shape
    # Sparse noise: exponential distribution remapped to [0,1].
    # scale controls how much wild food survives. History: 0.12 crushed nearly every
    # tile below metabolic break-even (forage income < energy_decay), starving founders.
    # 0.30 lifted that to break-even, but agents only plateaued at energy ~0.43 — still
    # below the 0.55 reproduction threshold, so births never happened and the population
    # couldn't bootstrap. 0.50 (~1.67x more wild food per tile) gives a competent forager
    # in a good biome a POSITIVE energy balance so it can reach the repro surplus, while
    # the exponential stays right-skewed (deserts/tundra/winter still harsh) — food stays
    # patchy, so travel and settling-in-fertile-zones pressure remain.
    sparse = rng.exponential(scale=0.50, size=(H, W)).astype(np.float32)
    sparse = np.clip(sparse, 0, 1)
    return np.clip(wild_food * sparse, 0, 1).astype(np.float32)


def make_resource_layers(
    biome_map: np.ndarray,
    elevation: np.ndarray,
    moisture: np.ndarray,
    river_mask: np.ndarray,
    river_distance: np.ndarray,
    cfg: WorldConfig,
    rng: np.random.Generator,
) -> dict:
    H, W = biome_map.shape
    result = {}

    for i, key in enumerate(_RESOURCE_KEYS):
        base = _build_base_layer(biome_map, i)
        # Add regional noise variation (30% noise, 70% biome base)
        noise = make_resource_noise(W, H, seed=cfg.seed + 3000 + i * 997)
        noise = _normalize(noise)
        raw = base * 0.7 + noise * 0.3
        result[key] = np.clip(raw, 0, 1).astype(np.float32)

    # River fertility bonus (suppressed on desert/tundra/mountain — water, not farmland)
    bonus = _river_fertility_bonus(river_distance)
    fert_scale = _river_fertility_scale_map(biome_map)
    fertility = result["soil_fertility"] + bonus * 0.6 * fert_scale
    result["soil_fertility"] = np.clip(fertility, 0, 1).astype(np.float32)

    # Water proximity driven by river distance (closer = higher)
    water_from_river = (1.0 - river_distance) ** 2
    result["water_proximity"] = np.clip(
        result["water_proximity"] * 0.4 + water_from_river * 0.6, 0, 1
    ).astype(np.float32)

    # Ocean tiles: zero out all terrestrial resources
    ocean = elevation < cfg.sea_level
    for key in _RESOURCE_KEYS:
        if key not in ("water_proximity",):
            result[key][ocean] = 0.0
    result["water_proximity"][ocean] = 1.0

    # Sparsify wild food — hunting should require travel
    result["wild_food"] = _sparsify_wild_food(result["wild_food"], rng)

    return result
