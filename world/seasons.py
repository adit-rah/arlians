import math
from dataclasses import dataclass
import numpy as np
from .config import WorldConfig


@dataclass(frozen=True)
class SeasonState:
    t: int
    season_phase: float         # [0, 1) position in year
    temperature_modifier: float
    moisture_modifier: float
    fertility_modifier: float
    wild_food_modifier: float
    water_level_modifier: float


def compute_season_state(t: int, cfg: WorldConfig) -> SeasonState:
    """Derive all seasonal multipliers from t. Fully deterministic."""
    phase = (t % cfg.season_period) / cfg.season_period
    tau = 2 * math.pi * phase

    # Temperature: peaks in summer (phase=0.5), dips in winter (phase=0.0/1.0)
    temp_mod = 0.70 + 0.30 * math.sin(tau - math.pi / 2)

    # Moisture: highest in spring (snowmelt + rain)
    moist_mod = 0.85 + 0.15 * math.sin(tau + math.pi * 0.5)

    # Fertility: follows temperature with a slight lag (growing season)
    fert_mod = 0.50 + 0.50 * math.sin(tau - math.pi / 2 + 0.3)
    fert_mod = max(0.2, fert_mod)

    # Wild food: deeper amplitude, slightly phase-lagged after fertility
    wild_mod = 0.40 + 0.55 * math.sin(tau - math.pi / 2 + 0.6)
    wild_mod = max(0.05, wild_mod)

    # Water level: rivers run high in spring, low in late summer
    water_mod = 0.80 + 0.20 * math.sin(tau + math.pi)

    return SeasonState(
        t=t,
        season_phase=phase,
        temperature_modifier=temp_mod,
        moisture_modifier=moist_mod,
        fertility_modifier=fert_mod,
        wild_food_modifier=wild_mod,
        water_level_modifier=water_mod,
    )


def apply_seasonal_modifiers(
    base_resources: dict,
    season: SeasonState,
) -> dict:
    """Return a new dict of resource arrays with seasonal modifiers applied."""
    result = {}
    for key, arr in base_resources.items():
        if key == "soil_fertility":
            result[key] = np.clip(arr * season.fertility_modifier, 0, 1).astype(np.float32)
        elif key == "wild_food":
            result[key] = np.clip(arr * season.wild_food_modifier, 0, 1).astype(np.float32)
        elif key == "water_proximity":
            result[key] = np.clip(arr * season.water_level_modifier, 0, 1).astype(np.float32)
        else:
            result[key] = arr  # wood, stone, minerals are abiotic
    return result
