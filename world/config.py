from dataclasses import dataclass


@dataclass(frozen=True)
class WorldConfig:
    width: int = 1024
    height: int = 1024
    seed: int = 42

    # Noise for elevation
    elevation_octaves: int = 8
    elevation_persistence: float = 0.5
    elevation_lacunarity: float = 2.0
    elevation_scale: float = 3.0

    # Noise for moisture
    moisture_octaves: int = 6
    moisture_persistence: float = 0.5
    moisture_lacunarity: float = 2.0
    moisture_scale: float = 2.5

    # Noise for temperature baseline
    temp_octaves: int = 4
    temp_persistence: float = 0.4
    temp_scale: float = 2.0

    # Sea level: tiles below this elevation are ocean
    sea_level: float = 0.38

    # Rivers: top X% of flow-accumulation cells become rivers
    river_density: float = 0.003   # 0.3% of land tiles = ~1350 river cells on 1024x1024
    # Max source cells for path tracing
    river_max_sources: int = 20000

    # Seasons
    season_period: int = 360

    # Resource noise scale
    resource_noise_scale: float = 4.0
