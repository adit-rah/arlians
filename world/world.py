import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .config import WorldConfig
from .noise_layers import make_elevation, make_moisture, make_temperature_baseline
from .biome import classify_biomes
from .rivers import compute_river_mask, compute_river_distance
from .resources import make_resource_layers
from .seasons import compute_season_state, apply_seasonal_modifiers

# Channel order for the (13, H, W) atlas
LAYER_NAMES = [
    "elevation",         # 0
    "moisture",          # 1
    "temperature_base",  # 2
    "biome",             # 3  (stored normalized)
    "river_mask",        # 4
    "river_distance",    # 5
    "flow_accumulation", # 6  (stored normalized log-scale)
    "soil_fertility",    # 7
    "water_proximity",   # 8
    "wood",              # 9
    "stone",             # 10
    "wild_food",         # 11
    "minerals",          # 12
]

NUM_CHANNELS = len(LAYER_NAMES)          # 13: atlas channels
NUM_OBS_CHANNELS = NUM_CHANNELS + 1     # 14: atlas + season_phase


@dataclass
class World:
    cfg: WorldConfig

    elevation: np.ndarray
    moisture: np.ndarray
    temperature_base: np.ndarray
    biome_map: np.ndarray          # int8
    river_mask: np.ndarray         # bool
    river_distance: np.ndarray
    flow_accumulation: np.ndarray  # int32
    base_resources: dict

    _static_channels: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def generate(cls, cfg: WorldConfig = WorldConfig()) -> "World":
        """Run the full generation pipeline and return a populated World."""
        print(f"[world] Generating {cfg.width}x{cfg.height} world (seed={cfg.seed})...")
        rng = np.random.default_rng(cfg.seed)

        print("[world] 1/5  noise layers...")
        elev = make_elevation(cfg)
        moist = make_moisture(cfg)
        temp = make_temperature_baseline(cfg)

        print("[world] 2/5  biomes...")
        biome_map = classify_biomes(elev, moist, temp, cfg)

        print("[world] 3/5  rivers...")
        river_mask, flow_accum = compute_river_mask(elev, cfg, rng)
        river_dist = compute_river_distance(river_mask)

        print("[world] 4/5  resources...")
        resources = make_resource_layers(biome_map, elev, moist, river_mask, river_dist, cfg, rng)

        print("[world] 5/5  done.")
        return cls(
            cfg=cfg,
            elevation=elev,
            moisture=moist,
            temperature_base=temp,
            biome_map=biome_map,
            river_mask=river_mask,
            river_distance=river_dist,
            flow_accumulation=flow_accum,
            base_resources=resources,
        )

    def _build_static_channels(self) -> np.ndarray:
        """Build channels 0-6: static geography that never changes."""
        # Normalize flow_accumulation to [0,1] via log scale
        fa = self.flow_accumulation.astype(np.float32)
        fa_norm = np.log1p(fa) / np.log1p(fa.max() + 1)
        return np.stack([
            self.elevation,
            self.moisture,
            self.temperature_base,
            self.biome_map.astype(np.float32) / 9.0,  # 9 = max biome index
            self.river_mask.astype(np.float32),
            self.river_distance,
            fa_norm,
        ], axis=0)  # (7, H, W)

    def get_atlas(self) -> np.ndarray:
        """Return full static atlas (7 geo channels + 6 base resource channels), shape (13, H, W)."""
        if self._static_channels is None:
            object.__setattr__(self, "_static_channels", self._build_static_channels())
        resource_channels = np.stack([
            self.base_resources["soil_fertility"],
            self.base_resources["water_proximity"],
            self.base_resources["wood"],
            self.base_resources["stone"],
            self.base_resources["wild_food"],
            self.base_resources["minerals"],
        ], axis=0)  # (6, H, W)
        return np.concatenate([self._static_channels, resource_channels], axis=0)  # (13, H, W)

    def get_obs(self, t: int) -> np.ndarray:
        """
        Return (14, H, W) float32 for time step t.
        Channels 0-6:  static geography.
        Channels 7-12: season-modified resources.
        Channel 13:    season_phase broadcast as constant layer.
        """
        if self._static_channels is None:
            object.__setattr__(self, "_static_channels", self._build_static_channels())

        season = compute_season_state(t, self.cfg)
        seasonal = apply_seasonal_modifiers(self.base_resources, season)

        resource_channels = np.stack([
            seasonal["soil_fertility"],
            seasonal["water_proximity"],
            seasonal["wood"],
            seasonal["stone"],
            seasonal["wild_food"],
            seasonal["minerals"],
        ], axis=0)  # (6, H, W)

        H, W = self.elevation.shape
        phase_layer = np.full((1, H, W), season.season_phase, dtype=np.float32)

        return np.concatenate([
            self._static_channels,
            resource_channels,
            phase_layer,
        ], axis=0)  # (14, H, W): 7 geo + 6 resources + 1 season_phase

    def get_window(
        self,
        t: int,
        center_y: int,
        center_x: int,
        radius: int,
    ) -> np.ndarray:
        """
        Return (13, 2r+1, 2r+1) float32 local view for an agent.
        Out-of-bounds tiles are zero-padded.
        """
        obs = self.get_obs(t)
        C, H, W = obs.shape
        size = 2 * radius + 1
        window = np.zeros((C, size, size), dtype=np.float32)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                wr = dy + radius
                wc = dx + radius
                wy = center_y + dy
                wx = center_x + dx
                if 0 <= wy < H and 0 <= wx < W:
                    window[:, wr, wc] = obs[:, wy, wx]
        return window
