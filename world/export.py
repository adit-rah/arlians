import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from .world import World, LAYER_NAMES
from .biome import BIOME_NAMES
from .seasons import compute_season_state


def export_world(world: World, output_dir) -> None:
    """Save the static atlas and metadata to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    atlas = world.get_atlas()  # (13, H, W)
    np.save(out / "atlas.npy", atlas)
    np.save(out / "biome_map.npy", world.biome_map)
    np.save(out / "river_mask.npy", world.river_mask.astype(np.uint8))

    H, W = world.elevation.shape
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": world.cfg.seed,
        "shape": [H, W],
        "layers": [
            {
                "index": i,
                "name": name,
                "min": float(atlas[i].min()),
                "max": float(atlas[i].max()),
            }
            for i, name in enumerate(LAYER_NAMES)
        ],
        "biomes": BIOME_NAMES,
        "obs_channels": 14,
        "obs_channel_description": (
            "channels 0-6: geography, 7-12: seasonal resources, 13: season_phase"
        ),
    }
    (out / "atlas_meta.json").write_text(json.dumps(meta, indent=2))

    cfg_dict = {k: v for k, v in world.cfg.__dict__.items()}
    (out / "world_config.json").write_text(json.dumps(cfg_dict, indent=2))

    print(f"[export] Saved atlas ({atlas.shape}) to {out}/")


def export_season_snapshot(world: World, t: int, output_dir) -> None:
    """Save a seasonal observation snapshot for timestep t."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    obs = world.get_obs(t)
    fname = out / f"obs_t{t:06d}.npy"
    np.save(fname, obs)
    season = compute_season_state(t, world.cfg)
    print(f"[export] Saved obs at t={t} (phase={season.season_phase:.2f}) -> {fname}")
