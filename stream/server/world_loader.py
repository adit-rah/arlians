"""
Load a pre-exported world directory (from generate.py / world.export) into world.World.
"""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from world.config import WorldConfig
from world.world import World


def load_world_from_dir(world_dir: str | Path) -> World:
    """
    Reconstruct World from ``atlas.npy``, ``biome_map.npy``, ``river_mask.npy``,
    and ``world_config.json`` written by ``world.export.export_world``.
    """
    root = Path(world_dir)
    atlas_path = root / "atlas.npy"
    if not atlas_path.is_file():
        raise FileNotFoundError(f"Missing {atlas_path} — run: python generate.py --output {root}")

    atlas = np.load(atlas_path)
    if atlas.ndim != 3 or atlas.shape[0] < 13:
        raise ValueError(f"Expected atlas (13, H, W), got {atlas.shape}")

    _, H, W = atlas.shape

    cfg_path = root / "world_config.json"
    if cfg_path.is_file():
        cfg_dict = json.loads(cfg_path.read_text())
        valid = {f.name for f in fields(WorldConfig)}
        cfg = WorldConfig(**{k: v for k, v in cfg_dict.items() if k in valid})
    else:
        cfg = WorldConfig(width=W, height=H)

    elevation = atlas[0].astype(np.float32)
    moisture = atlas[1].astype(np.float32)
    temperature_base = atlas[2].astype(np.float32)

    biome_path = root / "biome_map.npy"
    if biome_path.is_file():
        biome_map = np.load(biome_path).astype(np.int8)
    else:
        biome_map = np.clip(np.round(atlas[3] * 9.0), 0, 9).astype(np.int8)

    river_path = root / "river_mask.npy"
    if river_path.is_file():
        river_mask = np.load(river_path).astype(bool)
    else:
        river_mask = atlas[4] > 0.5

    river_distance = atlas[5].astype(np.float32)

    # flow_accumulation: atlas ch6 is log-normalized; use zeros (sim stepping does not need exact flow)
    flow_accumulation = np.zeros((H, W), dtype=np.int32)

    base_resources = {
        "soil_fertility": atlas[7].astype(np.float32),
        "water_proximity": atlas[8].astype(np.float32),
        "wood": atlas[9].astype(np.float32),
        "stone": atlas[10].astype(np.float32),
        "wild_food": atlas[11].astype(np.float32),
        "minerals": atlas[12].astype(np.float32),
    }

    return World(
        cfg=cfg,
        elevation=elevation,
        moisture=moisture,
        temperature_base=temperature_base,
        biome_map=biome_map,
        river_mask=river_mask,
        river_distance=river_distance,
        flow_accumulation=flow_accumulation,
        base_resources=base_resources,
    )


def bootstrap_payload(world: World, palette: dict) -> dict:
    """Build GET /api/v1/bootstrap JSON body."""
    import base64

    biome = world.biome_map.astype(np.int8)
    H, W = biome.shape
    biome_b64 = base64.standard_b64encode(biome.tobytes()).decode("ascii")
    return {
        "protocolVersion": 1,
        "H": H,
        "W": W,
        "seed": int(world.cfg.seed),
        "biome": biome_b64,
        "palette": palette,
    }
