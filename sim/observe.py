"""
Observation tensor layout — frozen contract (build-spec Part II §1.4).

Each living agent observes:
  spatial : (N_SPATIAL, window_size, window_size)  — a local map window, zero-padded
  vector  : (vector_len,)                          — interoceptive + genome state

This module also OWNS the vectorized window construction that replaces
`world.get_window` (which is a slow Python double-loop). The Phase-0 fleet implements
`build_observation`; the channel indices below are the frozen part and must not move.
"""
from __future__ import annotations

import numpy as np

from .config import SimConfig

# ---- spatial channel layout (analogue of world.LAYER_NAMES) ----
# Channels 0-13 are exactly world.get_obs(t): 7 geo + 6 seasonal resources + season_phase.
SPATIAL_CHANNELS = [
    # 0-13: from world.get_obs(t)
    "w_elevation", "w_moisture", "w_temperature", "w_biome", "w_river_mask",
    "w_river_distance", "w_flow_accum", "w_soil_fertility", "w_water_proximity",
    "w_wood", "w_stone", "w_wild_food", "w_minerals", "w_season_phase",
    # 14-19: mutable world state
    "wild_remaining", "crop_stage", "crop_health",
    "structure_type", "structure_hp", "stored_food",
    # 20-23: entities + signal
    "agent_count", "ally_health_mean", "predator_presence", "nearby_signal",
]
N_SPATIAL = len(SPATIAL_CHANNELS)  # 24
WORLD_OBS_CHANNELS = 14            # channels 0..13 come straight from world.get_obs

# ---- interoceptive vector layout ----
# [energy, hydration, thermal, health, age_norm, inv_food, inv_wood, inv_stone,
#  inv_minerals, weapon, *genome(G)]  -> length = 10 + G
INTERO_BASE = [
    "energy", "hydration", "thermal", "health", "age_norm",
    "inv_food", "inv_wood", "inv_stone", "inv_minerals", "weapon",
]
INTERO_BASE_LEN = len(INTERO_BASE)  # 10


def vector_len(cfg: SimConfig) -> int:
    """Length of the interoceptive vector: 10 base dims + G genome dims."""
    return INTERO_BASE_LEN + cfg.genome_dim


def build_observation(world, state, store, t: int, cfg: SimConfig):
    """
    Build observations for ALL living agents in one vectorized pass.

    Returns an Obs (see simulation.Obs):
      alive_mask : (M,) bool
      spatial    : (M, N_SPATIAL, window_size, window_size) f32  (dead slots = 0)
      vector     : (M, vector_len(cfg)) f32                        (dead slots = 0)

    Implementation approach (build-spec §2.2): assemble the full (N_SPATIAL, H, W)
    stack once per step, np.pad by window_radius, then gather every agent's window
    with fancy indexing — NO per-agent Python loop, NO per-agent world.get_obs.

    FROZEN: signature + return layout. Implemented by the Phase-0 fleet, with the
    correctness-pin test asserting channels 0-13 equal a reference world.get_window.
    """
    raise NotImplementedError("build_observation is implemented in Phase 0")
