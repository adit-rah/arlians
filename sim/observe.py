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
    from .simulation import Obs  # local import to avoid circular at module level

    M = cfg.max_agents
    H, W = state.shape
    r = cfg.window_radius
    ws = cfg.window_size  # 2r+1

    alive_mask = store.living_agents_mask()  # (M,) bool

    # ------------------------------------------------------------------
    # 1.  Build the full (N_SPATIAL, H, W) stack once
    # ------------------------------------------------------------------
    # Channels 0-13: world.get_obs(t)
    world_obs = world.get_obs(t)  # (14, H, W) float32

    # Channels 14-19: mutable WorldState
    mutable = np.stack([
        state.wild_remaining,                                    # 14
        state.crop_stage,                                        # 15
        state.crop_health,                                       # 16
        state.structure_type.astype(np.float32) / 3.0,          # 17 (normalised)
        state.structure_hp,                                      # 18
        state.stored_food / cfg.storage_capacity,                # 19
    ], axis=0)  # (6, H, W)

    # Channels 20-23: entity-derived — scatter living entities onto the grid
    agent_count    = np.zeros((H, W), dtype=np.float32)   # 20
    ally_health    = np.zeros((H, W), dtype=np.float32)   # accumulator for 21
    ally_health_n  = np.zeros((H, W), dtype=np.float32)   # count for mean
    pred_presence  = np.zeros((H, W), dtype=np.float32)   # 22
    signal_sum     = np.zeros((H, W), dtype=np.float32)   # accumulator for 23

    living_idx   = np.flatnonzero(alive_mask)
    predator_idx = np.flatnonzero(store.alive & store.is_predator)

    if living_idx.size > 0:
        ys = store.y[living_idx]
        xs = store.x[living_idx]
        np.add.at(agent_count, (ys, xs), 1.0)
        np.add.at(ally_health, (ys, xs), store.health[living_idx])
        np.add.at(ally_health_n, (ys, xs), 1.0)
        # signal: normalise symbol value to [0,1]
        np.add.at(signal_sum, (ys, xs), store.last_signal[living_idx].astype(np.float32))

    if predator_idx.size > 0:
        py = store.y[predator_idx]
        px = store.x[predator_idx]
        np.add.at(pred_presence, (py, px), 1.0)
        pred_presence = np.clip(pred_presence, 0.0, 1.0)

    # ally_health_mean: mean health of living agents on tile; 0 where no agents
    with np.errstate(invalid="ignore", divide="ignore"):
        ally_health_mean = np.where(ally_health_n > 0, ally_health / ally_health_n, 0.0).astype(np.float32)

    # nearby_signal: mean (normalised) signal per tile
    with np.errstate(invalid="ignore", divide="ignore"):
        nearby_signal = np.where(
            ally_health_n > 0,
            (signal_sum / ally_health_n) / max(cfg.n_symbols - 1, 1),
            0.0,
        ).astype(np.float32)

    entity_channels = np.stack([
        agent_count,        # 20
        ally_health_mean,   # 21
        pred_presence,      # 22
        nearby_signal,      # 23
    ], axis=0)  # (4, H, W)

    # Full stack (N_SPATIAL, H, W)
    full_stack = np.concatenate([world_obs, mutable, entity_channels], axis=0).astype(np.float32)
    # shape: (24, H, W)

    # ------------------------------------------------------------------
    # 2.  Pad the stack by window_radius so edge agents get zero-fill
    # ------------------------------------------------------------------
    # np.pad with (r, r) on H and W axes; channel axis gets (0, 0)
    padded = np.pad(full_stack, ((0, 0), (r, r), (r, r)), mode="constant", constant_values=0.0)
    # shape: (24, H+2r, W+2r)

    # ------------------------------------------------------------------
    # 3.  Gather every agent's window with fancy indexing — NO Python loop
    # ------------------------------------------------------------------
    # For a living agent at grid position (y, x) the padded indices are (y+r, x+r).
    # We want the window [y+r-r .. y+r+r] = [y .. y+2r] in the padded array.
    # We collect ALL windows via a single advanced-index operation.
    #
    # row_starts[i] = padded_y for agent i = store.y[i]  (since padding adds r)
    # Actually: in the padded array agent at (y, x) -> center at (y+r, x+r).
    # window rows: y+r-r .. y+r+r = y .. y+2r  (ws=2r+1 rows)
    # window cols: x+r-r .. x+r+r = x .. x+2r  (ws=2r+1 cols)

    # Build spatial output array; dead slots stay zero.
    spatial = np.zeros((M, N_SPATIAL, ws, ws), dtype=np.float32)

    if living_idx.size > 0:
        ys_all = store.y[living_idx]   # (n_live,) — grid y, not padded
        xs_all = store.x[living_idx]   # (n_live,) — grid x, not padded

        # Offsets within the window: (ws,) arrays
        offsets = np.arange(ws, dtype=np.int32)  # 0 .. 2r

        # Row indices into padded array: shape (n_live, ws)
        row_idx = ys_all[:, None] + offsets[None, :]   # broadcast
        # Col indices into padded array: shape (n_live, ws)
        col_idx = xs_all[:, None] + offsets[None, :]

        # Fancy index: padded[:, row_idx, col_idx] would give (C, n_live, ws) each.
        # We need (n_live, C, ws, ws). Build the full gather in one op:
        # padded shape: (C, H+2r, W+2r)
        # We want: out[n, c, wr, wc] = padded[c, ys_all[n]+wr, xs_all[n]+wc]
        #
        # Use: padded[:, row_idx[:, :, None], col_idx[:, None, :]]
        #   -> shape (C, n_live, ws, ws) — then transpose to (n_live, C, ws, ws)
        windows = padded[:, row_idx[:, :, None], col_idx[:, None, :]]
        # shape: (C, n_live, ws, ws)
        spatial[living_idx] = np.transpose(windows, (1, 0, 2, 3))

    # ------------------------------------------------------------------
    # 4.  Build the interoceptive vector, shape (M, vector_len)
    # ------------------------------------------------------------------
    vlen = INTERO_BASE_LEN + cfg.genome_dim
    vector = np.zeros((M, vlen), dtype=np.float32)

    if living_idx.size > 0:
        age_norm = store.age[living_idx].astype(np.float32) / 1000.0
        vector[living_idx, 0]  = store.energy[living_idx]
        vector[living_idx, 1]  = store.hydration[living_idx]
        vector[living_idx, 2]  = store.thermal[living_idx]
        vector[living_idx, 3]  = store.health[living_idx]
        vector[living_idx, 4]  = age_norm
        vector[living_idx, 5]  = store.inv_food[living_idx]
        vector[living_idx, 6]  = store.inv_wood[living_idx]
        vector[living_idx, 7]  = store.inv_stone[living_idx]
        vector[living_idx, 8]  = store.inv_minerals[living_idx]
        vector[living_idx, 9]  = store.weapon[living_idx].astype(np.float32)
        vector[living_idx, 10:10 + cfg.genome_dim] = store.genome[living_idx]

    return Obs(alive_mask=alive_mask, spatial=spatial, vector=vector)
