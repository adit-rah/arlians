"""
World dynamics — pure functions advancing mutable state one step (build-spec §2.3-2.5).

Filled incrementally by the env-dynamics fleet role:
  Phase 1: forage depletion/regrowth (§2.4), drive decay + health update (§2.5)
  Phase 3: crop growth + winter-failure rot (§2.3)
  Phase 4: thermal update, spoilage, structure decay

All functions operate in-place / vectorized on WorldState + EntityStore, driven by
SimConfig + the world's seasonal scalars. No magic numbers — everything from SimConfig.
"""
from __future__ import annotations

import numpy as np

from .config import SimConfig
from .state import EntityStore, WorldState
from .reproduce import traits_from_genome


# ---------------------------------------------------------------------------
# Phase 1 drive dynamics
# ---------------------------------------------------------------------------

def decay_drives(store: EntityStore, cfg: SimConfig) -> None:
    """
    Decay per-step energy and hydration for all LIVING, non-predator agents.

    Phase 1: energy decays.
      energy -= cfg.energy_decay * metab   (clip >= 0)

    Phase 2: hydration also decays.
      hydration -= cfg.hydration_decay * water_need   (clip >= 0)

    Thermal is not wired until Phase 4; it stays at its init value.
    """
    living = store.alive & ~store.is_predator
    if not living.any():
        return

    traits = traits_from_genome(store.genome)
    # energy -= energy_decay * metab; only for living agents
    store.energy[living] = np.clip(
        store.energy[living] - cfg.energy_decay * traits.metab[living],
        0.0, None,
    ).astype(np.float32)
    # hydration -= hydration_decay * water_need; only for living agents (Phase 2)
    store.hydration[living] = np.clip(
        store.hydration[living] - cfg.hydration_decay * traits.water_need[living],
        0.0, None,
    ).astype(np.float32)


def update_health(store: EntityStore, cfg: SimConfig) -> None:
    """
    Regenerate or damage health based on homeostatic drive status.

    Phase 2 rules (energy AND hydration are active drives; thermal not yet):
      - If energy >= cfg.drive_safe_band AND hydration >= cfg.drive_safe_band
                                          -> health += cfg.health_regen  (clip <= 1)
      - If energy == 0                    -> health -= cfg.starve_damage
      - If hydration == 0                 -> health -= cfg.starve_damage
        (each drive at 0 contributes independently; both at 0 → 2× damage)
    """
    living = store.alive & ~store.is_predator
    if not living.any():
        return

    energy    = store.energy[living]
    hydration = store.hydration[living]
    health    = store.health[living]

    # Both active drives must be in safe band to regen health
    safe = (energy >= cfg.drive_safe_band) & (hydration >= cfg.drive_safe_band)
    health = np.where(safe, health + cfg.health_regen, health)

    # Each drive at zero deals independent starve_damage
    starving    = energy    <= 0.0
    dehydrated  = hydration <= 0.0
    damage = (starving.astype(np.float32) + dehydrated.astype(np.float32)) * cfg.starve_damage
    health = health - damage

    store.health[living] = np.clip(health, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 1 foraging & eating
# ---------------------------------------------------------------------------

def forage(
    store: EntityStore,
    state: WorldState,
    idx: np.ndarray,
    cfg: SimConfig,
) -> None:
    """
    Execute FORAGE for the agents indicated by `idx` (integer array of slot indices).

    Each foraging agent on their tile takes:
        take = min(capacity - inv_food, wild_remaining[tile], cfg.forage_yield)
    and that amount is deducted from wild_remaining and added to inv_food.

    `idx` must be pre-filtered to contain only living agents that chose FORAGE.
    The order of idx is the conflict-resolution order (ascending slot index for
    determinism); agents processed later see the updated wild_remaining.

    Uses traits_from_genome to get per-agent carry capacity.
    """
    if idx.size == 0:
        return

    traits = traits_from_genome(store.genome)
    # Per-agent carry capacity = base * genome capacity trait
    cap_per_agent = (cfg.carry_capacity * traits.capacity[idx]).astype(np.float32)

    ys = store.y[idx]
    xs = store.x[idx]

    # Process tile by tile for correct conflict resolution.
    # Group foragers by tile so we apply depletion atomically per tile;
    # within a tile, process in the order they appear in idx (ascending slot).
    # We iterate unique tiles but keep the order from idx for within-tile ordering.
    for slot_pos, (slot, y, x, cap) in enumerate(zip(idx, ys, xs, cap_per_agent)):
        available = state.wild_remaining[y, x]
        if available <= 0.0:
            continue
        room = cap - store.inv_food[slot]
        if room <= 0.0:
            continue
        take = float(min(room, available, cfg.forage_yield))
        if take <= 0.0:
            continue
        state.wild_remaining[y, x] = max(0.0, available - take)
        store.inv_food[slot] = min(cap, store.inv_food[slot] + take)


def eat(store: EntityStore, idx: np.ndarray, cfg: SimConfig) -> None:
    """
    Convert carried food to energy for agents in `idx` that chose EAT.

    Mechanics:
        need     = (1 - energy) / cfg.eat_restore   [food units needed to fill up]
        consumed = min(inv_food, need)
        energy  += cfg.eat_restore * consumed        (clip <= 1)
        inv_food -= consumed                         (clip >= 0)

    `idx` must be pre-filtered to living agents that chose EAT.
    """
    if idx.size == 0:
        return

    energy   = store.energy[idx]
    inv_food = store.inv_food[idx]

    need     = (1.0 - energy) / cfg.eat_restore
    consumed = np.minimum(inv_food, need).astype(np.float32)
    consumed = np.maximum(consumed, 0.0)

    store.energy[idx]   = np.clip(energy + cfg.eat_restore * consumed, 0.0, 1.0).astype(np.float32)
    store.inv_food[idx] = np.maximum(inv_food - consumed, 0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 2 drinking
# ---------------------------------------------------------------------------

def drink(
    store: EntityStore,
    water_proximity: np.ndarray,
    idx: np.ndarray,
    cfg: SimConfig,
) -> None:
    """
    Execute DRINK for the agents indicated by `idx` (integer array of slot indices).

    For each agent in `idx` whose tile has water_proximity >= cfg.drink_min_water,
    hydration is restored:
        hydration = min(1.0, hydration + cfg.drink_restore)

    Agents on a dry tile (water_proximity < drink_min_water) are no-ops — the mask
    should have already prevented this, but we guard here for safety.

    `idx` must be pre-filtered to contain only living agents that chose DRINK.
    `water_proximity` is a (H, W) float32 array (base_resources or get_obs channel 8).
    """
    if idx.size == 0:
        return

    ys = store.y[idx]
    xs = store.x[idx]
    tile_water = water_proximity[ys, xs]

    # Only agents on sufficiently watered tiles benefit
    can_drink = tile_water >= cfg.drink_min_water
    drink_idx = idx[can_drink]

    if drink_idx.size == 0:
        return

    store.hydration[drink_idx] = np.clip(
        store.hydration[drink_idx] + cfg.drink_restore, 0.0, 1.0
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Phase 1 world regeneration & spoilage
# ---------------------------------------------------------------------------

def regrow_wild(state: WorldState, world, t: int, cfg: SimConfig) -> None:
    """
    Regrow wild food toward the seasonal cap each step.

    cap = world.base_resources["wild_food"] * season.wild_food_modifier
    wild_remaining += cfg.forage_regen * (cap - wild_remaining)
    Clip to >= 0 (can't go below zero).
    """
    from world.seasons import compute_season_state
    season = compute_season_state(t, world.cfg)
    cap = (world.base_resources["wild_food"] * season.wild_food_modifier).astype(np.float32)
    state.wild_remaining += cfg.forage_regen * (cap - state.wild_remaining)
    np.clip(state.wild_remaining, 0.0, None, out=state.wild_remaining)


def spoil_carried(store: EntityStore, cfg: SimConfig) -> None:
    """
    Spoil food carried in agent inventories each step.

        inv_food -= cfg.spoilage_carried   (clip >= 0)

    Only living agents are affected.
    """
    living = store.alive & ~store.is_predator
    if not living.any():
        return
    store.inv_food[living] = np.maximum(
        store.inv_food[living] - cfg.spoilage_carried, 0.0
    ).astype(np.float32)
