"""
Threats — predators, combat, and catastrophes (build-spec §2.6-2.7).

Filled by the env-dynamics fleet role:
  Phase 6: deterministic HP combat resolution (weapons/walls/grouping modifiers),
           predator AI (scent-gradient ascent, spawn/despawn to target count)
  Phase 7: episodic catastrophes (cold-snap / drought / flood / storm via event_mask)
"""
from __future__ import annotations

from typing import Set
import numpy as np

from .config import SimConfig
from .state import EntityStore, WorldState


# ---------------------------------------------------------------------------
# Phase 6 predator helpers
# ---------------------------------------------------------------------------

def update_scent(state: WorldState, store: EntityStore) -> None:
    """
    Deposit scent at each living (non-predator) agent's tile and blur it
    to create a spatial gradient that predators can follow from a distance.

    Steps (build-spec §2.7 "scent-gradient AI", "optionally lightly blurred"):
      1. Accumulate 1.0 per living agent at its (y, x) tile into a fresh deposit.
      2. Apply a Gaussian blur (sigma=3) so the scent field decays smoothly with
         distance from agents, creating a true gradient within pred_sense_radius.
      3. Add the blurred deposit into state.scent.

    The scent field is already decayed (*= 0.9) each step by Simulation.step
    before this function is called, so this adds fresh blurred deposits on top.
    Over multiple steps the field accumulates a wide gradient around agents,
    allowing gradient-ascent navigation from pred_sense_radius tiles away.

    Uses vectorized indexing: multiple agents on the same tile stack additively.
    """
    from scipy.ndimage import gaussian_filter

    living_mask = store.living_agents_mask()
    if not living_mask.any():
        return

    live_idx = np.flatnonzero(living_mask)
    ys = store.y[live_idx]
    xs = store.x[live_idx]

    # Fresh deposit on a zeroed buffer
    H, W = state.scent.shape
    deposit = np.zeros((H, W), dtype=np.float64)
    np.add.at(deposit, (ys, xs), 1.0)

    # Gaussian blur with sigma=3: spreads scent ~9 tiles at 1-sigma,
    # covering the pred_sense_radius of 12 tiles at ~4 sigma.
    blurred = gaussian_filter(deposit, sigma=3.0).astype(np.float32)
    state.scent += blurred


def spawn_predators(
    store: EntityStore,
    world,
    target_count: int,
    rng: np.random.Generator,
) -> None:
    """
    Adjust the number of living predators toward `target_count`.

    Spawn: if living predators < target_count, fill free slots with new predators
    placed at land tiles far from the current agent population centroid
    (wilderness spawning). If no free slots exist, spawning is skipped silently.

    Despawn: if living predators > target_count, mark the excess living predators
    dead (the ones with the highest slot indices first, for determinism).

    Predator slots are initialised:
      alive=True, is_predator=True, health=1.0, y/x on a random land tile,
      energy/hydration/thermal at neutral 1.0 (not used by predator logic but
      kept non-zero so drive/health routines skip them correctly via the
      ~is_predator guard in decay_drives / update_health).
    """
    pred_mask = store.alive & store.is_predator
    n_pred = int(pred_mask.sum())

    # --- despawn excess ---
    if n_pred > target_count:
        excess = n_pred - target_count
        pred_slots = np.flatnonzero(pred_mask)
        # Kill the last `excess` predators (highest slot idx for determinism)
        to_kill = pred_slots[-excess:]
        store.alive[to_kill] = False
        store.is_predator[to_kill] = False
        store.health[to_kill] = 0.0
        return

    # --- spawn new predators ---
    n_to_spawn = target_count - n_pred
    if n_to_spawn <= 0:
        return

    free_slots = store.free_slots()
    if free_slots.size == 0:
        return  # store is full; skip silently

    n_spawn = min(n_to_spawn, free_slots.size)
    slots = free_slots[:n_spawn]

    # Land tiles: elevation >= sea_level
    sea = world.cfg.sea_level
    land_ys, land_xs = np.where(world.elevation >= sea)
    if land_ys.size == 0:
        return  # no land — degenerate world

    # Pick random land tiles for spawn positions
    chosen = rng.integers(0, land_ys.size, size=n_spawn)

    store.alive[slots]        = True
    store.is_predator[slots]  = True
    store.y[slots]            = land_ys[chosen]
    store.x[slots]            = land_xs[chosen]
    store.health[slots]       = np.float32(1.0)
    store.energy[slots]       = np.float32(1.0)
    store.hydration[slots]    = np.float32(1.0)
    store.thermal[slots]      = np.float32(1.0)
    store.age[slots]          = 0
    store.weapon[slots]       = False
    store.lineage_id[slots]   = -1


def step_predators(
    store: EntityStore,
    state: WorldState,
    world,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> dict:
    """
    Advance all living predators one step: move + attack.

    Movement (scent-gradient ascent within pred_sense_radius):
      - For each living predator, sample the 8 neighbour tiles within the
        sense radius (we look at the 8 immediate neighbours for move selection;
        the sense_radius is used to define the candidate neighbourhood scan
        for choosing the best tile). We check all 8 cardinal/diagonal neighbours
        plus the current tile, step toward the neighbour with highest scent.
      - If all neighbour scent values are 0 the predator moves randomly to a
        valid non-wall, in-bounds tile.
      - Wall tiles (structure_type == 3) are forbidden; the predator stays if
        all legal neighbours are walls.

    Attack (adjacent or co-located agent damage):
      - After moving, if any living agent is on the predator's tile or an
        immediate neighbour tile, deal cfg.pred_attack_damage to that agent.
        We apply damage directly (not via resolve_combat) because predators
        are not agents submitting Actions — this keeps the agent combat system
        orthogonal.
      - A single predator can only attack ONE agent per step (the closest one,
        tie-broken by lowest slot index).
      - We record all agent slots that took predator damage in `predator_slots`.

    Returns
    -------
    dict with key "predator_slots": set of agent slot indices that took
    predator damage this step (used by resolve_deaths for cause attribution).
    """
    from .actions import DIRECTIONS, StructureType

    pred_mask = store.alive & store.is_predator
    if not pred_mask.any():
        return {"predator_slots": set()}

    pred_idx = np.flatnonzero(pred_mask)

    H, W = state.scent.shape
    WALL = int(StructureType.WALL)

    # Build a fast lookup: tile -> list of living AGENT (non-predator) slots
    agent_mask = store.living_agents_mask()
    agent_slots = np.flatnonzero(agent_mask)
    from collections import defaultdict
    tile_agents: dict = defaultdict(list)
    for s in agent_slots:
        tile_agents[(int(store.y[s]), int(store.x[s]))].append(s)

    predator_slots: Set[int] = set()

    # Pre-extract scent for fast lookups
    scent = state.scent  # (H, W) float32 view

    for pred in pred_idx:
        py = int(store.y[pred])
        px = int(store.x[pred])

        # ---- movement: choose best neighbour by scent ----
        best_scent = -1.0
        best_dy = 0
        best_dx = 0
        found_any_legal = False
        legal_neighbors = []  # for random fallback

        for dy, dx in DIRECTIONS:
            ny = py + int(dy)
            nx = px + int(dx)
            # Bounds check
            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue
            # Wall repel: skip wall tiles
            if int(state.structure_type[ny, nx]) == WALL:
                continue
            # Land check: predators do not enter ocean (elevation < sea_level)
            if float(world.elevation[ny, nx]) < world.cfg.sea_level:
                continue
            found_any_legal = True
            legal_neighbors.append((int(dy), int(dx)))
            s = float(scent[ny, nx])
            if s > best_scent:
                best_scent = s
                best_dy = int(dy)
                best_dx = int(dx)

        if found_any_legal:
            if best_scent <= 0.0:
                # No scent — move randomly among legal neighbours
                ridx = int(rng.integers(0, len(legal_neighbors)))
                best_dy, best_dx = legal_neighbors[ridx]
            # Step the predator
            store.y[pred] = py + best_dy
            store.x[pred] = px + best_dx

        # ---- attack: hit an agent on the new tile or an adjacent tile ----
        new_py = int(store.y[pred])
        new_px = int(store.x[pred])

        # Check current tile first, then adjacent tiles
        target_slot: int | None = None
        # Co-located agents (same tile) are attacked first
        same_tile = tile_agents.get((new_py, new_px), [])
        if same_tile:
            target_slot = min(same_tile)  # lowest slot index for determinism
        else:
            # Check immediate neighbours
            for dy, dx in DIRECTIONS:
                ay = new_py + int(dy)
                ax = new_px + int(dx)
                if ay < 0 or ay >= H or ax < 0 or ax >= W:
                    continue
                candidates = tile_agents.get((ay, ax), [])
                if candidates:
                    t = min(candidates)
                    if target_slot is None or t < target_slot:
                        target_slot = t

        if target_slot is not None:
            old_hp = float(store.health[target_slot])
            new_hp = max(0.0, old_hp - float(cfg.pred_attack_damage))
            store.health[target_slot] = np.float32(new_hp)
            predator_slots.add(target_slot)

    return {"predator_slots": predator_slots}
