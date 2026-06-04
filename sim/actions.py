"""
Action space — frozen contract (build-spec Part II §1.5).

Multi-discrete: three independent heads sampled per step per agent.
  primary : Discrete(N_PRIMARY)  — what to do
  param   : Discrete(N_PARAM)    — direction (MOVE/ATTACK) or structure type (BUILD)
  emit    : Discrete(n_symbols)  — signal symbol; fires alongside primary, costs nothing

Invalid (primary, param) combinations are masked to -inf before sampling; the mask is
produced by `build_mask`. Implementing the mask logic is later-phase work — only the
SHAPE/semantics here are frozen.
"""
from __future__ import annotations

from enum import IntEnum
import numpy as np

from .config import SimConfig


class Action(IntEnum):
    """primary head values."""
    NOOP = 0
    MOVE = 1
    FORAGE = 2
    DRINK = 3
    PLANT = 4
    HARVEST = 5
    BUILD = 6
    CRAFT = 7        # craft a weapon
    EAT = 8
    STORE = 9
    RETRIEVE = 10
    ATTACK = 11
    REST = 12
    REPRODUCE = 13


N_PRIMARY = len(Action)   # 14


class StructureType(IntEnum):
    """Matches WorldState.structure_type values; also the BUILD param encoding (1..3)."""
    NONE = 0
    SHELTER = 1
    STORAGE = 2
    WALL = 3


# param head: 8 values. For MOVE/ATTACK these are the 8 compass directions;
# for BUILD, values 1..3 select StructureType (0 / 4..7 unused -> masked).
N_PARAM = 8

# (dy, dx) for param direction 0..7: N, NE, E, SE, S, SW, W, NW
DIRECTIONS = np.array(
    [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)],
    dtype=np.int32,
)


def build_mask(world, state, store, cfg: SimConfig) -> np.ndarray:
    """
    Return a boolean validity mask of shape (M, N_PRIMARY): True where the primary
    action is currently legal for that agent (dead slots all-False).

    FROZEN: signature + return shape. Implementation is phased (Phase 1+ fill rules:
    HARVEST only where crop_stage>=1, REPRODUCE only if energy>=threshold & cd==0 &
    a free slot exists, BUILD only with sufficient inventory, etc.).

    Phase 1 rules:
      - Dead slots: all False.
      - NOOP, MOVE, REST: always True for living agents.
      - FORAGE: True only if wild_remaining[tile] > 0.
      - EAT: True only if inv_food > 0.

    Phase 2 rules (added):
      - DRINK: True only if water_proximity[tile] >= cfg.drink_min_water.
        (uses static base_resources["water_proximity"] for consistency with step)

    All other actions (PLANT, HARVEST, BUILD, CRAFT, STORE, RETRIEVE,
    ATTACK, REPRODUCE): False until their phase is implemented.
    """
    M = cfg.max_agents
    mask = np.zeros((M, N_PRIMARY), dtype=bool)

    living = store.alive & ~store.is_predator
    if not living.any():
        return mask

    live_idx = np.flatnonzero(living)

    # NOOP, MOVE, REST always allowed
    mask[live_idx, Action.NOOP]  = True
    mask[live_idx, Action.MOVE]  = True
    mask[live_idx, Action.REST]  = True

    # FORAGE: only where wild_remaining > 0 at agent's tile
    ys = store.y[live_idx]
    xs = store.x[live_idx]
    has_wild = state.wild_remaining[ys, xs] > 0.0
    mask[live_idx[has_wild], Action.FORAGE] = True

    # EAT: only if carrying food
    has_food = store.inv_food[live_idx] > 0.0
    mask[live_idx[has_food], Action.EAT] = True

    # DRINK (Phase 2): only where water_proximity >= drink_min_water
    water_prox = world.base_resources["water_proximity"]
    near_water = water_prox[ys, xs] >= cfg.drink_min_water
    mask[live_idx[near_water], Action.DRINK] = True

    # PLANT (Phase 3): on land tiles (elevation >= sea_level) where crop_stage == 0
    sea_level  = world.cfg.sea_level
    on_land    = world.elevation[ys, xs] >= sea_level
    tile_empty = state.crop_stage[ys, xs] == 0.0
    mask[live_idx[on_land & tile_empty], Action.PLANT] = True

    # HARVEST (Phase 3): only where crop_stage >= 1.0 (mature)
    tile_mature = state.crop_stage[ys, xs] >= 1.0
    mask[live_idx[tile_mature], Action.HARVEST] = True

    # BUILD (Phase 4): on land tiles with no existing structure (structure_type==0),
    # where the agent can afford at least shelter or storage.
    from .reproduce import traits_from_genome
    traits = traits_from_genome(store.genome)

    no_structure   = state.structure_type[ys, xs] == 0
    buildable_land = on_land & no_structure

    shelter_wood  = cfg.shelter_cost.get("wood",  0.0)
    shelter_stone = cfg.shelter_cost.get("stone", 0.0)
    storage_wood  = cfg.storage_cost.get("wood",  0.0)
    storage_stone = cfg.storage_cost.get("stone", 0.0)

    can_afford_shelter = (
        (store.inv_wood[live_idx]  >= shelter_wood)
        & (store.inv_stone[live_idx] >= shelter_stone)
    )
    can_afford_storage = (
        (store.inv_wood[live_idx]  >= storage_wood)
        & (store.inv_stone[live_idx] >= storage_stone)
    )
    can_afford_any = can_afford_shelter | can_afford_storage
    mask[live_idx[buildable_land & can_afford_any], Action.BUILD] = True

    # STORE (Phase 4): on a storage tile (structure_type==2) while carrying food
    on_storage   = state.structure_type[ys, xs] == 2
    has_food_inv = store.inv_food[live_idx] > 0.0
    mask[live_idx[on_storage & has_food_inv], Action.STORE] = True

    # RETRIEVE (Phase 4): on a storage tile with stored_food > 0
    has_stored = state.stored_food[ys, xs] > 0.0
    mask[live_idx[on_storage & has_stored], Action.RETRIEVE] = True

    # REPRODUCE (Phase 5): energy >= repro_energy_threshold AND repro_cd == 0
    # AND at least one free slot exists globally (soft cap check).
    if store.free_slots().size > 0:
        can_repro = (
            (store.energy[live_idx] >= cfg.repro_energy_threshold)
            & (store.repro_cd[live_idx] == 0)
        )
        mask[live_idx[can_repro], Action.REPRODUCE] = True

    return mask
