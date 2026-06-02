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
    """
    raise NotImplementedError("build_mask is implemented incrementally per phase")
