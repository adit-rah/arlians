"""
Simulation facade — the PettingZoo-style interface the learner sees (build-spec §1.6).

FROZEN CONTRACT: the Obs/Actions/StepOut shapes and the Simulation method signatures.
The world is CONTINUING — `reset()` exists only for tests and the first run; training
never resets mid-stream. A dead slot is refilled only by a REPRODUCE action (a birth),
which is the learner's "auto-reset" (build-spec §3.2).

Method bodies are filled incrementally per phase. `step()` dispatches to the phased
update modules (dynamics / threats / reproduce); `observe()` delegates to observe.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from world import World
from .config import SimConfig
from .state import WorldState, EntityStore
from .curriculum import Curriculum
from . import observe as _observe


@dataclass
class Obs:
    """Observations for all M slots; dead slots are zeroed and excluded via alive_mask."""
    alive_mask: np.ndarray   # (M,) bool
    spatial: np.ndarray      # (M, N_SPATIAL, window_size, window_size) f32
    vector: np.ndarray       # (M, vector_len) f32


@dataclass
class Actions:
    """One action per slot. Entries for dead slots are ignored."""
    primary: np.ndarray      # (M,) int  — Action enum value
    param: np.ndarray        # (M,) int  — direction / structure type
    emit: np.ndarray         # (M,) int  — signal symbol in [0, n_symbols)


@dataclass
class StepOut:
    obs: Obs
    reward: np.ndarray       # (M,) f32 — per-living-agent; 0 for dead slots
    done: np.ndarray         # (M,) bool — True the step a slot's agent dies
    info: Dict[str, Any]     # metrics / diagnostics


class Simulation:
    def __init__(self, world: World, cfg: SimConfig, curriculum: Optional[Curriculum] = None):
        self.world = world
        self.cfg = cfg
        self.curriculum = curriculum if curriculum is not None else Curriculum(cfg)
        H, W = world.elevation.shape
        self.H, self.W = H, W
        self.state = WorldState.create(H, W)
        self.store = EntityStore.create(cfg)
        self.t: int = 0

    # ---- contract surface ----
    def reset(self, seed: Optional[int] = None) -> Obs:
        """Allocate fresh state and seed `init_agents` at spawn-eligible tiles.
        FROZEN signature; body filled in Phase 0/1."""
        raise NotImplementedError("reset() is implemented in Phase 0/1")

    def step(self, actions: Actions) -> StepOut:
        """Advance ONE day for all live agents (build-spec §2.1 ordering).
        FROZEN signature; body filled incrementally per phase."""
        raise NotImplementedError("step() is implemented incrementally per phase")

    def observe(self) -> Obs:
        return _observe.build_observation(self.world, self.state, self.store, self.t, self.cfg)

    @property
    def alive_mask(self) -> np.ndarray:
        return self.store.living_agents_mask()
