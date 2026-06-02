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
        H, W = self.H, self.W

        # Re-allocate mutable state from scratch
        self.state = WorldState.create(H, W)
        self.store = EntityStore.create(self.cfg)
        self.t = 0

        # Initialise wild_remaining from the static base resource layer
        self.state.wild_remaining[:] = self.world.base_resources["wild_food"]

        # Spawn-eligible tiles: land (elevation >= sea_level)
        elev = self.world.elevation                       # (H, W) float32
        sea  = self.world.cfg.sea_level
        land_ys, land_xs = np.where(elev >= sea)          # flat index arrays over land tiles
        n_land = land_ys.shape[0]
        if n_land == 0:
            raise RuntimeError("No land tiles available for agent spawning")

        rng = np.random.default_rng(seed)
        chosen = rng.integers(0, n_land, size=self.cfg.init_agents)
        spawn_y = land_ys[chosen]
        spawn_x = land_xs[chosen]

        n = self.cfg.init_agents
        slots = np.arange(n)  # first n slots in the store

        self.store.alive[slots]       = True
        self.store.is_predator[slots] = False
        self.store.y[slots]           = spawn_y
        self.store.x[slots]           = spawn_x
        self.store.energy[slots]      = 1.0
        self.store.hydration[slots]   = 1.0
        self.store.thermal[slots]     = 1.0
        self.store.health[slots]      = 1.0
        self.store.age[slots]         = 0
        self.store.genome[slots]      = 0.5   # neutral body traits
        self.store.lineage_id[slots]  = slots  # each agent starts its own lineage

        return self.observe()

    def step(self, actions: Actions) -> StepOut:
        """Advance ONE day for all live agents (build-spec §2.1 ordering).
        FROZEN signature; body filled incrementally per phase.
        Phase 0: NO-OP — advance clock, decay scent, skip all action processing."""
        self.t += 1
        self.state.scent *= 0.9

        M = self.cfg.max_agents
        obs  = self.observe()
        return StepOut(
            obs=obs,
            reward=np.zeros(M, dtype=np.float32),
            done=np.zeros(M, dtype=bool),
            info={"t": self.t, "n_agents": self.store.n_living_agents()},
        )

    def observe(self) -> Obs:
        return _observe.build_observation(self.world, self.state, self.store, self.t, self.cfg)

    @property
    def alive_mask(self) -> np.ndarray:
        return self.store.living_agents_mask()
