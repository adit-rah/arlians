"""
Mutable simulation state — the part of the world that changes over time.

Two containers, both frozen contracts (build-spec Part II §1.2, §1.3):

  WorldState  : per-tile mutable grid arrays, shape (H, W), aligned to world.World.
  EntityStore : struct-of-arrays for all entities (agents AND predators), length M.

Every field for the WHOLE project is declared here up front; fields belonging to a
later build phase are simply allocated and left at their zero/neutral value until
that phase wires them in. Subagents MUST NOT add/rename/remove fields without a
contract re-lock — they only start *using* a field at its phase.

`create()` allocates everything zeroed; seeding initial agents and setting non-zero
neutrals (e.g. genome=0.5) is the Simulation's job on reset().
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SimConfig


@dataclass
class WorldState:
    """Mutable per-tile grid state, shape (H, W). Static geography lives in world.World."""
    # phase 1
    wild_remaining: np.ndarray   # f32 — foraged food left on tile (init = base wild_food)
    # phase 3
    crop_stage: np.ndarray       # f32 — 0 = empty, (0,1] = growing -> mature
    crop_health: np.ndarray      # f32 — 1 = healthy; rots toward 0 below crop_min_fertility
    crop_owner: np.ndarray       # i32 — lineage_id that planted (-1 = none)
    # phase 4
    structure_type: np.ndarray   # i8  — 0 none / 1 shelter / 2 storage / 3 wall
    structure_hp: np.ndarray     # f32
    stored_food: np.ndarray      # f32 — contents of a storage tile
    # phase 6
    scent: np.ndarray            # f32 — agent-density field (decayed each step) for predator AI
    # phase 7
    event_mask: np.ndarray       # f32 — active-catastrophe multiplier overlay

    @classmethod
    def create(cls, height: int, width: int) -> "WorldState":
        z = lambda dt=np.float32: np.zeros((height, width), dtype=dt)  # noqa: E731
        return cls(
            wild_remaining=z(),
            crop_stage=z(),
            crop_health=z(),
            crop_owner=np.full((height, width), -1, dtype=np.int32),
            structure_type=z(np.int8),
            structure_hp=z(),
            stored_food=z(),
            scent=z(),
            event_mask=np.ones((height, width), dtype=np.float32),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.wild_remaining.shape  # type: ignore[return-value]


@dataclass
class EntityStore:
    """Struct-of-arrays for up to M entities. Agents and predators share the store
    (distinguished by `is_predator`). Index == slot id; dead slots are reused at birth."""
    # core (phase 1)
    alive: np.ndarray            # bool
    is_predator: np.ndarray      # bool (phase 6)
    y: np.ndarray                # i32
    x: np.ndarray                # i32
    energy: np.ndarray           # f32 (phase 1)
    hydration: np.ndarray        # f32 (phase 2)
    thermal: np.ndarray          # f32 (phase 4)
    health: np.ndarray           # f32 (phase 1) — death at <= 0
    age: np.ndarray              # i32 (phase 1) — steps alive
    # inventory (phase 1: food/wood; phase 6: minerals/weapon)
    inv_food: np.ndarray         # f32
    inv_wood: np.ndarray         # f32
    inv_stone: np.ndarray        # f32
    inv_minerals: np.ndarray     # f32
    weapon: np.ndarray           # bool (phase 6)
    # lineage / reproduction (phase 5)
    lineage_id: np.ndarray       # i32 — family id
    repro_cd: np.ndarray         # i32 — reproduction cooldown counter
    genome: np.ndarray           # f32 (M, G) — evolvable body traits; neutral 0.5
    # signaling (phase 7)
    last_signal: np.ndarray      # i8 — symbol emitted last step

    @classmethod
    def create(cls, cfg: SimConfig) -> "EntityStore":
        M = cfg.max_agents
        zf = lambda: np.zeros(M, dtype=np.float32)      # noqa: E731
        zi = lambda: np.zeros(M, dtype=np.int32)        # noqa: E731
        zb = lambda: np.zeros(M, dtype=bool)            # noqa: E731
        return cls(
            alive=zb(),
            is_predator=zb(),
            y=zi(),
            x=zi(),
            energy=zf(),
            hydration=zf(),
            thermal=zf(),
            health=zf(),
            age=zi(),
            inv_food=zf(),
            inv_wood=zf(),
            inv_stone=zf(),
            inv_minerals=zf(),
            weapon=zb(),
            lineage_id=np.full(M, -1, dtype=np.int32),
            repro_cd=zi(),
            genome=np.full((M, cfg.genome_dim), 0.5, dtype=np.float32),
            last_signal=np.zeros(M, dtype=np.int8),
        )

    @property
    def capacity(self) -> int:
        return self.alive.shape[0]

    def living_agents_mask(self) -> np.ndarray:
        """alive AND not a predator."""
        return self.alive & ~self.is_predator

    def n_living_agents(self) -> int:
        return int(self.living_agents_mask().sum())

    def free_slots(self) -> np.ndarray:
        """Indices of unoccupied slots (available for births / predator spawns)."""
        return np.flatnonzero(~self.alive)
