"""
Viz snapshots for delta diffing (stream protocol v1).

Captures only fields the client renderer needs — not full EntityStore/genome.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim.state import EntityStore, WorldState


@dataclass
class EntityViz:
    alive: bool
    y: int
    x: int
    is_predator: bool


@dataclass
class VizSnapshot:
    """Frozen view of sim state at one timestep."""

    t: int
    season_phase: float
    n_living: int
    wild_remaining: np.ndarray
    crop_stage: np.ndarray
    crop_health: np.ndarray
    structure_type: np.ndarray
    stored_food: np.ndarray
    event_mask: np.ndarray
    entities: dict[int, EntityViz] = field(default_factory=dict)

    @classmethod
    def from_sim(cls, sim, season_phase: float) -> "VizSnapshot":
        state: WorldState = sim.state
        store: EntityStore = sim.store
        living = store.living_agents_mask()
        preds = store.alive & store.is_predator
        n_living = int((living | preds).sum())

        entities: dict[int, EntityViz] = {}
        for slot in np.flatnonzero(store.alive):
            s = int(slot)
            entities[s] = EntityViz(
                alive=True,
                y=int(store.y[s]),
                x=int(store.x[s]),
                is_predator=bool(store.is_predator[s]),
            )

        return cls(
            t=int(sim.t),
            season_phase=float(season_phase),
            n_living=n_living,
            wild_remaining=state.wild_remaining,
            crop_stage=state.crop_stage,
            crop_health=state.crop_health,
            structure_type=state.structure_type,
            stored_food=state.stored_food,
            event_mask=state.event_mask,
            entities=entities,
        )

    def copy_arrays(self) -> "VizSnapshot":
        """Deep-copy grid arrays for use as previous snapshot."""
        return VizSnapshot(
            t=self.t,
            season_phase=self.season_phase,
            n_living=self.n_living,
            wild_remaining=self.wild_remaining.copy(),
            crop_stage=self.crop_stage.copy(),
            crop_health=self.crop_health.copy(),
            structure_type=self.structure_type.copy(),
            stored_food=self.stored_food.copy(),
            event_mask=self.event_mask.copy(),
            entities={k: EntityViz(v.alive, v.y, v.x, v.is_predator) for k, v in self.entities.items()},
        )
