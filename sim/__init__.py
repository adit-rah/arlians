"""
Arlians simulation layer.

A mutable, time-evolving simulation built on top of the read-only `world` package.
`world.World` provides the static geographic substrate + seasonal scalars; this package
adds mutable tile state, entities (agents + predators), a step loop, drives, threats,
reproduction, and a PettingZoo-style observation/action interface for multi-agent RL.

See the design + build spec at planning-prompt-glowing-lake.md (Parts I-III).

FROZEN CONTRACTS (do not change without an explicit re-lock):
    config.py     - SimConfig (all tunables)
    state.py      - WorldState + EntityStore (mutable arrays / struct-of-arrays)
    actions.py    - Action enum + space layout
    observe.py    - observation tensor layout
    simulation.py - Simulation facade (reset/step/observe)
"""

from .config import SimConfig
from .state import WorldState, EntityStore
from .simulation import Simulation, Obs, Actions, StepOut

__all__ = [
    "SimConfig",
    "WorldState",
    "EntityStore",
    "Simulation",
    "Obs",
    "Actions",
    "StepOut",
]
