"""
Difficulty curriculum controller (build-spec Part II §3.4).

Holds a single difficulty scalar D in [0,1] that scales the harshness of winter,
exposure damage, and the food economy. Ramps D up when the population has been
stable for `stable_years_to_ramp` years; holds or steps down on collapse risk.

Phase gates are MANUAL human checkpoints (orchestrator reports, founder approves);
this controller only modulates difficulty *within* a phase.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .config import SimConfig


@dataclass
class Curriculum:
    cfg: SimConfig
    D: float = field(init=False)
    _stable_years: int = field(default=0, init=False)
    _pop_history: List[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.D = self.cfg.D_init

    def on_year_end(self, population: int, collapsed: bool) -> None:
        """Call once per simulated year. Ramps difficulty when stable, backs off on collapse."""
        self._pop_history.append(population)
        if collapsed or population == 0:
            self._stable_years = 0
            self.D = max(0.0, self.D - self.cfg.D_step)
            return
        self._stable_years += 1
        if self._stable_years >= self.cfg.stable_years_to_ramp:
            self.D = min(1.0, self.D + self.cfg.D_step)
            self._stable_years = 0

    @property
    def exposure_scale(self) -> float:
        """Multiplier applied to exposure/thermal damage (build-spec §2.5)."""
        return self.D
