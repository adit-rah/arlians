"""
SimConfig — every tunable for the Arlians simulation, frozen in one place.

Values are the starting defaults from build-spec Part II §1.1. They are tuned
empirically against the per-phase gates; the *field set and names* are a frozen
contract (subagents must not rename/remove fields). No magic numbers elsewhere in
the codebase — everything flows from here.

Grid dimensions and the season period are NOT stored here; they come from the
underlying `world.World` (`world.cfg.width/height/season_period`).
"""
from dataclasses import dataclass, field
from typing import Dict


def _cost(**kwargs: float) -> Dict[str, float]:
    """Resource cost map, e.g. _cost(wood=3, stone=1). Keys: wood/stone/minerals/food."""
    return dict(kwargs)


@dataclass(frozen=True)
class SimConfig:
    # ----- population / grid -----
    max_agents: int = 4096          # M: fixed entity-slot count (agents + predators); soft cap below
    window_radius: int = 7          # r: observation window is (2r+1)=15 on a side
    init_agents: int = 256          # agents seeded at spawn-eligible tiles on reset

    # ----- drives (per-step, at genome-neutral body) -----
    energy_decay: float = 0.05      # ~20 days to starve from full
    hydration_decay: float = 0.10   # ~10 days to dehydrate from full
    thermal_target_temp: float = 0.45   # effective temp at/above this -> thermal recovers
    thermal_decay: float = 0.08     # thermal lost/step when cold & unsheltered (×curriculum D)
    thermal_warm_regen: float = 0.05    # thermal gained/step when warm enough
    drive_safe_band: float = 0.30   # health regenerates only if all drives >= this

    # ----- health -----
    health_regen: float = 0.02      # /step inside safe band and not attacked
    starve_damage: float = 0.05     # /step per drive sitting at 0
    exposure_damage: float = 0.04   # /step cold & unsheltered (×curriculum D)

    # ----- drinking (phase 2) -----
    drink_min_water: float = 0.5    # min tile water_proximity required to DRINK
    drink_restore: float = 1.0      # hydration restored per DRINK on a watered tile

    # ----- food economy -----
    eat_restore: float = 0.6        # energy restored per unit food eaten
    forage_yield: float = 0.4       # food gained foraging a full wild tile
    forage_regen: float = 0.004     # wild_remaining regrow/step toward seasonal cap
    carry_capacity: float = 1.0     # base inventory food cap (×genome capacity)
    storage_capacity: float = 10.0  # food held per storage structure
    spoilage_carried: float = 0.01  # food lost/step in inventory
    spoilage_stored: float = 0.002  # food lost/step in a storage structure

    # ----- crops -----
    crop_base_growth: float = 0.02      # stage/step at fert=water=season=1 (~50d on prime tile)
    crop_yield: float = 1.2             # food at full maturity (must beat forage_yield)
    crop_min_fertility: float = 0.15    # eff_fert below this -> crop rots
    crop_rot: float = 0.05              # crop_health lost/step under rot conditions

    # ----- structures -----
    shelter_cost: Dict[str, float] = field(default_factory=lambda: _cost(wood=3))
    storage_cost: Dict[str, float] = field(default_factory=lambda: _cost(wood=4))
    wall_cost: Dict[str, float] = field(default_factory=lambda: _cost(stone=3, wood=1))
    structure_decay: float = 0.001      # hp/step
    shelter_exposure_mult: float = 0.1  # exposure damage multiplier for sheltered occupants
    shelter_temp_bonus: float = 0.3     # added to effective temperature when sheltered
    structure_init_hp: float = 1.0      # hp of a freshly built structure
    material_capacity: float = 10.0     # max inv_wood / inv_stone / inv_minerals each (×genome capacity)

    # ----- combat -----
    attack_damage: float = 0.15
    weapon_cost: Dict[str, float] = field(default_factory=lambda: _cost(wood=2, stone=1, minerals=1))
    weapon_attack_mult: float = 2.0     # armed attacker damage multiplier
    wall_defense_mult: float = 0.3      # damage taken when defending behind a wall
    group_defense_per_ally: float = 0.9 # damage ×0.9 per co-located ally...
    group_defense_floor: float = 0.3    # ...but never below this multiplier

    # ----- predators -----
    pred_per_agents: float = 0.05       # target predator count = this × living agents
    pred_sense_radius: int = 12
    pred_attack_damage: float = 0.2

    # ----- reproduction / genome -----
    repro_energy_threshold: float = 0.7 # min energy to reproduce
    repro_energy_cost: float = 0.4      # energy spent by parent
    repro_cooldown: int = 20            # steps before an agent can reproduce again
    genome_dim: int = 6                 # G: per-agent evolvable body-trait vector length
    mutation_sigma: float = 0.05        # gaussian mutation std (genes clipped to [0,1])

    # ----- signaling -----
    n_symbols: int = 8                  # emit() vocabulary size (3-bit channel)

    # ----- reward (evolution-only: purely individual homeostasis + survival) -----
    w_h: float = 1.0                    # weight on homeostatic comfort term
    w_a: float = 0.05                   # per-step alive bonus

    # ----- curriculum -----
    D_init: float = 0.3                 # initial difficulty scalar in [0,1]
    D_step: float = 0.1                 # ramp increment when population is stable
    stable_years_to_ramp: int = 3       # consecutive stable years before ramping D

    def __post_init__(self) -> None:
        assert self.max_agents > 0
        assert self.window_radius >= 1
        assert 0 < self.init_agents <= self.max_agents
        assert self.genome_dim >= 1
        assert self.n_symbols >= 1
        assert 0.0 <= self.D_init <= 1.0

    @property
    def window_size(self) -> int:
        """Side length of the observation window: 2r+1."""
        return 2 * self.window_radius + 1
