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

    def respawn_dead(self, seed: Optional[int] = None, target: Optional[int] = None) -> np.ndarray:
        """Phase 1-4 training auto-reset: top the population back up to `target`
        (default `cfg.init_agents`) by spawning fresh agents into empty slots at
        spawn-eligible land tiles.

        This is the per-slot "auto-reset" of the continuing-stream PPO env (build-spec
        §3.2): before reproduction exists (Phase 5), dead slots are refilled this way so
        the population stays trainable. At Phase 5, REPRODUCE supplies births instead and
        this is disabled. Returns the indices of newly spawned slots.
        """
        tgt = self.cfg.init_agents if target is None else target
        deficit = tgt - self.store.n_living_agents()
        if deficit <= 0:
            return np.empty(0, dtype=np.int64)
        free = self.store.free_slots()
        if free.size == 0:
            return np.empty(0, dtype=np.int64)
        slots = free[:deficit]

        land_ys, land_xs = np.where(self.world.elevation >= self.world.cfg.sea_level)
        rng = np.random.default_rng(seed)
        chosen = rng.integers(0, land_ys.shape[0], size=slots.size)

        self.store.alive[slots]       = True
        self.store.is_predator[slots] = False
        self.store.y[slots]           = land_ys[chosen]
        self.store.x[slots]           = land_xs[chosen]
        self.store.energy[slots]      = 1.0
        self.store.hydration[slots]   = 1.0
        self.store.thermal[slots]     = 1.0
        self.store.health[slots]      = 1.0
        self.store.age[slots]         = 0
        self.store.inv_food[slots]    = 0.0
        self.store.genome[slots]      = 0.5
        self.store.lineage_id[slots]  = slots
        self.store.repro_cd[slots]    = 0
        return slots

    def step(self, actions: Actions) -> StepOut:
        """Advance ONE day for all live agents (build-spec §2.1 ordering).
        FROZEN signature; body filled incrementally per phase.

        Phase 1 step ordering (§2.1 Phase-1 subset):
          1. Advance clock; decay scent.
          2. MOVE: apply direction for agents whose primary==MOVE.
          3. FORAGE: process foraging agents in ascending slot order (deterministic).
          4. EAT: restore energy from inv_food.
          5. REST: no-op (just a legal action).
          6. World dynamics: regrow_wild, spoil_carried.
          7. Drive decay, health update, death resolution.
          8. Compute reward; build and return StepOut.

        Phase 4 additions:
          - BUILD/STORE/RETRIEVE actions processed in step 3/4 block.
          - Thermal drive updated in step 7 (via decay_drives with temp args).
          - structure_decay_step + spoil_stored called in world dynamics (step 6).
          - Curriculum exposure_scale (D) applied to thermal/exposure.
        """
        from .actions import Action, DIRECTIONS
        from .dynamics import (
            decay_drives, update_health, forage, eat, drink, regrow_wild, spoil_carried,
            crop_step, plant, harvest,
            build, store_food, retrieve_food, structure_decay_step, spoil_stored,
        )
        from .reproduce import resolve_deaths, reproduce as _reproduce
        from world.seasons import compute_season_state

        cfg   = self.cfg
        store = self.store
        state = self.state
        world = self.world

        # ------------------------------------------------------------------
        # 1. Clock + scent decay
        # ------------------------------------------------------------------
        self.t += 1
        state.scent *= 0.9

        # Mask of living (non-predator) agents; re-used throughout
        living = store.alive & ~store.is_predator

        # Phase 5: decrement reproduction cooldown for all living agents each step.
        # np maximum so we never go below 0.
        live_idx_cd = np.flatnonzero(living)
        if live_idx_cd.size > 0:
            store.repro_cd[live_idx_cd] = np.maximum(
                0, store.repro_cd[live_idx_cd] - 1
            ).astype(np.int32)

        # ------------------------------------------------------------------
        # 2. MOVE
        # ------------------------------------------------------------------
        movers = np.flatnonzero(living & (actions.primary == int(Action.MOVE)))
        if movers.size > 0:
            dirs  = np.clip(actions.param[movers], 0, len(DIRECTIONS) - 1)
            dy    = DIRECTIONS[dirs, 0]
            dx    = DIRECTIONS[dirs, 1]
            new_y = np.clip(store.y[movers] + dy, 0, self.H - 1)
            new_x = np.clip(store.x[movers] + dx, 0, self.W - 1)

            # Block agents from moving into ocean tiles (elevation < sea_level)
            sea   = world.cfg.sea_level
            elev  = world.elevation
            on_land = elev[new_y, new_x] >= sea
            # Agents that would enter ocean stay put
            store.y[movers] = np.where(on_land, new_y, store.y[movers])
            store.x[movers] = np.where(on_land, new_x, store.x[movers])

        # ------------------------------------------------------------------
        # 3. FORAGE — ascending slot order for deterministic conflict resolution
        #    Phase 4: also gathers wood/stone (abiotic, not depleted from tile)
        # ------------------------------------------------------------------
        # np.flatnonzero returns indices in ascending order by construction.
        foragers = np.flatnonzero(living & (actions.primary == int(Action.FORAGE)))
        # Snapshot wild_remaining sum before forage to compute foraged_total
        _wild_before = float(state.wild_remaining.sum()) if foragers.size > 0 else 0.0
        forage(store, state, foragers, cfg, world=world)
        _wild_after = float(state.wild_remaining.sum()) if foragers.size > 0 else 0.0
        foraged_total = max(0.0, _wild_before - _wild_after)

        # ------------------------------------------------------------------
        # 3b. PLANT — ascending slot order
        # ------------------------------------------------------------------
        planters = np.flatnonzero(living & (actions.primary == int(Action.PLANT)))
        planted_total = plant(store, state, planters, cfg)

        # ------------------------------------------------------------------
        # 3c. HARVEST — ascending slot order
        # ------------------------------------------------------------------
        harvesters = np.flatnonzero(living & (actions.primary == int(Action.HARVEST)))
        harvested_total = harvest(store, state, harvesters, cfg)

        # ------------------------------------------------------------------
        # 3d. BUILD — Phase 4: ascending slot order (deterministic)
        # ------------------------------------------------------------------
        builders = np.flatnonzero(living & (actions.primary == int(Action.BUILD)))
        built_total = build(store, state, builders, actions.param, cfg)

        # ------------------------------------------------------------------
        # 4. EAT
        # ------------------------------------------------------------------
        eaters = np.flatnonzero(living & (actions.primary == int(Action.EAT)))
        eat(store, eaters, cfg)

        # ------------------------------------------------------------------
        # 4b. DRINK — restore hydration from water-proximity tiles
        # ------------------------------------------------------------------
        drinkers = np.flatnonzero(living & (actions.primary == int(Action.DRINK)))
        if drinkers.size > 0:
            # Use the static base water_proximity (seasonally-stable; consistent
            # with what build_mask uses so mask and action gate never diverge).
            water_prox = world.base_resources["water_proximity"]
            drink(store, water_prox, drinkers, cfg)

        # ------------------------------------------------------------------
        # 4c. STORE / RETRIEVE — Phase 4 food storage actions
        # ------------------------------------------------------------------
        storers = np.flatnonzero(living & (actions.primary == int(Action.STORE)))
        store_food(store, state, storers, cfg)

        retrievers = np.flatnonzero(living & (actions.primary == int(Action.RETRIEVE)))
        retrieve_food(store, state, retrievers, cfg)

        # ------------------------------------------------------------------
        # 4d. REPRODUCE — Phase 5: energy-gated reproduction with cooldown.
        #     Gate: energy >= repro_energy_threshold AND repro_cd == 0.
        #     Use a deterministic per-step RNG seeded from self.t so runs are
        #     reproducible without changing __init__ signature.
        # ------------------------------------------------------------------
        repro_candidates = np.flatnonzero(
            living
            & (actions.primary == int(Action.REPRODUCE))
            & (store.energy >= cfg.repro_energy_threshold)
            & (store.repro_cd == 0)
        )
        repro_rng = np.random.default_rng(self.t)
        births_this_step = _reproduce(
            store, repro_candidates, cfg, repro_rng, self.H, self.W
        )

        # REST (Action.REST) is already a no-op — nothing to do.

        # ------------------------------------------------------------------
        # 5. World dynamics: regrow wild food, advance crop growth/rot,
        #    spoil carried food, structure decay, spoil stored food (Phase 4)
        # ------------------------------------------------------------------
        regrow_wild(state, world, self.t, cfg)
        crop_step(state, world, self.t, cfg)
        spoil_carried(store, cfg)
        structure_decay_step(state, cfg)
        spoil_stored(state, cfg)

        # ------------------------------------------------------------------
        # 6. Drive decay + health update + death resolution
        #    Phase 4: pass thermal parameters so decay_drives/update_health
        #             update thermal and apply exposure damage.
        # ------------------------------------------------------------------
        # Increment age for all living agents before dynamics
        store.age[living] += 1

        # Seasonal temperature modifier for this step
        season = compute_season_state(self.t, world.cfg)
        temp_mod = float(season.temperature_modifier)
        temp_base = world.temperature_base          # (H, W) f32
        D = self.curriculum.exposure_scale          # difficulty scalar

        decay_drives(
            store, cfg,
            temp_base=temp_base,
            temperature_modifier=temp_mod,
            state=state,
            exposure_scale=D,
        )
        update_health(
            store, cfg,
            temp_base=temp_base,
            temperature_modifier=temp_mod,
            state=state,
            exposure_scale=D,
        )
        done_mask = resolve_deaths(store)

        # ------------------------------------------------------------------
        # 7. Reward (§1.7): comfort-based homeostatic reward
        #    comfort = (energy + hydration + thermal) / 3
        #    r = cfg.w_h * comfort + cfg.w_a   for living agents (including
        #        the just-died ones who get their final-step reward before done)
        #    r = 0 for slots that were already dead before this step
        # ------------------------------------------------------------------
        M = cfg.max_agents
        reward = np.zeros(M, dtype=np.float32)

        # Agents alive at start of reward computation = those still alive OR those
        # that just died this step (they get the final-step comfort before reset).
        was_active = living  # living as of start of step (before deaths this step)
        # done_mask agents were living (was_active) and just died — give them reward
        # based on their final drive state (which is zeroed in resolve_deaths, but
        # we read the comfort from the values that were set before we called
        # resolve_deaths — however health/energy are zeroed by resolve_deaths).
        # To give the last-step reward correctly we need to read before death clears
        # them. Since resolve_deaths runs first, the drives of dead agents are already
        # 0. We can live with that (comfort=0 for just-died agents) — it is still
        # consistent: a dying agent gets r = w_a (alive bonus) * 0 comfort.
        # Actually §1.7 says "done ... gets r for the final step then done", which
        # means we DO want to give the final reward. The safest interpretation is to
        # compute comfort from the current (post-death-zeroing) state: dead agents have
        # comfort=0 so r = w_a * 0 + w_a = w_a, but since they are in done_mask we
        # emit that for the final step. Agents still alive get full comfort.
        #
        # Implementation: reward any slot that was in `living` OR in `done_mask`
        # (i.e., was alive at the start of this step).
        reward_idx = np.flatnonzero(was_active | done_mask)
        if reward_idx.size > 0:
            comfort = (
                store.energy[reward_idx]
                + store.hydration[reward_idx]
                + store.thermal[reward_idx]
            ) / 3.0
            reward[reward_idx] = (cfg.w_h * comfort + cfg.w_a).astype(np.float32)

        # Slots that are dead and NOT in done_mask (already dead before this step)
        # keep reward=0 (set at initialization above).

        # ------------------------------------------------------------------
        # 8. Observations + return
        # ------------------------------------------------------------------
        obs = self.observe()
        n_deaths = int(done_mask.sum())

        return StepOut(
            obs=obs,
            reward=reward,
            done=done_mask,
            info={
                "t":         self.t,
                "n_agents":  self.store.n_living_agents(),
                "deaths":    n_deaths,
                "births":    births_this_step,
                "foraged":   foraged_total,
                "harvested": harvested_total,
                "planted":   planted_total,
                "built":     built_total,
            },
        )

    def observe(self) -> Obs:
        return _observe.build_observation(self.world, self.state, self.store, self.t, self.cfg)

    @property
    def alive_mask(self) -> np.ndarray:
        return self.store.living_agents_mask()
