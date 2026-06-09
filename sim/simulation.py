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

        # Reward overhaul: intrinsic-curiosity weight, set per-update by the trainer's
        # anneal schedule and 0 outside training (so a bare step() carries no novelty term);
        # plus the global count-based novelty visit table that backs that bonus.
        self._beta_intr: float = 0.0
        self._novelty_counts: np.ndarray = self._new_novelty_counts()

    def _new_novelty_counts(self) -> np.ndarray:
        """Fresh zeroed visit-count table for the count-based novelty bonus.
        Buckets: drive_bins^3 (E,H,T) × food(3) × tile-water(2) × tile-wild(2)."""
        b = self.cfg.novelty_bins_drive
        self._novelty_shape = (b, b, b, 3, 2, 2)
        return np.zeros(self._novelty_shape, dtype=np.int64)

    def _novelty_bucket(self, idx: np.ndarray) -> tuple:
        """Discretize each agent's drive/resource state into novelty-table coordinates."""
        store = self.store
        b = self.cfg.novelty_bins_drive
        clipb = lambda v: np.clip((v * b).astype(np.int64), 0, b - 1)  # noqa: E731
        e = clipb(store.energy[idx])
        h = clipb(store.hydration[idx])
        th = clipb(store.thermal[idx])
        food = np.clip((store.inv_food[idx] * 2.0).astype(np.int64), 0, 2)
        ys, xs = store.y[idx], store.x[idx]
        water_prox = self.world.base_resources["water_proximity"]
        has_water = (water_prox[ys, xs] >= self.cfg.drink_min_water).astype(np.int64)
        has_wild = (self.state.wild_remaining[ys, xs] > 0.0).astype(np.int64)
        return (e, h, th, food, has_water, has_wild)

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
        # Founders are not children: parent_slot/parent_birth_id stay -1 (fresh from create).
        # Stamp each with a unique birth_id (1..n) so later reuse keeps ids monotonic.
        self.store.birth_id[slots]    = np.arange(1, n + 1, dtype=np.int64)

        # Fresh cohort => fresh curiosity map (also covers the survival-sim reset-floor reseed).
        self._novelty_counts = self._new_novelty_counts()

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
        # Respawned agents are fresh founders, not children (parent_slot/parent_birth_id were
        # cleared to -1 on death). Stamp a unique monotonic birth_id so the identity guard
        # rejects any stale parent reference to this recycled slot.
        base = int(self.store.birth_id.max()) + 1
        self.store.birth_id[slots]    = base + np.arange(slots.size, dtype=np.int64)
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
            resolve_combat, craft_weapon,
        )
        from .reproduce import resolve_deaths, reproduce as _reproduce
        from .threats import update_scent, spawn_predators, step_predators, roll_catastrophe, apply_catastrophes
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
        # 2b. ATTACK — Phase 6: deterministic HP combat using PRE-STEP snapshot.
        #     resolve_combat takes a snapshot internally before applying damage,
        #     so mutual attacks both land this step (§2.1 ordering).
        # ------------------------------------------------------------------
        attackers = np.flatnonzero(living & (actions.primary == int(Action.ATTACK)))
        combat_result = resolve_combat(store, state, attackers, actions.param, cfg)
        conflict_slots_this_step = combat_result["conflict_slots"]

        # ------------------------------------------------------------------
        # 2c. CRAFT — Phase 6: weapon crafting (unarmed + afford check)
        # ------------------------------------------------------------------
        crafters = np.flatnonzero(
            living
            & (actions.primary == int(Action.CRAFT))
            & (~store.weapon)
        )
        weapons_crafted = craft_weapon(store, crafters, cfg)

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
        planters = np.flatnonzero(
            living
            & (actions.primary == int(Action.PLANT))
            & (store.energy >= cfg.plant_energy_threshold)
        )
        plant_rng = np.random.default_rng(self.t + 3_000_000)
        planted_total = plant(store, state, planters, cfg, world, self.t, plant_rng)

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
        # _reproduce processes repro_candidates in ascending order, allocating one
        # child per parent until entity-slot capacity is exhausted
        # (reproduce.py:207-209), so the first `births_this_step` candidates are
        # exactly the successful parents — used for the reproduction reward (step 7).
        reproduced_slots = repro_candidates[:births_this_step]

        # REST (Action.REST) is already a no-op — nothing to do.

        # ------------------------------------------------------------------
        # 4e. Signaling (Phase 7): record the emit symbol for each living agent
        #     so that it is visible to neighbours via obs channel 23 next step.
        #     Clip to [0, n_symbols) to guard against out-of-range values.
        # ------------------------------------------------------------------
        if living.any():
            live_emit_idx = np.flatnonzero(living)
            clipped = np.clip(
                actions.emit[live_emit_idx].astype(np.int32),
                0,
                cfg.n_symbols - 1,
            ).astype(np.int8)
            store.last_signal[live_emit_idx] = clipped

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
        # 5b. Predator phase (Phase 6 §2.7 / build-spec §2.1 step 7):
        #   a. update_scent: deposit fresh scent at each living agent's tile.
        #   b. spawn_predators: maintain target count = pred_per_agents * n_living.
        #   c. step_predators: move + attack; returns slots that took predator damage.
        # Use a deterministic per-step RNG seeded from self.t (independent from
        # the reproduction RNG so they don't interfere).
        # ------------------------------------------------------------------
        pred_rng = np.random.default_rng(self.t + 1_000_000)
        update_scent(state, store)
        target_pred = round(cfg.pred_per_agents * store.n_living_agents())
        spawn_predators(store, world, target_pred, pred_rng)
        pred_result = step_predators(store, state, world, cfg, pred_rng)
        predator_slots_this_step = pred_result["predator_slots"]
        n_predators_now = int((store.alive & store.is_predator).sum())

        # ------------------------------------------------------------------
        # 5c. Catastrophe phase (Phase 7 §2.7):
        #   a. Lazily initialise the active-events list on the sim instance.
        #   b. roll_catastrophe: with prob catastrophe_prob, start a new event.
        #   c. apply_catastrophes: rebuild event_mask, apply per-type effects,
        #      drop expired events.
        # Uses a deterministic per-step RNG seeded from self.t (offset 7_000_000).
        # ------------------------------------------------------------------
        if not hasattr(self, "_events"):
            self._events: list = []
        cat_rng = np.random.default_rng(self.t + 7_000_000)
        self._events = roll_catastrophe(self._events, world, self.t, cfg, cat_rng)
        self._events = apply_catastrophes(state, store, world, self._events, self.t, cfg)
        n_catastrophes_now = len(self._events)

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
        deaths_by_cause: Dict[str, int] = {}
        done_mask = resolve_deaths(
            store,
            metrics_deaths=deaths_by_cause,
            conflict_slots=conflict_slots_this_step,
            predator_slots=predator_slots_this_step,
        )

        # ------------------------------------------------------------------
        # 7. Reward — innate-drive backbone (reward overhaul).
        #    Permanent instincts (never annealed):
        #      survival:    cfg.w_a per step alive. The PRIORITY — death ends the reward
        #                   stream, so over a long discounted horizon survival dominates.
        #      homeostasis: cfg.w_h * cbrt(E*H*T). Geometric mean = Liebig's law of the
        #                   minimum: comfort is capped by the WEAKEST drive, so a full
        #                   hydration meter can't paper over a starving energy level (kills
        #                   the over-drinking local optimum). A pure function of the drives.
        #      reproduction: a small innate urge (cfg.w_r_birth) per birth PLUS a dominant
        #                   DEFERRED inclusive-fitness payout (cfg.w_r_surv) paid to the
        #                   parent the step its child reaches viability age — so breeding
        #                   only pays when offspring actually survive (self-regulating
        #                   against breed-to-death; never fully suppressible via the urge).
        #    Scaffolding (annealed to 0 by the trainer via self._beta_intr):
        #      intrinsic novelty: count-based curiosity over drive/resource state, to
        #                   bootstrap past the food-economy wall.
        #    Strategies (farm/build/store/craft/defend/signal) are NEVER rewarded here —
        #    they must EMERGE as the instrumentally-optimal way to satisfy the instincts.
        # ------------------------------------------------------------------
        M = cfg.max_agents
        reward = np.zeros(M, dtype=np.float32)

        # Survival + homeostasis for every slot that was alive at the start of this step
        # (still alive OR just died). resolve_deaths already zeroed the just-died slots'
        # drives, so their comfort is 0 and they collect only the cfg.w_a survival base —
        # consistent with a dying agent getting its final-step reward before `done`.
        was_active = living  # living as of the start of this step (before deaths)
        reward_idx = np.flatnonzero(was_active | done_mask)
        if reward_idx.size > 0:
            comfort = np.cbrt(
                store.energy[reward_idx]
                * store.hydration[reward_idx]
                * store.thermal[reward_idx]
            )
            reward[reward_idx] = (cfg.w_h * comfort + cfg.w_a).astype(np.float32)

        # Innate reproductive urge: small per-birth bonus so PPO never fully suppresses
        # breeding. Kept small vs. comfort so survival stays the priority.
        if reproduced_slots.size > 0:
            reward[reproduced_slots] += np.float32(cfg.w_r_birth)

        # Deferred inclusive fitness: pay a parent cfg.w_r_surv the step its child reaches
        # viability age. The (parent_slot, parent_birth_id) identity guard pays only if that
        # slot STILL holds the same parent — slots get recycled. If the parent is dead or
        # its slot was reused, the credit is DROPPED (inclusive fitness already accrued to
        # the genome via selection: the child lived). This is the true, spam-proof fitness
        # signal that replaces the old flat per-birth bonus.
        living_now = store.living_agents_mask()
        viable = (
            living_now
            & (store.age == cfg.fitness_viable_age)
            & (store.parent_slot >= 0)
        )
        viable_idx = np.flatnonzero(viable)
        if viable_idx.size > 0:
            psl = store.parent_slot[viable_idx]
            same = store.alive[psl] & (store.birth_id[psl] == store.parent_birth_id[viable_idx])
            paid = psl[same]
            if paid.size > 0:
                # np.add.at so a parent with two children maturing the same step gets 2×.
                np.add.at(reward, paid, np.float32(cfg.w_r_surv))

        # Intrinsic curiosity (SCAFFOLDING; self._beta_intr is annealed to 0 by the trainer
        # and 0 outside training). Reward inverse-sqrt rarity of each agent's discretized
        # (drive-bin, food, tile-water, tile-wild) state, then bump its visit count. This
        # pushes agents to DISCOVER how to fix a low drive (eat when starving, drink when
        # thirsty) past the food-economy wall — without rewarding any specific strategy.
        if self._beta_intr > 0.0:
            alive_idx = np.flatnonzero(living_now)
            if alive_idx.size > 0:
                counts = self._novelty_counts.reshape(-1)
                flat = np.ravel_multi_index(self._novelty_bucket(alive_idx), self._novelty_shape)
                r_nov = 1.0 / np.sqrt(counts[flat].astype(np.float32) + 1.0)
                reward[alive_idx] += np.float32(self._beta_intr) * r_nov
                np.add.at(counts, flat, 1)  # np.add.at handles repeated buckets in one step

        # Slots dead before this step keep reward=0 (set at initialization above).

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
                "t":                  self.t,
                "n_agents":           self.store.n_living_agents(),
                "n_predators":        n_predators_now,
                "deaths":             n_deaths,
                "deaths_by_cause":    deaths_by_cause,
                "births":             births_this_step,
                "foraged":            foraged_total,
                "harvested":          harvested_total,
                "planted":            planted_total,
                "built":              built_total,
                "weapons_crafted":    weapons_crafted,
                "attacks":            int(attackers.size),
                "n_catastrophes":     n_catastrophes_now,
                "catastrophe_active": n_catastrophes_now > 0,
            },
        )

    def observe(self) -> Obs:
        return _observe.build_observation(self.world, self.state, self.store, self.t, self.cfg)

    @property
    def alive_mask(self) -> np.ndarray:
        return self.store.living_agents_mask()
