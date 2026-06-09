"""
Reward-overhaul tests — the innate-drive backbone (survival + homeostasis + deferred
inclusive fitness) and the annealed intrinsic-curiosity scaffolding.

Covers:
  1. Deferred inclusive fitness is paid to a parent the step its child reaches viability.
  2. Identity guard: a recycled parent slot (same index, new occupant) is NOT paid.
  3. reward_schedule anneals beta_intr -> 0 and ent_coef -> floor, monotonically, in bounds.
  4. Count-based novelty: visit counts accumulate and the per-agent bonus decays on revisit.
"""
from __future__ import annotations

import numpy as np

from world import World
from world.config import WorldConfig
from sim.config import SimConfig
from sim.simulation import Simulation, Actions
from sim.actions import Action
from train.ppo import reward_schedule


def _world(width: int = 32, height: int = 32, seed: int = 42) -> World:
    return World.generate(WorldConfig(width=width, height=height, seed=seed))


def _noop(M: int) -> Actions:
    z = np.zeros(M, dtype=np.int32)
    return Actions(primary=np.full(M, int(Action.NOOP), dtype=np.int32), param=z.copy(), emit=z.copy())


def _calm_cfg(**kw) -> SimConfig:
    """A config where nothing dies or perturbs in a single step: no decay, no catastrophes."""
    defaults = dict(
        max_agents=16,
        init_agents=3,
        energy_decay=0.0,
        hydration_decay=0.0,
        thermal_decay=0.0,
        catastrophe_prob=0.0,
        fitness_viable_age=20,
    )
    defaults.update(kw)
    return SimConfig(**defaults)


def _full_drives(store, slots):
    for s in slots:
        store.energy[s] = 1.0
        store.hydration[s] = 1.0
        store.thermal[s] = 1.0
        store.health[s] = 1.0


# ---------------------------------------------------------------------------
# 1. Deferred inclusive fitness — happy path
# ---------------------------------------------------------------------------

def test_deferred_fitness_paid_at_child_viability():
    cfg = _calm_cfg()
    sim = Simulation(_world(), cfg)
    sim.reset(seed=0)                       # seeds founders in slots 0,1,2
    store = sim.store
    parent, child, control = 0, 1, 2
    _full_drives(store, (parent, child, control))

    # Park parent and control on the SAME tile so their comfort is identical; the only
    # reward difference is the deferred fitness payout the parent earns.
    store.y[control] = store.y[parent]
    store.x[control] = store.x[parent]

    # Wire child -> parent and put the child one step short of viability.
    store.parent_slot[child] = parent
    store.parent_birth_id[child] = store.birth_id[parent]
    store.age[child] = cfg.fitness_viable_age - 1   # step() increments age -> exactly viable
    # control has no children (parent_slot stays -1 from reset)

    out = sim.step(_noop(cfg.max_agents))

    bonus = out.reward[parent] - out.reward[control]
    assert np.isclose(bonus, cfg.w_r_surv, atol=1e-5), (
        f"parent should earn exactly w_r_surv over an identical childless control; got {bonus}"
    )


def test_deferred_fitness_doubles_for_two_children_same_step():
    cfg = _calm_cfg(init_agents=4)
    sim = Simulation(_world(), cfg)
    sim.reset(seed=0)
    store = sim.store
    parent, c1, c2, control = 0, 1, 2, 3
    _full_drives(store, (parent, c1, c2, control))
    store.y[control] = store.y[parent]
    store.x[control] = store.x[parent]

    for c in (c1, c2):
        store.parent_slot[c] = parent
        store.parent_birth_id[c] = store.birth_id[parent]
        store.age[c] = cfg.fitness_viable_age - 1

    out = sim.step(_noop(cfg.max_agents))
    bonus = out.reward[parent] - out.reward[control]
    assert np.isclose(bonus, 2 * cfg.w_r_surv, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Identity guard — recycled parent slot is not paid
# ---------------------------------------------------------------------------

def test_deferred_fitness_dropped_when_parent_slot_recycled():
    cfg = _calm_cfg()
    sim = Simulation(_world(), cfg)
    sim.reset(seed=0)
    store = sim.store
    recycled, child, control = 0, 1, 2
    _full_drives(store, (recycled, child, control))
    store.y[control] = store.y[recycled]
    store.x[control] = store.x[recycled]

    # Child references slot 0 with the parent's ORIGINAL birth_id...
    store.parent_slot[child] = recycled
    store.parent_birth_id[child] = store.birth_id[recycled]
    store.age[child] = cfg.fitness_viable_age - 1
    # ...but slot 0 has since been recycled to a DIFFERENT agent (new birth_id).
    store.birth_id[recycled] = store.birth_id[recycled] + 1000

    out = sim.step(_noop(cfg.max_agents))

    # The stranger now occupying slot 0 must NOT collect the original parent's reward.
    assert np.isclose(out.reward[recycled], out.reward[control], atol=1e-5), (
        "recycled slot collected a stranger's parental reward — identity guard failed"
    )


# ---------------------------------------------------------------------------
# 3. Annealing schedule
# ---------------------------------------------------------------------------

def test_reward_schedule_anneals_monotonically_and_in_bounds():
    cfg = SimConfig()
    prev = reward_schedule(0, cfg)
    assert np.isclose(prev["beta_intr"], cfg.intr_init)
    assert np.isclose(prev["ent_coef"], cfg.ent_init)

    for u in range(1, cfg.ent_anneal_updates + 50):
        cur = reward_schedule(u, cfg)
        # bounds
        assert 0.0 <= cur["beta_intr"] <= cfg.intr_init + 1e-9
        assert cfg.ent_floor - 1e-9 <= cur["ent_coef"] <= cfg.ent_init + 1e-9
        # monotone non-increasing
        assert cur["beta_intr"] <= prev["beta_intr"] + 1e-9
        assert cur["ent_coef"] <= prev["ent_coef"] + 1e-9
        prev = cur

    # fully annealed past the horizons
    assert reward_schedule(cfg.intr_anneal_updates, cfg)["beta_intr"] == 0.0
    assert np.isclose(reward_schedule(cfg.ent_anneal_updates, cfg)["ent_coef"], cfg.ent_floor)


# ---------------------------------------------------------------------------
# 4. Intrinsic novelty — accumulation + decay
# ---------------------------------------------------------------------------

def test_novelty_counts_accumulate_and_bonus_decays_on_revisit():
    cfg = _calm_cfg(intr_init=0.5)
    sim = Simulation(_world(), cfg)
    sim.reset(seed=0)
    sim._beta_intr = cfg.intr_init   # trainer normally sets this per update

    n_living = sim.store.n_living_agents()
    assert sim._novelty_counts.sum() == 0

    out1 = sim.step(_noop(cfg.max_agents))
    # one count increment per living agent this step
    assert sim._novelty_counts.sum() == n_living

    out2 = sim.step(_noop(cfg.max_agents))
    # NOOP + zero decay => agents sit in the same bucket; counts rise again...
    assert sim._novelty_counts.sum() == 2 * n_living
    # ...and the (constant comfort/survival) reward strictly drops because novelty decays.
    living = sim.store.living_agents_mask()
    assert out2.reward[living].mean() < out1.reward[living].mean()


def test_no_novelty_term_when_beta_zero():
    cfg = _calm_cfg()
    sim = Simulation(_world(), cfg)
    sim.reset(seed=0)
    assert sim._beta_intr == 0.0        # default outside training
    sim.step(_noop(cfg.max_agents))
    assert sim._novelty_counts.sum() == 0   # no counting work when the term is off
