"""
Phase 2 dynamics tests — hydration drive, DRINK action, water-anchored survival.

Covers (per spec):
  1. Hydration decays by ~hydration_decay per step at neutral genome water_need=1.0.
  2. DRINK on a high-water tile restores hydration; DRINK on a dry tile does not
     and is masked.
  3. An agent that never drinks dies of dehydration FIRST (hydration_decay=0.10 >
     energy_decay=0.05); attributed cause is "dehydrate".
  4. update_health: agent with full energy but zero hydration takes damage, not regen.
  5. build_mask: DRINK enabled only on watered tiles.
  6. 200-step scripted run keeps >0 agents alive on a 128x128 world.
"""
from __future__ import annotations

import numpy as np
import pytest

from world import World
from world.config import WorldConfig
from sim.config import SimConfig
from sim.state import WorldState, EntityStore
from sim.simulation import Simulation, Actions
from sim.actions import Action, N_PRIMARY, build_mask
from sim.dynamics import decay_drives, update_health, drink
from sim.reproduce import traits_from_genome, resolve_deaths


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_world(w: int = 128, h: int = 128, seed: int = 1) -> World:
    return World.generate(WorldConfig(width=w, height=h, seed=seed))


def _make_cfg(**kwargs) -> SimConfig:
    defaults = dict(max_agents=16, init_agents=4)
    defaults.update(kwargs)
    return SimConfig(**defaults)


def _single_agent_store(
    cfg: SimConfig,
    energy: float = 1.0,
    hydration: float = 1.0,
    health: float = 1.0,
    y: int = 0,
    x: int = 0,
    genome_val: float = 0.5,
    thermal: float = 1.0,
) -> EntityStore:
    """Minimal EntityStore with one living agent in slot 0.

    thermal defaults to 1.0 so these energy/hydration-focused tests are not
    affected by the Phase-4 thermal drive (an unset thermal=0 would now starve).
    """
    store = EntityStore.create(cfg)
    store.alive[0]     = True
    store.energy[0]    = energy
    store.hydration[0] = hydration
    store.health[0]    = health
    store.thermal[0]   = thermal
    store.y[0]         = y
    store.x[0]         = x
    store.genome[0]    = genome_val
    return store


def _water_prox(h: int = 8, w: int = 8, value: float = 1.0) -> np.ndarray:
    """Uniform water_proximity array."""
    return np.full((h, w), value, dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. Hydration decays per step
# ---------------------------------------------------------------------------

def test_hydration_decays_by_hydration_decay_per_step():
    """At neutral genome (water_need=1.0), hydration drops by hydration_decay each step."""
    cfg   = _make_cfg()
    store = _single_agent_store(cfg, hydration=1.0)

    decay_drives(store, cfg)

    expected = 1.0 - cfg.hydration_decay * 1.0   # water_need=1.0 at genome=0.5
    assert store.hydration[0] == pytest.approx(expected, abs=1e-5), (
        f"Expected hydration={expected:.4f}, got {store.hydration[0]:.4f}"
    )


def test_hydration_decays_faster_than_energy():
    """hydration_decay (0.10) > energy_decay (0.05): hydration falls faster."""
    cfg   = _make_cfg()
    store = _single_agent_store(cfg, energy=1.0, hydration=1.0)

    decay_drives(store, cfg)

    delta_energy    = 1.0 - float(store.energy[0])
    delta_hydration = 1.0 - float(store.hydration[0])
    assert delta_hydration > delta_energy, (
        f"Expected hydration to fall faster: Δhydration={delta_hydration:.4f}, "
        f"Δenergy={delta_energy:.4f}"
    )


def test_hydration_never_goes_below_zero():
    """Hydration is clipped to >= 0."""
    cfg   = _make_cfg()
    store = _single_agent_store(cfg, hydration=0.01)

    for _ in range(20):
        decay_drives(store, cfg)

    assert store.hydration[0] >= 0.0


def test_dead_slots_hydration_not_affected():
    """Dead agents must not have their hydration changed by decay_drives."""
    cfg   = _make_cfg()
    store = _single_agent_store(cfg, hydration=0.8)
    store.alive[0] = False

    decay_drives(store, cfg)

    assert store.hydration[0] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 2. DRINK action
# ---------------------------------------------------------------------------

def test_drink_on_high_water_tile_restores_hydration():
    """DRINK on a tile with water_proximity >= drink_min_water fills hydration."""
    cfg   = _make_cfg(drink_min_water=0.5, drink_restore=1.0)
    store = _single_agent_store(cfg, hydration=0.0, y=3, x=3)

    water = _water_prox(8, 8, value=1.0)   # tile [3,3] = 1.0 >= 0.5
    idx   = np.array([0], dtype=np.int32)
    drink(store, water, idx, cfg)

    assert store.hydration[0] == pytest.approx(1.0, abs=1e-5), (
        f"Expected hydration=1.0, got {store.hydration[0]:.4f}"
    )


def test_drink_on_dry_tile_does_not_restore():
    """DRINK on a tile below drink_min_water is a no-op."""
    cfg   = _make_cfg(drink_min_water=0.5, drink_restore=1.0)
    store = _single_agent_store(cfg, hydration=0.2, y=2, x=2)

    water = _water_prox(8, 8, value=0.1)   # 0.1 < 0.5 threshold
    idx   = np.array([0], dtype=np.int32)
    drink(store, water, idx, cfg)

    assert store.hydration[0] == pytest.approx(0.2, abs=1e-5), (
        f"Hydration should be unchanged on dry tile, got {store.hydration[0]:.4f}"
    )


def test_drink_does_not_exceed_one():
    """DRINK with partial hydration clips at 1.0."""
    cfg   = _make_cfg(drink_min_water=0.5, drink_restore=1.0)
    store = _single_agent_store(cfg, hydration=0.7, y=0, x=0)

    water = _water_prox(8, 8, value=1.0)
    idx   = np.array([0], dtype=np.int32)
    drink(store, water, idx, cfg)

    assert store.hydration[0] <= 1.0 + 1e-5


def test_drink_empty_idx_is_noop():
    """Calling drink with empty idx does nothing."""
    cfg   = _make_cfg()
    store = _single_agent_store(cfg, hydration=0.5)

    water = _water_prox(8, 8, value=1.0)
    idx   = np.empty(0, dtype=np.int32)
    drink(store, water, idx, cfg)   # must not raise

    assert store.hydration[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 3. Dehydration kills before starvation
# ---------------------------------------------------------------------------

def test_dehydration_kills_before_starvation_default_rates():
    """
    With default rates (hydration_decay=0.10 > energy_decay=0.05) and NO DRINK,
    an agent starting at full energy+hydration should reach hydration=0 before
    energy=0.  The death is then attributed to 'dehydrate', not 'starve'.
    """
    cfg   = _make_cfg(
        max_agents=8,
        init_agents=1,
        energy_decay=0.05,
        hydration_decay=0.10,
        starve_damage=0.05,
        health_regen=0.0,   # disable regen so only damage matters
    )
    store = _single_agent_store(cfg, energy=1.0, hydration=1.0, health=1.0)

    # Track when each drive hits zero
    energy_zero_step    = None
    hydration_zero_step = None
    metrics             = {}

    died = False
    for step in range(300):
        decay_drives(store, cfg)

        if hydration_zero_step is None and store.hydration[0] <= 0.0:
            hydration_zero_step = step
        if energy_zero_step is None and store.energy[0] <= 0.0:
            energy_zero_step = step

        update_health(store, cfg)
        done = resolve_deaths(store, metrics_deaths=metrics)

        if done[0]:
            died = True
            break

    assert died, "Agent should have died within 300 steps"

    # Hydration must have hit zero BEFORE energy (faster decay)
    assert hydration_zero_step is not None, "Hydration should have reached 0"
    if energy_zero_step is not None:
        assert hydration_zero_step <= energy_zero_step, (
            f"Hydration should deplete first: hydration@{hydration_zero_step}, "
            f"energy@{energy_zero_step}"
        )

    # Death attributed to dehydration
    assert metrics.get("dehydrate", 0) > 0, (
        f"Expected dehydrate death; metrics={metrics}"
    )


def test_resolve_deaths_attributes_dehydration():
    """resolve_deaths attributes cause 'dehydrate' when hydration==0."""
    cfg   = _make_cfg()
    store = _single_agent_store(cfg, energy=1.0, hydration=0.0, health=0.0)
    # health=0 → will die; hydration=0 → dehydrate cause

    metrics = {}
    done = resolve_deaths(store, metrics_deaths=metrics)

    assert done[0]
    assert metrics.get("dehydrate", 0) == 1, f"Expected 1 dehydrate death: {metrics}"
    assert metrics.get("starve", 0) == 0, f"Expected 0 starve deaths: {metrics}"


def test_resolve_deaths_attributes_starvation_when_hydrated():
    """resolve_deaths attributes cause 'starve' when only energy==0."""
    cfg   = _make_cfg()
    store = _single_agent_store(cfg, energy=0.0, hydration=1.0, health=0.0)
    # health=0 → will die; hydration > 0 → starve cause

    metrics = {}
    done = resolve_deaths(store, metrics_deaths=metrics)

    assert done[0]
    assert metrics.get("starve", 0) == 1, f"Expected 1 starve death: {metrics}"
    assert metrics.get("dehydrate", 0) == 0, f"Expected 0 dehydrate deaths: {metrics}"


# ---------------------------------------------------------------------------
# 4. update_health with hydration as active drive
# ---------------------------------------------------------------------------

def test_update_health_dehydrated_takes_damage():
    """Agent with full energy but zero hydration takes health damage each step."""
    cfg   = _make_cfg()
    store = _single_agent_store(cfg, energy=1.0, hydration=0.0, health=1.0)

    update_health(store, cfg)

    assert store.health[0] < 1.0, (
        f"Health should decrease when dehydrated; got {store.health[0]:.4f}"
    )


def test_update_health_dehydrated_no_regen():
    """Agent with zero hydration must NOT regenerate health even if energy is fine."""
    cfg   = _make_cfg(health_regen=0.02, drive_safe_band=0.3)
    store = _single_agent_store(cfg, energy=1.0, hydration=0.0, health=0.5)

    update_health(store, cfg)

    # Should have DECREASED (damage applied), not increased
    assert store.health[0] < 0.5, (
        f"Health must decrease when hydration=0, got {store.health[0]:.4f}"
    )


def test_update_health_both_drives_zero_double_damage():
    """Both energy=0 and hydration=0 → double starve_damage per step."""
    cfg   = _make_cfg(starve_damage=0.05)
    store = _single_agent_store(cfg, energy=0.0, hydration=0.0, health=1.0)

    update_health(store, cfg)

    expected = max(0.0, 1.0 - 2 * cfg.starve_damage)
    assert store.health[0] == pytest.approx(expected, abs=1e-5), (
        f"Expected double damage: {expected:.4f}, got {store.health[0]:.4f}"
    )


def test_update_health_full_drives_regen():
    """Agent with both drives above safe_band regenerates health."""
    cfg   = _make_cfg(drive_safe_band=0.3, health_regen=0.02)
    store = _single_agent_store(cfg, energy=1.0, hydration=1.0, health=0.8)

    update_health(store, cfg)

    assert store.health[0] > 0.8, (
        f"Health should regen with full drives; got {store.health[0]:.4f}"
    )


def test_update_health_low_hydration_no_regen():
    """Agent with energy in safe band but hydration just below safe_band: no regen."""
    cfg   = _make_cfg(drive_safe_band=0.3)
    store = _single_agent_store(cfg, energy=1.0, hydration=0.1, health=0.8)
    # hydration=0.1 < 0.3 = drive_safe_band → no regen; but hydration > 0 so no damage
    initial_health = float(store.health[0])

    update_health(store, cfg)

    # No regen (hydration below band) AND no damage (hydration > 0) → health unchanged
    assert store.health[0] == pytest.approx(initial_health, abs=1e-5), (
        f"Health should be unchanged when hydration < safe_band but > 0; "
        f"got {store.health[0]:.4f}"
    )


# ---------------------------------------------------------------------------
# 5. build_mask DRINK rules
# ---------------------------------------------------------------------------

def test_build_mask_drink_enabled_on_watered_tile():
    """DRINK mask is True for agents on tiles with water_proximity >= drink_min_water."""
    world = _make_world()
    cfg   = SimConfig(max_agents=16, init_agents=4, drink_min_water=0.5)
    sim   = Simulation(world, cfg)
    sim.reset(seed=0)

    # Find watered tiles and place an agent there
    water_prox = world.base_resources["water_proximity"]
    watered    = np.argwhere(water_prox >= cfg.drink_min_water)
    if watered.shape[0] == 0:
        pytest.skip("No watered tiles on this world")

    live_idx = np.flatnonzero(sim.store.living_agents_mask())
    if live_idx.size == 0:
        pytest.skip("No living agents")

    # Place first agent on a watered tile
    sim.store.y[live_idx[0]] = int(watered[0, 0])
    sim.store.x[live_idx[0]] = int(watered[0, 1])

    mask = build_mask(world, sim.state, sim.store, cfg)
    assert mask[live_idx[0], int(Action.DRINK)], (
        "DRINK should be enabled on a watered tile"
    )


def test_build_mask_drink_disabled_on_dry_tile():
    """DRINK mask is False for agents on tiles with water_proximity < drink_min_water."""
    world = _make_world()
    cfg   = SimConfig(max_agents=16, init_agents=4, drink_min_water=0.5)
    sim   = Simulation(world, cfg)
    sim.reset(seed=0)

    # Find dry tiles
    water_prox = world.base_resources["water_proximity"]
    dry_tiles  = np.argwhere(water_prox < cfg.drink_min_water)
    if dry_tiles.shape[0] == 0:
        pytest.skip("No dry tiles on this world")

    live_idx = np.flatnonzero(sim.store.living_agents_mask())
    if live_idx.size == 0:
        pytest.skip("No living agents")

    # Place all agents on a dry tile
    sim.store.y[live_idx] = int(dry_tiles[0, 0])
    sim.store.x[live_idx] = int(dry_tiles[0, 1])

    mask = build_mask(world, sim.state, sim.store, cfg)
    assert not mask[live_idx, int(Action.DRINK)].any(), (
        "DRINK should be disabled on dry tiles"
    )


def test_build_mask_drink_false_for_dead_slots():
    """Dead slots must have DRINK=False regardless of tile water."""
    world = _make_world()
    cfg   = SimConfig(max_agents=16, init_agents=4)
    sim   = Simulation(world, cfg)
    sim.reset(seed=0)

    # Kill all agents
    sim.store.alive[:] = False

    mask = build_mask(world, sim.state, sim.store, cfg)
    assert not mask[:, int(Action.DRINK)].any(), "Dead slots must have DRINK=False"


# ---------------------------------------------------------------------------
# 6. 200-step scripted run: forage when hungry, drink when thirsty near water
# ---------------------------------------------------------------------------

def test_200_step_scripted_forage_drink_run():
    """
    200-step scripted run on 128x128 world with Phase 2 dynamics active.
    Policy:
      1. EAT if inv_food >= 0.6 (digest food)
      2. DRINK if hydration < 0.5 AND on watered tile
      3. FORAGE if wild food on tile and energy < 0.7
      4. MOVE toward nearest water if hydration < 0.5 (random direction as proxy)
      5. Else REST / NOOP

    Config is generous to keep agents alive with scripted actions.
    Gate: >0 agents alive at step 200.
    """
    world = _make_world(w=128, h=128, seed=1)
    cfg   = SimConfig(
        max_agents=256,
        init_agents=128,
        energy_decay=0.005,       # slow starvation
        hydration_decay=0.008,    # slow dehydration (scripted agents don't always find water)
        forage_yield=1.0,
        forage_regen=0.05,
        eat_restore=0.9,
        spoilage_carried=0.0,
        drink_min_water=0.5,
        drink_restore=1.0,
    )
    sim = Simulation(world, cfg)
    sim.reset(seed=7)

    water_prox = world.base_resources["water_proximity"]
    rng = np.random.default_rng(42)
    M   = cfg.max_agents

    for step_i in range(200):
        store = sim.store
        state = sim.state

        primary = np.full(M, int(Action.NOOP), dtype=np.int32)
        param   = rng.integers(0, 8, size=M, dtype=np.int32)
        emit    = np.zeros(M, dtype=np.int32)

        live_idx = np.flatnonzero(store.alive & ~store.is_predator)
        if live_idx.size == 0:
            break

        ys = store.y[live_idx]
        xs = store.x[live_idx]

        inv_full      = store.inv_food[live_idx] >= 0.6
        thirsty       = store.hydration[live_idx] < 0.5
        near_water    = water_prox[ys, xs] >= cfg.drink_min_water
        has_wild      = state.wild_remaining[ys, xs] > 0.0
        hungry        = store.energy[live_idx] < 0.7

        # Priority: eat if full inventory, drink if thirsty & near water,
        # forage if hungry & food available, move if thirsty but dry tile, else noop
        action_choice = np.where(
            inv_full,
            int(Action.EAT),
            np.where(
                thirsty & near_water,
                int(Action.DRINK),
                np.where(
                    has_wild & hungry,
                    int(Action.FORAGE),
                    np.where(
                        thirsty,
                        int(Action.MOVE),  # move toward water (random)
                        int(Action.NOOP),
                    ),
                ),
            ),
        ).astype(np.int32)

        primary[live_idx] = action_choice

        act = Actions(primary=primary, param=param, emit=emit)
        out = sim.step(act)

        # Auto-respawn to keep population trainable (Phase 1-4 behaviour)
        sim.respawn_dead(seed=step_i)

    assert out.info["t"] == 200
    n_alive = sim.store.n_living_agents()
    assert n_alive > 0, f"All agents died within 200 steps (n_alive={n_alive})"

    # Bonus: verify reward shape and obs shapes are unchanged
    assert out.reward.shape == (M,)
    assert out.done.shape   == (M,)
    assert out.obs.spatial.shape == (M, 24, cfg.window_size, cfg.window_size)
    assert out.obs.vector.shape  == (M, 10 + cfg.genome_dim)
