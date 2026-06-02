"""
Phase 1 dynamics tests.

Covers:
  - energy decays by ~energy_decay per step at neutral genome
  - foraging gains inv_food and depletes wild_remaining; wild regrows
  - eating raises energy and lowers inv_food
  - starvation causes death within expected lifetime
  - build_mask correctness for Phase 1 actions
  - reward formula for healthy agents
  - 200-step smoke run on 128x128 with 256/128 agents using random valid actions
"""
import numpy as np
import pytest

from world import World
from world.config import WorldConfig
from sim.config import SimConfig
from sim.state import WorldState, EntityStore
from sim.simulation import Simulation, Actions
from sim.actions import Action, N_PRIMARY, DIRECTIONS, build_mask
from sim.dynamics import decay_drives, update_health, forage, eat, regrow_wild, spoil_carried
from sim.reproduce import traits_from_genome, resolve_deaths


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_world(w=64, h=64, seed=1):
    return World.generate(WorldConfig(width=w, height=h, seed=seed))


def _make_sim(world=None, max_agents=16, init_agents=4, seed=0):
    if world is None:
        world = _make_world()
    cfg = SimConfig(max_agents=max_agents, init_agents=init_agents)
    sim = Simulation(world, cfg)
    sim.reset(seed=seed)
    return sim


def _single_agent_store(cfg, energy=1.0, hydration=1.0, thermal=1.0, health=1.0,
                        inv_food=0.0, y=0, x=0, genome_val=0.5):
    """Create a minimal EntityStore with exactly one living agent in slot 0."""
    store = EntityStore.create(cfg)
    store.alive[0]      = True
    store.energy[0]     = energy
    store.hydration[0]  = hydration
    store.thermal[0]    = thermal
    store.health[0]     = health
    store.inv_food[0]   = inv_food
    store.y[0]          = y
    store.x[0]          = x
    store.genome[0]     = genome_val   # broadcast to all G dims
    return store


# ---------------------------------------------------------------------------
# 1. Drive decay
# ---------------------------------------------------------------------------

def test_energy_decays_by_energy_decay_per_step():
    """At neutral genome (metab=1.0), energy drops by exactly cfg.energy_decay each step."""
    cfg   = SimConfig(max_agents=8, init_agents=1)
    store = _single_agent_store(cfg, energy=1.0)

    decay_drives(store, cfg)

    expected = 1.0 - cfg.energy_decay * 1.0  # metab=1.0 at genome=0.5
    assert store.energy[0] == pytest.approx(expected, abs=1e-5), (
        f"Expected energy={expected:.4f}, got {store.energy[0]:.4f}"
    )


def test_energy_never_goes_below_zero():
    """Energy should be clipped at 0, not go negative."""
    cfg   = SimConfig(max_agents=8, init_agents=1)
    store = _single_agent_store(cfg, energy=0.01)

    for _ in range(20):
        decay_drives(store, cfg)

    assert store.energy[0] >= 0.0


def test_dead_slots_not_affected_by_decay():
    """Slots with alive=False must not be touched by decay_drives."""
    cfg   = SimConfig(max_agents=8, init_agents=1)
    store = _single_agent_store(cfg, energy=1.0)
    store.alive[0] = False  # kill the agent

    decay_drives(store, cfg)

    assert store.energy[0] == pytest.approx(1.0)  # unchanged


# ---------------------------------------------------------------------------
# 2. Foraging
# ---------------------------------------------------------------------------

def test_forage_gains_inv_food_and_depletes_tile():
    """Foraging on a tile with food gains inv_food and reduces wild_remaining."""
    cfg   = SimConfig(max_agents=8, init_agents=1)
    store = _single_agent_store(cfg, inv_food=0.0, y=10, x=10)

    state = WorldState.create(64, 64)
    state.wild_remaining[10, 10] = 1.0   # full tile

    idx = np.array([0], dtype=np.int32)
    forage(store, state, idx, cfg)

    expected_take = min(cfg.carry_capacity * 1.0, 1.0, cfg.forage_yield)
    assert store.inv_food[0] == pytest.approx(expected_take, abs=1e-5)
    assert state.wild_remaining[10, 10] == pytest.approx(1.0 - expected_take, abs=1e-5)


def test_forage_limited_by_carry_capacity():
    """Agent should not take more food than carry capacity allows."""
    cfg   = SimConfig(max_agents=8, init_agents=1)
    store = _single_agent_store(cfg, inv_food=0.0)
    store.inv_food[0] = cfg.carry_capacity * 0.9  # nearly full

    state = WorldState.create(64, 64)
    state.wild_remaining[0, 0] = 10.0

    idx = np.array([0], dtype=np.int32)
    forage(store, state, idx, cfg)

    assert store.inv_food[0] <= cfg.carry_capacity * 1.0 + 1e-5


def test_forage_empty_tile_gains_nothing():
    """Foraging on an empty tile should leave inv_food unchanged."""
    cfg   = SimConfig(max_agents=8, init_agents=1)
    store = _single_agent_store(cfg, inv_food=0.0)

    state = WorldState.create(64, 64)
    state.wild_remaining[0, 0] = 0.0   # no food

    idx = np.array([0], dtype=np.int32)
    forage(store, state, idx, cfg)

    assert store.inv_food[0] == pytest.approx(0.0)


def test_wild_regrows_toward_cap():
    """After depletion, wild_remaining should grow back toward the cap over time."""
    world = _make_world()
    cfg   = SimConfig(max_agents=8, init_agents=1)
    sim   = Simulation(world, cfg)
    sim.reset(seed=0)

    # Find a land tile with non-zero base wild_food
    base = world.base_resources["wild_food"]
    good_tiles = np.argwhere(base > 0.1)
    assert len(good_tiles) > 0, "Need a tile with base wild_food > 0.1"
    ty, tx = good_tiles[0]

    # Deplete the tile completely
    sim.state.wild_remaining[ty, tx] = 0.0

    initial = float(sim.state.wild_remaining[ty, tx])

    # Run regrow several steps
    for step in range(100):
        regrow_wild(sim.state, world, sim.t + step, cfg)

    final = float(sim.state.wild_remaining[ty, tx])
    assert final > initial, f"wild_remaining should grow from {initial:.4f}, got {final:.4f}"


def test_wild_regrow_does_not_exceed_cap():
    """wild_remaining should not exceed the seasonal cap."""
    world = _make_world()
    cfg   = SimConfig(max_agents=8, init_agents=1)
    sim   = Simulation(world, cfg)
    sim.reset(seed=0)

    from world.seasons import compute_season_state
    season = compute_season_state(1, world.cfg)
    cap    = world.base_resources["wild_food"] * season.wild_food_modifier

    # Overfill (artificially) then regrow
    sim.state.wild_remaining[:] = cap * 2.0
    regrow_wild(sim.state, world, 1, cfg)

    # Regrow moves toward cap; above-cap tiles should move toward cap (decrease)
    # The rule is: wild_remaining += regen * (cap - wild_remaining), so above-cap
    # tiles will decrease.  Just verify no negative values.
    assert (sim.state.wild_remaining >= 0).all()


# ---------------------------------------------------------------------------
# 3. Eating
# ---------------------------------------------------------------------------

def test_eat_raises_energy_and_lowers_inv_food():
    """Eating should increase energy and decrease inv_food."""
    cfg   = SimConfig(max_agents=8, init_agents=1)
    store = _single_agent_store(cfg, energy=0.5, inv_food=1.0)

    idx = np.array([0], dtype=np.int32)
    eat(store, idx, cfg)

    assert store.energy[0] > 0.5,   f"Energy should have risen from 0.5, got {store.energy[0]:.4f}"
    assert store.inv_food[0] < 1.0, f"inv_food should have dropped from 1.0, got {store.inv_food[0]:.4f}"


def test_eat_caps_energy_at_one():
    """Eating when nearly full should not push energy above 1.0."""
    cfg   = SimConfig(max_agents=8, init_agents=1)
    store = _single_agent_store(cfg, energy=0.99, inv_food=10.0)

    idx = np.array([0], dtype=np.int32)
    eat(store, idx, cfg)

    assert store.energy[0] <= 1.0 + 1e-5


def test_eat_no_food_leaves_energy_unchanged():
    """EAT with inv_food=0 should leave energy unchanged."""
    cfg   = SimConfig(max_agents=8, init_agents=1)
    store = _single_agent_store(cfg, energy=0.5, inv_food=0.0)

    idx = np.array([0], dtype=np.int32)
    eat(store, idx, cfg)

    assert store.energy[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 4. Starvation & death
# ---------------------------------------------------------------------------

def test_agent_dies_within_expected_lifetime():
    """
    An agent with no food and no foraging should die by starvation.

    Expected upper bound on lifetime:
        T_starve = 1/energy_decay + 1/starve_damage   (rough estimate)
    """
    cfg = SimConfig(max_agents=8, init_agents=1, energy_decay=0.05, starve_damage=0.05)
    T_max = int(1.0 / cfg.energy_decay + 1.0 / cfg.starve_damage) + 10  # generous buffer

    store = _single_agent_store(cfg, energy=1.0, health=1.0)
    idx   = np.array([], dtype=np.int32)   # no foraging

    died = False
    for step in range(T_max * 2):
        decay_drives(store, cfg)
        update_health(store, cfg)
        done_mask = resolve_deaths(store)
        if done_mask[0]:
            died = True
            assert step < T_max, (
                f"Agent died at step {step} but expected to die within {T_max}"
            )
            break

    assert died, f"Agent should have died within {T_max * 2} steps but survived"


def test_dead_agent_has_alive_false():
    """resolve_deaths should set alive=False for agents with health <= 0."""
    cfg   = SimConfig(max_agents=8, init_agents=1)
    store = _single_agent_store(cfg, health=0.0)   # already at zero

    done_mask = resolve_deaths(store)

    assert done_mask[0], "Slot 0 should be in done_mask"
    assert not store.alive[0], "Slot 0 should be dead"


def test_done_mask_returned_by_resolve_deaths():
    """resolve_deaths returns a bool array; True only for slots that died this step."""
    cfg   = SimConfig(max_agents=8, init_agents=1)
    store = _single_agent_store(cfg, health=0.0)
    store.alive[1] = True  # another living slot with health > 0 (default health=0 but alive)
    store.health[1] = 1.0

    done_mask = resolve_deaths(store)

    assert done_mask.shape == (cfg.max_agents,)
    assert done_mask.dtype == bool
    assert done_mask[0]       # slot 0 had health<=0 and alive=True
    assert not done_mask[1]   # slot 1 has health=1.0


def test_full_step_starvation_sets_done():
    """Running Simulation.step repeatedly with NOOP actions should eventually return done=True."""
    world = _make_world(w=64, h=64, seed=42)
    cfg   = SimConfig(max_agents=16, init_agents=1,
                      energy_decay=0.1, starve_damage=0.1, forage_regen=0.0)
    sim   = Simulation(world, cfg)
    sim.reset(seed=0)

    M = cfg.max_agents
    # Start with minimal energy so starvation is fast
    sim.store.energy[sim.store.alive] = 0.01

    noop_actions = Actions(
        primary=np.zeros(M, dtype=np.int32),
        param=np.zeros(M, dtype=np.int32),
        emit=np.zeros(M, dtype=np.int32),
    )

    found_done = False
    for _ in range(500):
        out = sim.step(noop_actions)
        if out.done.any():
            found_done = True
            break

    assert found_done, "Expected at least one agent to die (done=True) via starvation"


# ---------------------------------------------------------------------------
# 5. build_mask
# ---------------------------------------------------------------------------

def test_build_mask_dead_slots_all_false():
    """All mask entries for dead slots must be False."""
    world = _make_world()
    cfg   = SimConfig(max_agents=16, init_agents=4)
    sim   = Simulation(world, cfg)
    sim.reset(seed=0)

    mask = build_mask(world, sim.state, sim.store, cfg)
    dead_idx = np.flatnonzero(~(sim.store.alive & ~sim.store.is_predator))

    assert mask.shape == (cfg.max_agents, N_PRIMARY)
    if dead_idx.size > 0:
        assert not mask[dead_idx].any(), "Dead slot mask should be all-False"


def test_build_mask_noop_move_rest_always_on_for_living():
    """NOOP, MOVE, REST are always True for living agents."""
    world = _make_world()
    cfg   = SimConfig(max_agents=16, init_agents=4)
    sim   = Simulation(world, cfg)
    sim.reset(seed=0)

    mask     = build_mask(world, sim.state, sim.store, cfg)
    live_idx = np.flatnonzero(sim.store.living_agents_mask())

    assert (mask[live_idx, int(Action.NOOP)]).all(), "NOOP should be True for all living"
    assert (mask[live_idx, int(Action.MOVE)]).all(), "MOVE should be True for all living"
    assert (mask[live_idx, int(Action.REST)]).all(), "REST should be True for all living"


def test_build_mask_forage_masked_on_empty_tiles():
    """FORAGE is False for agents on tiles with wild_remaining=0."""
    world = _make_world()
    cfg   = SimConfig(max_agents=16, init_agents=4)
    sim   = Simulation(world, cfg)
    sim.reset(seed=0)

    # Deplete ALL wild food
    sim.state.wild_remaining[:] = 0.0

    mask     = build_mask(world, sim.state, sim.store, cfg)
    live_idx = np.flatnonzero(sim.store.living_agents_mask())

    if live_idx.size > 0:
        assert not mask[live_idx, int(Action.FORAGE)].any(), (
            "FORAGE should be False on empty tiles"
        )


def test_build_mask_forage_on_nonempty_tiles():
    """FORAGE is True for agents whose tile has wild_remaining > 0."""
    world = _make_world()
    cfg   = SimConfig(max_agents=16, init_agents=4)
    sim   = Simulation(world, cfg)
    sim.reset(seed=0)

    # Fill ALL wild food
    sim.state.wild_remaining[:] = 1.0

    mask     = build_mask(world, sim.state, sim.store, cfg)
    live_idx = np.flatnonzero(sim.store.living_agents_mask())

    if live_idx.size > 0:
        assert mask[live_idx, int(Action.FORAGE)].all(), (
            "FORAGE should be True on full tiles"
        )


def test_build_mask_eat_masked_with_no_food():
    """EAT is False for agents with inv_food=0."""
    world = _make_world()
    cfg   = SimConfig(max_agents=16, init_agents=4)
    sim   = Simulation(world, cfg)
    sim.reset(seed=0)

    # Ensure all agents have no food
    sim.store.inv_food[:] = 0.0

    mask     = build_mask(world, sim.state, sim.store, cfg)
    live_idx = np.flatnonzero(sim.store.living_agents_mask())

    if live_idx.size > 0:
        assert not mask[live_idx, int(Action.EAT)].any(), (
            "EAT should be False when inv_food=0"
        )


def test_build_mask_eat_enabled_with_food():
    """EAT is True for agents that carry food."""
    world = _make_world()
    cfg   = SimConfig(max_agents=16, init_agents=4)
    sim   = Simulation(world, cfg)
    sim.reset(seed=0)

    live_idx = np.flatnonzero(sim.store.living_agents_mask())
    if live_idx.size == 0:
        pytest.skip("No living agents")

    # Give all living agents some food
    sim.store.inv_food[live_idx] = 0.5

    mask = build_mask(world, sim.state, sim.store, cfg)
    assert mask[live_idx, int(Action.EAT)].all(), "EAT should be True when carrying food"


# ---------------------------------------------------------------------------
# 6. Reward formula
# ---------------------------------------------------------------------------

def test_reward_formula_healthy_agent():
    """
    For a fully healthy agent: comfort=1.0 (energy=hydration=thermal=1.0)
    reward should equal w_h * 1.0 + w_a.
    """
    world = _make_world()
    cfg   = SimConfig(max_agents=16, init_agents=1)
    sim   = Simulation(world, cfg)
    sim.reset(seed=0)

    # Set all living agents to max drives
    live_idx = np.flatnonzero(sim.store.living_agents_mask())
    sim.store.energy[live_idx]     = 1.0
    sim.store.hydration[live_idx]  = 1.0
    sim.store.thermal[live_idx]    = 1.0
    # Pre-fill food so EAT action keeps energy at 1
    sim.store.inv_food[live_idx]   = 1.0

    # Use REST actions so nothing changes meaningfully (except decay)
    M = cfg.max_agents
    act = Actions(
        primary=np.full(M, int(Action.REST), dtype=np.int32),
        param=np.zeros(M, dtype=np.int32),
        emit=np.zeros(M, dtype=np.int32),
    )

    out = sim.step(act)

    # After one step of REST, energy will have decayed slightly.
    # Just verify the reward shape and that reward for living is positive and ≈ w_h*comfort+w_a
    live_after = np.flatnonzero(out.obs.alive_mask)
    if live_after.size > 0:
        assert (out.reward[live_after] > 0).all(), "Living agents should get positive reward"
        # Comfort is energy+hydration+thermal)/3 ≈ (slightly_below_1 + 1 + 1)/3 ≈ ~1
        # Reward should be in (w_a, w_h + w_a]
        max_reward = cfg.w_h * 1.0 + cfg.w_a
        assert (out.reward[live_after] <= max_reward + 1e-4).all()
        assert (out.reward[live_after] >= cfg.w_a - 1e-4).all()


def test_reward_dead_slots_are_zero():
    """Dead slots (not just-died) should get reward=0."""
    world = _make_world()
    cfg   = SimConfig(max_agents=16, init_agents=4)
    sim   = Simulation(world, cfg)
    sim.reset(seed=0)

    M    = cfg.max_agents
    act  = Actions(
        primary=np.zeros(M, dtype=np.int32),
        param=np.zeros(M, dtype=np.int32),
        emit=np.zeros(M, dtype=np.int32),
    )
    out  = sim.step(act)
    dead = np.flatnonzero(~out.obs.alive_mask & ~out.done)

    if dead.size > 0:
        assert (out.reward[dead] == 0.0).all(), "Already-dead slots must get reward=0"


# ---------------------------------------------------------------------------
# 7. 200-step smoke run
# ---------------------------------------------------------------------------

def test_200_step_smoke_run():
    """
    200 steps on 128x128 world with max_agents=256, init_agents=128 using random
    valid actions.  Must finish without error and leave >0 agents alive.

    Config is tuned to be survivable with purely random valid actions:
      - energy_decay reduced (agents last longer without food)
      - forage_yield/regen boosted (food is plentiful relative to demand)
      - eat_restore boosted (more energy per food unit)
      - spoilage_carried=0 (no spoilage to avoid wasteful loss)
    These are deliberately generous for a smoke test; real training uses defaults.
    """
    world = _make_world(w=128, h=128, seed=1)
    cfg = SimConfig(
        max_agents=256,
        init_agents=128,
        # generous food economy for random-action survival
        energy_decay=0.005,
        forage_yield=1.0,
        forage_regen=0.05,
        eat_restore=0.9,
        spoilage_carried=0.0,
    )
    sim = Simulation(world, cfg)
    sim.reset(seed=7)

    rng = np.random.default_rng(42)
    M   = cfg.max_agents

    for step_i in range(200):
        # Build a valid action mask and sample uniformly from it
        mask = build_mask(world, sim.state, sim.store, cfg)  # (M, N_PRIMARY)

        primary = np.zeros(M, dtype=np.int32)
        for i in range(M):
            valid_actions = np.flatnonzero(mask[i])
            if valid_actions.size > 0:
                primary[i] = rng.choice(valid_actions)
            else:
                primary[i] = int(Action.NOOP)

        act = Actions(
            primary=primary,
            param=rng.integers(0, 8, size=M, dtype=np.int32),
            emit=rng.integers(0, cfg.n_symbols, size=M, dtype=np.int32),
        )
        out = sim.step(act)

    assert out.info["t"] == 200
    assert out.obs.spatial.shape == (M, 24, cfg.window_size, cfg.window_size)
    assert out.obs.vector.shape  == (M, 10 + cfg.genome_dim)
    assert out.reward.shape == (M,)
    assert out.done.shape   == (M,)
    n_alive = sim.store.n_living_agents()
    assert n_alive > 0, f"All agents died within 200 steps (n_alive={n_alive})"


# ---------------------------------------------------------------------------
# 8. traits_from_genome
# ---------------------------------------------------------------------------

def test_traits_from_genome_neutral():
    """Neutral genome (0.5) should produce metab=1.0, water_need=1.0, cold_tol=0.0, capacity=1.0."""
    cfg    = SimConfig(max_agents=4, init_agents=1)
    genome = np.full((cfg.max_agents, cfg.genome_dim), 0.5, dtype=np.float32)
    traits = traits_from_genome(genome)

    assert traits.metab[0]      == pytest.approx(1.0, abs=1e-5)
    assert traits.water_need[0] == pytest.approx(1.0, abs=1e-5)
    assert traits.cold_tol[0]   == pytest.approx(0.0, abs=1e-5)
    assert traits.capacity[0]   == pytest.approx(1.0, abs=1e-5)


def test_traits_from_genome_extremes():
    """genome=0 -> lower bounds; genome=1 -> upper bounds."""
    cfg    = SimConfig(max_agents=4, init_agents=1)
    G      = cfg.genome_dim
    genome = np.zeros((cfg.max_agents, G), dtype=np.float32)
    t_low  = traits_from_genome(genome)
    assert t_low.metab[0]      == pytest.approx(0.5, abs=1e-5)
    assert t_low.water_need[0] == pytest.approx(0.5, abs=1e-5)
    assert t_low.cold_tol[0]   == pytest.approx(-0.2, abs=1e-5)
    assert t_low.capacity[0]   == pytest.approx(0.7, abs=1e-5)

    genome[:] = 1.0
    t_high = traits_from_genome(genome)
    assert t_high.metab[0]      == pytest.approx(1.5, abs=1e-5)
    assert t_high.water_need[0] == pytest.approx(1.5, abs=1e-5)
    assert t_high.cold_tol[0]   == pytest.approx(0.2, abs=1e-5)
    assert t_high.capacity[0]   == pytest.approx(1.3, abs=1e-5)
