"""
Phase 7 CODE gate — catastrophe tracking, specialization index, and
signal<->action mutual information (build-spec §4.1, §5 Phase 7).

Scenarios
---------
1. Catastrophe tracking: catastrophe_steps > 0 when events are active;
   metric correctly tracks active-catastrophe fraction over the year.

2. Specialization index reflects role differentiation:
   (a) all-same-action population -> specialization_index ~ 0
   (b) split-role population      -> specialization_index > (a)

3. Signal<->action MI detects usage:
   (a) structured (signal k -> action k) population -> signal_action_mi > 0
   (b) random / independent signal population       -> signal_action_mi ~ 0
   Assert (a) > (b).

4. Metric integration: year_summary has catastrophe_steps, specialization_index,
   signal_action_mi (all finite) PLUS all prior Phase 1-6 keys; JSONL round-trips.

5. Visualization: render a frame during an active catastrophe to
   data/phase7_final.png.
"""
from __future__ import annotations

import json
import math
import os
import types
from typing import Optional

import numpy as np
import pytest

from world import World
from world.config import WorldConfig
from sim.config import SimConfig
from sim.state import EntityStore, WorldState
from sim.actions import Action, N_PRIMARY
from sim.simulation import Simulation, Actions
from sim.metrics import MetricsLogger, _compute_specialization_index, _compute_signal_action_mi


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_world(width: int = 48, height: int = 48, seed: int = 42) -> World:
    return World.generate(WorldConfig(width=width, height=height, seed=seed))


def _make_cfg(**kwargs) -> SimConfig:
    """Default survivable config with catastrophes enabled."""
    defaults = dict(
        max_agents=256,
        init_agents=64,
        energy_decay=0.002,
        hydration_decay=0.002,
        thermal_decay=0.0,      # no thermal stress — focus on catastrophe/signal
        D_init=0.0,
        spoilage_carried=0.0,
        forage_yield=0.6,
        forage_regen=0.04,
        pred_per_agents=0.0,    # disable predators to isolate catastrophe signal
        catastrophe_prob=0.5,   # very high probability for reliable triggering
        catastrophe_duration=10,
        catastrophe_radius=20,
        catastrophe_magnitude=0.5,
        n_symbols=8,
    )
    defaults.update(kwargs)
    return SimConfig(**defaults)


def _scripted_survive(sim: Simulation, cfg: SimConfig) -> Actions:
    """Simple survival policy: forage when possible, else eat, else rest."""
    from sim.actions import build_mask
    m = build_mask(sim.world, sim.state, sim.store, cfg)
    pr = np.full(cfg.max_agents, int(Action.REST), np.int32)
    for i in np.flatnonzero(sim.store.living_agents_mask()):
        e = sim.store.energy[i]
        f = sim.store.inv_food[i]
        h = sim.store.hydration[i]
        if e < 0.5 and f > 0:
            pr[i] = int(Action.EAT)
        elif m[i, int(Action.FORAGE)] and f < 0.5:
            pr[i] = int(Action.FORAGE)
        elif m[i, int(Action.DRINK)] and h < 0.6:
            pr[i] = int(Action.DRINK)
    param = np.zeros(cfg.max_agents, np.int32)
    emit  = np.zeros(cfg.max_agents, np.int32)
    return Actions(pr, param, emit)


def _make_stub_sim(cfg: SimConfig, world: World) -> object:
    """Build a minimal stub sim object that MetricsLogger.record_step can use,
    backed by a real WorldState + EntityStore so all field accesses work."""
    sim = types.SimpleNamespace()
    sim.world = world
    sim.state = WorldState.create(*world.elevation.shape)
    sim.store = EntityStore.create(cfg)
    # Initialise wild_remaining so land_mask checks work
    sim.state.wild_remaining[:] = world.base_resources["wild_food"]
    return sim


def _populate_store(store: EntityStore, n_agents: int, world: World, seed: int = 0) -> np.ndarray:
    """Spawn n_agents in the first n slots on random land tiles.  Returns slot indices."""
    land_ys, land_xs = np.where(world.elevation >= world.cfg.sea_level)
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0, land_ys.shape[0], size=n_agents)
    slots = np.arange(n_agents)
    store.alive[slots] = True
    store.is_predator[slots] = False
    store.y[slots] = land_ys[chosen]
    store.x[slots] = land_xs[chosen]
    store.energy[slots] = 1.0
    store.hydration[slots] = 1.0
    store.thermal[slots] = 1.0
    store.health[slots] = 1.0
    store.last_signal[slots] = 0
    return slots


# ---------------------------------------------------------------------------
# Test 1: Catastrophe tracking
# ---------------------------------------------------------------------------

def test_catastrophe_steps_tracked():
    """
    Run a simulation with very high catastrophe_prob (guaranteed near-daily events).
    After a year, catastrophe_steps must be > 0.0 and <= 1.0.
    """
    world = _make_world(width=48, height=48, seed=11)
    cfg = _make_cfg(catastrophe_prob=0.8, catastrophe_duration=30, max_agents=256, init_agents=64)
    sim = Simulation(world, cfg)
    sim.reset(seed=0)
    log = MetricsLogger()

    seen_cat = 0
    STEPS = 360
    for step in range(STEPS):
        acts = _scripted_survive(sim, cfg)
        out = sim.step(acts)
        sim.respawn_dead(seed=step)
        log.record_step(sim, out.info, actions=acts)
        if out.info.get("catastrophe_active", False):
            seen_cat += 1

    s = log.year_summary()

    assert "catastrophe_steps" in s, "year_summary missing 'catastrophe_steps'"
    cs = s["catastrophe_steps"]
    assert math.isfinite(cs), f"catastrophe_steps not finite: {cs}"
    assert 0.0 <= cs <= 1.0, f"catastrophe_steps out of [0,1]: {cs}"
    # With catastrophe_prob=0.8 and duration=30, we should see events most of the year
    assert cs > 0.0, (
        f"catastrophe_steps=0 but {seen_cat} catastrophe steps were observed; "
        f"check info dict wiring"
    )


def test_catastrophe_steps_zero_without_info():
    """
    When record_step is called WITHOUT info (backward-compat path), catastrophe_steps
    should be 0.0 (no data to detect catastrophes from).
    """
    world = _make_world(width=48, height=48, seed=12)
    cfg = _make_cfg(catastrophe_prob=1.0)
    sim = Simulation(world, cfg)
    sim.reset(seed=0)
    log = MetricsLogger()

    for step in range(20):
        acts = _scripted_survive(sim, cfg)
        sim.step(acts)
        log.record_step(sim)   # NO info passed
        sim.respawn_dead(seed=step)

    s = log.year_summary()
    # No info -> no catastrophe data -> should be 0.0
    assert s["catastrophe_steps"] == 0.0, (
        f"Expected catastrophe_steps=0.0 when no info provided, got {s['catastrophe_steps']}"
    )


def test_catastrophe_steps_matches_manual_count():
    """
    Cross-check: manually count steps with active catastrophe and compare to the
    fraction reported by MetricsLogger.
    """
    world = _make_world(width=48, height=48, seed=13)
    cfg = _make_cfg(catastrophe_prob=0.4, catastrophe_duration=20)
    sim = Simulation(world, cfg)
    sim.reset(seed=7)
    log = MetricsLogger()

    cat_count = 0
    STEPS = 180
    for step in range(STEPS):
        acts = _scripted_survive(sim, cfg)
        out = sim.step(acts)
        sim.respawn_dead(seed=step)
        log.record_step(sim, out.info, actions=acts)
        if out.info.get("catastrophe_active", False) or out.info.get("n_catastrophes", 0) > 0:
            cat_count += 1

    s = log.year_summary()
    expected_frac = cat_count / STEPS
    assert abs(s["catastrophe_steps"] - expected_frac) < 1e-9, (
        f"catastrophe_steps mismatch: logger={s['catastrophe_steps']:.6f}, "
        f"manual={expected_frac:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 2: Specialization index
# ---------------------------------------------------------------------------

def test_specialization_index_same_action_near_zero():
    """
    Population where every agent always takes the SAME action (FORAGE) every step.
    specialization_index should be 0.0 (or very close to 0).
    """
    N_AGENTS = 20
    N_STEPS = 50

    from sim.actions import N_PRIMARY

    # Build synthetic histograms: all agents do only FORAGE (action 2)
    hist = {}
    for i in range(N_AGENTS):
        h = np.zeros(N_PRIMARY, dtype=np.int64)
        h[int(Action.FORAGE)] = N_STEPS   # all FORAGE
        hist[i] = h

    idx = _compute_specialization_index(hist)
    assert idx == pytest.approx(0.0, abs=1e-8), (
        f"All-same-action population should have specialization_index ~ 0, got {idx}"
    )


def test_specialization_index_differentiated_is_higher():
    """
    Population split into two groups:
      - Group A (slots 0-9): always FORAGE (action 2)
      - Group B (slots 10-19): always REST (action 12)

    specialization_index should be strictly > 0 (and > the all-same-action case).
    """
    N_STEPS = 50

    from sim.actions import N_PRIMARY

    hist = {}
    # Group A: all FORAGE
    for i in range(10):
        h = np.zeros(N_PRIMARY, dtype=np.int64)
        h[int(Action.FORAGE)] = N_STEPS
        hist[i] = h
    # Group B: all REST
    for i in range(10, 20):
        h = np.zeros(N_PRIMARY, dtype=np.int64)
        h[int(Action.REST)] = N_STEPS
        hist[i] = h

    idx = _compute_specialization_index(hist)
    assert idx > 0.0, (
        f"Differentiated-role population should have specialization_index > 0, got {idx}"
    )


def test_specialization_index_differentiated_greater_than_same():
    """
    Assert (differentiated specialization) > (same-action specialization).
    """
    from sim.actions import N_PRIMARY

    N_STEPS = 50

    # All-same histograms
    hist_same = {}
    for i in range(20):
        h = np.zeros(N_PRIMARY, dtype=np.int64)
        h[int(Action.FORAGE)] = N_STEPS
        hist_same[i] = h

    # Differentiated histograms: 10 foragers + 10 resters
    hist_diff = {}
    for i in range(10):
        h = np.zeros(N_PRIMARY, dtype=np.int64)
        h[int(Action.FORAGE)] = N_STEPS
        hist_diff[i] = h
    for i in range(10, 20):
        h = np.zeros(N_PRIMARY, dtype=np.int64)
        h[int(Action.REST)] = N_STEPS
        hist_diff[i] = h

    idx_same = _compute_specialization_index(hist_same)
    idx_diff = _compute_specialization_index(hist_diff)

    assert idx_diff > idx_same, (
        f"Differentiated index ({idx_diff:.6f}) must exceed "
        f"same-action index ({idx_same:.6f})"
    )


def test_specialization_index_via_record_step():
    """
    End-to-end: feed MetricsLogger scripted actions with two role groups and
    verify specialization_index in year_summary is > 0.
    """
    world = _make_world(width=48, height=48, seed=99)
    cfg = _make_cfg(
        max_agents=128, init_agents=20,
        catastrophe_prob=0.0,
        pred_per_agents=0.0,
        energy_decay=0.0,       # prevent starvation so agents live through 50 steps
        hydration_decay=0.0,
    )

    stub = _make_stub_sim(cfg, world)
    slots = _populate_store(stub.store, 20, world, seed=0)

    # Group A (slots 0-9): primary=FORAGE; Group B (slots 10-19): primary=REST
    N_STEPS = 60
    log = MetricsLogger()
    primary = np.zeros(cfg.max_agents, dtype=np.int32)

    for step in range(N_STEPS):
        primary[:] = int(Action.REST)       # default
        primary[slots[:10]] = int(Action.FORAGE)  # Group A
        primary[slots[10:]] = int(Action.REST)    # Group B

        acts = Actions(
            primary=primary.copy(),
            param=np.zeros(cfg.max_agents, np.int32),
            emit=np.zeros(cfg.max_agents, np.int32),
        )
        log.record_step(stub, info=None, actions=acts)

    s = log.year_summary()
    assert s["specialization_index"] > 0.0, (
        f"Expected specialization_index > 0 for role-split population, "
        f"got {s['specialization_index']}"
    )


# ---------------------------------------------------------------------------
# Test 3: Signal<->action mutual information
# ---------------------------------------------------------------------------

def test_signal_action_mi_structured_detection():
    """
    Structured population: agents with signal k always take action k
    (using k in range [0, min(N_PRIMARY, n_symbols))).

    signal_action_mi should be > 0.
    """
    n_sig = 4
    n_agents_per_class = 10
    N_STEPS = 40

    pairs = []
    for step in range(N_STEPS):
        step_pairs = []
        for k in range(n_sig):
            for _ in range(n_agents_per_class):
                step_pairs.append([k, k])   # signal k -> action k (deterministic)
        pairs.append(np.array(step_pairs, dtype=np.int32))

    mi = _compute_signal_action_mi(pairs)
    assert mi > 0.0, (
        f"Structured (signal=k -> action=k) should have MI > 0, got {mi:.6f}"
    )


def test_signal_action_mi_random_near_zero():
    """
    Random population: signal is sampled uniformly independently of action.
    Expected MI ~ 0 (may be slightly above due to finite-sample bias, but
    should be much lower than the structured case).
    """
    n_symbols = 4
    n_actions = N_PRIMARY
    N_AGENTS = 100
    N_STEPS = 200
    rng = np.random.default_rng(42)

    pairs = []
    for step in range(N_STEPS):
        signals  = rng.integers(0, n_symbols, size=N_AGENTS)
        actions_ = rng.integers(0, n_actions, size=N_AGENTS)
        pairs.append(np.stack([signals, actions_], axis=1).astype(np.int32))

    mi_random = _compute_signal_action_mi(pairs)
    # Random MI should be close to 0; allow small positive bias from finite samples
    assert mi_random < 0.5, (
        f"Random-signal MI should be < 0.5 bits, got {mi_random:.4f}"
    )


def test_signal_action_mi_structured_greater_than_random():
    """
    Assert structured MI > random MI.
    """
    n_sig = 4
    N_STEPS = 60

    # Structured: signal k -> action k
    structured = []
    for step in range(N_STEPS):
        sp = []
        for k in range(n_sig):
            for _ in range(15):
                sp.append([k, k])
        structured.append(np.array(sp, dtype=np.int32))

    # Random: signal independent of action
    rng = np.random.default_rng(7)
    random_p = []
    n_per = n_sig * 15
    for step in range(N_STEPS):
        sigs = rng.integers(0, n_sig, size=n_per)
        acts = rng.integers(0, N_PRIMARY, size=n_per)
        random_p.append(np.stack([sigs, acts], axis=1).astype(np.int32))

    mi_struct  = _compute_signal_action_mi(structured)
    mi_random  = _compute_signal_action_mi(random_p)

    assert mi_struct > mi_random, (
        f"Structured MI ({mi_struct:.4f}) must be > random MI ({mi_random:.4f})"
    )


def test_signal_action_mi_via_record_step():
    """
    End-to-end: feed MetricsLogger two groups — one structured (signal -> action
    deterministically), one random.  The structured case must have higher MI.
    """
    world = _make_world(width=48, height=48, seed=55)
    cfg = _make_cfg(
        max_agents=128, init_agents=20,
        catastrophe_prob=0.0,
        pred_per_agents=0.0,
        energy_decay=0.0,
        hydration_decay=0.0,
        n_symbols=8,
    )

    N_STEPS = 80
    n_agents = 20

    # ---- Structured logger ----
    stub_s = _make_stub_sim(cfg, world)
    slots_s = _populate_store(stub_s.store, n_agents, world, seed=1)
    log_struct = MetricsLogger()

    rng = np.random.default_rng(0)
    for step in range(N_STEPS):
        # Assign signal k to each agent; signal determines action (k % N_PRIMARY)
        signals = np.zeros(cfg.max_agents, np.int32)
        primary = np.zeros(cfg.max_agents, np.int32)
        for i, slot in enumerate(slots_s):
            k = int(i % cfg.n_symbols)
            stub_s.store.last_signal[slot] = np.int8(k)
            signals[slot] = k
            primary[slot] = int(k % N_PRIMARY)

        acts = Actions(
            primary=primary.copy(),
            param=np.zeros(cfg.max_agents, np.int32),
            emit=signals.copy(),
        )
        log_struct.record_step(stub_s, info=None, actions=acts)

    s_struct = log_struct.year_summary()

    # ---- Random logger ----
    stub_r = _make_stub_sim(cfg, world)
    slots_r = _populate_store(stub_r.store, n_agents, world, seed=2)
    log_rand = MetricsLogger()

    for step in range(N_STEPS):
        signals = np.zeros(cfg.max_agents, np.int32)
        primary = np.zeros(cfg.max_agents, np.int32)
        sigs = rng.integers(0, cfg.n_symbols, size=n_agents)
        acts_ = rng.integers(0, N_PRIMARY, size=n_agents)
        for i, slot in enumerate(slots_r):
            stub_r.store.last_signal[slot] = np.int8(int(sigs[i]))
            signals[slot] = int(sigs[i])
            primary[slot] = int(acts_[i])

        acts = Actions(
            primary=primary.copy(),
            param=np.zeros(cfg.max_agents, np.int32),
            emit=signals.copy(),
        )
        log_rand.record_step(stub_r, info=None, actions=acts)

    s_rand = log_rand.year_summary()

    mi_struct = s_struct["signal_action_mi"]
    mi_random = s_rand["signal_action_mi"]

    assert mi_struct > mi_random, (
        f"Structured MI ({mi_struct:.4f}) must exceed random MI ({mi_random:.4f})"
    )
    assert mi_struct > 0.0, f"Structured MI should be > 0, got {mi_struct}"


# ---------------------------------------------------------------------------
# Test 4: Metric integration — all Phase 7 keys present, prior keys retained,
#          JSONL round-trips
# ---------------------------------------------------------------------------

# All Phase 1-6 keys that must still appear (additive contract check)
_PRIOR_KEYS = (
    # Phase 0/1
    "population_mean", "population_min", "population_max", "deaths_by_cause",
    "mean_displacement", "wild_food_mean",
    # Phase 2
    "water_occupancy",
    # Phase 3
    "pct_calories_farmed", "fertile_occupancy",
    # Phase 4
    "structures_built", "stored_food_total", "mean_thermal",
    # Phase 5
    "births", "mean_genome", "genome_drift", "lineage_count",
    # Phase 6
    "n_predators", "walls_built", "weapons_held",
    "predation_deaths", "conflict_deaths",
)


def test_phase7_metrics_integration(tmp_path):
    """
    Run a short sim with catastrophes and verify MetricsLogger surfaces all
    Phase 7 keys plus retains all prior Phase 1-6 keys.  JSONL round-trips.
    """
    world = _make_world(width=48, height=48, seed=7)
    cfg = _make_cfg(
        max_agents=256, init_agents=64,
        catastrophe_prob=0.3,
        catastrophe_duration=15,
        pred_per_agents=0.0,
    )
    sim = Simulation(world, cfg)
    sim.reset(seed=1)
    log = MetricsLogger(out_path=str(tmp_path / "phase7.jsonl"))

    STEPS = 360
    for step in range(STEPS):
        acts = _scripted_survive(sim, cfg)
        out = sim.step(acts)
        sim.respawn_dead(seed=step)
        log.record_step(sim, out.info, actions=acts)

    s = log.year_summary()

    # --- Phase 7 keys present and finite ---
    for key in ("catastrophe_steps", "specialization_index", "signal_action_mi"):
        assert key in s, f"year_summary missing Phase 7 key '{key}'"
        assert math.isfinite(s[key]), f"Phase 7 key '{key}' not finite: {s[key]}"
        assert s[key] >= 0.0, f"Phase 7 key '{key}' should be >= 0: {s[key]}"

    assert s["catastrophe_steps"] <= 1.0, (
        f"catastrophe_steps should be in [0,1]: {s['catastrophe_steps']}"
    )

    # --- All prior Phase 1-6 keys still present ---
    for k in _PRIOR_KEYS:
        assert k in s, f"Prior key missing from year_summary: '{k}'"

    # --- JSONL round-trip ---
    lines = (tmp_path / "phase7.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1, f"Expected 1 JSONL line, got {len(lines)}"
    parsed = json.loads(lines[0])

    for key in ("catastrophe_steps", "specialization_index", "signal_action_mi"):
        assert key in parsed, f"Phase 7 key '{key}' missing from JSONL"
        assert math.isfinite(parsed[key]), f"JSONL Phase 7 key '{key}' not finite"

    for k in _PRIOR_KEYS:
        assert k in parsed, f"Prior key missing from JSONL: '{k}'"


def test_phase7_metrics_zero_when_no_actions():
    """
    When record_step is called WITHOUT actions arg, specialization_index and
    signal_action_mi should both be 0.0 (no data).
    """
    world = _make_world(width=48, height=48, seed=8)
    cfg = _make_cfg(max_agents=64, init_agents=16, catastrophe_prob=0.0)
    sim = Simulation(world, cfg)
    sim.reset(seed=0)
    log = MetricsLogger()

    for step in range(20):
        acts = _scripted_survive(sim, cfg)
        out = sim.step(acts)
        log.record_step(sim, out.info)   # no actions arg
        sim.respawn_dead(seed=step)

    s = log.year_summary()
    assert s["specialization_index"] == 0.0, (
        f"specialization_index should be 0.0 without actions, "
        f"got {s['specialization_index']}"
    )
    assert s["signal_action_mi"] == 0.0, (
        f"signal_action_mi should be 0.0 without actions, "
        f"got {s['signal_action_mi']}"
    )


def test_accumulators_reset_phase7():
    """
    Verify Phase 7 accumulators reset after year_summary.  A second call with no
    record_step should return zeros for all three Phase 7 metrics.
    """
    world = _make_world(width=48, height=48, seed=3)
    cfg = _make_cfg(max_agents=64, init_agents=16, catastrophe_prob=0.5)
    sim = Simulation(world, cfg)
    sim.reset(seed=0)
    log = MetricsLogger()

    for step in range(20):
        acts = _scripted_survive(sim, cfg)
        out = sim.step(acts)
        sim.respawn_dead(seed=step)
        log.record_step(sim, out.info, acts)

    log.year_summary()   # consumes data

    # Second summary with no steps should report zeros
    s2 = log.year_summary()
    assert s2["catastrophe_steps"] == 0.0, f"Expected 0.0 after reset, got {s2['catastrophe_steps']}"
    assert s2["specialization_index"] == 0.0, f"Expected 0.0 after reset, got {s2['specialization_index']}"
    assert s2["signal_action_mi"] == 0.0, f"Expected 0.0 after reset, got {s2['signal_action_mi']}"


# ---------------------------------------------------------------------------
# Test 5: Visualization — render a frame during an active catastrophe
# ---------------------------------------------------------------------------

def test_render_phase7_catastrophe_png():
    """
    Render a frame while a catastrophe is active (event_mask tinted) to
    data/phase7_final.png.  Verifies imageio can write the file, file is
    non-empty, and the output is a valid (H, W, 3) uint8 RGB array.
    """
    import imageio.v3 as iio
    from sim.render import render_frame
    from sim.threats import roll_catastrophe, apply_catastrophes

    world = _make_world(width=48, height=48, seed=99)
    cfg = _make_cfg(
        max_agents=128, init_agents=32,
        catastrophe_prob=1.0,      # force a catastrophe on step 1
        catastrophe_duration=100,
        catastrophe_magnitude=0.3,
        pred_per_agents=0.0,
    )
    sim = Simulation(world, cfg)
    sim.reset(seed=0)

    # Run a few steps to let a catastrophe appear
    acts_noop = Actions(
        primary=np.zeros(cfg.max_agents, dtype=np.int32),
        param=np.zeros(cfg.max_agents, dtype=np.int32),
        emit=np.zeros(cfg.max_agents, dtype=np.int32),
    )
    for _ in range(5):
        sim.step(acts_noop)
        if len(getattr(sim, "_events", [])) > 0:
            break

    # Render the frame
    frame = render_frame(world, sim.state, sim.store)
    assert frame.shape == (world.elevation.shape[0], world.elevation.shape[1], 3), (
        f"Frame shape mismatch: {frame.shape}"
    )
    assert frame.dtype == np.uint8, f"Frame dtype should be uint8, got {frame.dtype}"

    # Write to persistent data/ directory
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase7_final.png")
    iio.imwrite(out_path, frame)

    assert os.path.exists(out_path), f"PNG not written to {out_path}"
    assert os.path.getsize(out_path) > 0, "PNG file is empty"
