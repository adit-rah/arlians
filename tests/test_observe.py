"""
Phase-0 observation tests.

Covers:
  - Correctness pin: channels 0-13 of each living agent's spatial window match
    world.get_window() within atol=1e-5.
  - Shape contracts: spatial (M, 24, 15, 15) and vector (M, 16).
  - Dead slots are all-zero in both spatial and vector.
  - Alive rows are populated.
  - Smoke run: reset() + 360 step() calls on 128x128 with 512/256 agents.
  - Throughput microbench: time observe() and report agent-windows/sec.
"""
import time
import numpy as np
import pytest

from world import World
from world.config import WorldConfig
from sim.config import SimConfig
from sim.simulation import Simulation
from sim import observe as _observe


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_sim():
    """128x128 world, default sim config (window_radius=7 => 15x15 windows)."""
    world = World.generate(WorldConfig(width=128, height=128, seed=1))
    cfg   = SimConfig(max_agents=512, init_agents=128, window_radius=7, genome_dim=6)
    sim   = Simulation(world, cfg)
    sim.reset(seed=0)
    return sim


# ---------------------------------------------------------------------------
# 1. Shape tests
# ---------------------------------------------------------------------------

def test_spatial_shape(small_sim):
    obs = small_sim.observe()
    M   = small_sim.cfg.max_agents
    ws  = small_sim.cfg.window_size  # 15
    assert obs.spatial.shape == (M, _observe.N_SPATIAL, ws, ws), (
        f"Expected ({M}, 24, {ws}, {ws}), got {obs.spatial.shape}"
    )


def test_vector_shape(small_sim):
    obs  = small_sim.observe()
    M    = small_sim.cfg.max_agents
    vlen = _observe.vector_len(small_sim.cfg)  # 10 + 6 = 16
    assert obs.vector.shape == (M, vlen), (
        f"Expected ({M}, {vlen}), got {obs.vector.shape}"
    )


def test_n_spatial_is_24():
    assert _observe.N_SPATIAL == 24


def test_vector_len_is_16():
    cfg = SimConfig(genome_dim=6)
    assert _observe.vector_len(cfg) == 16


# ---------------------------------------------------------------------------
# 2. Dead slots are all-zero; alive rows populated
# ---------------------------------------------------------------------------

def test_dead_slots_are_zero(small_sim):
    obs        = small_sim.observe()
    alive_mask = obs.alive_mask
    dead_idx   = np.flatnonzero(~alive_mask)
    if dead_idx.size > 0:
        assert np.all(obs.spatial[dead_idx] == 0.0), "Dead spatial slots must be zero"
        assert np.all(obs.vector[dead_idx]  == 0.0), "Dead vector slots must be zero"


def test_alive_rows_are_populated(small_sim):
    obs        = small_sim.observe()
    alive_mask = obs.alive_mask
    live_idx   = np.flatnonzero(alive_mask)
    assert live_idx.size > 0, "Should have living agents after reset()"
    # At least the health channel (idx 3 in vector) should be 1.0 for fresh agents
    assert np.all(obs.vector[live_idx, 3] == pytest.approx(1.0, abs=1e-5)), (
        "Freshly reset agents must have health=1.0 in vector"
    )


def test_alive_mask_matches_store(small_sim):
    obs  = small_sim.observe()
    mask = small_sim.store.living_agents_mask()
    assert np.array_equal(obs.alive_mask, mask)


# ---------------------------------------------------------------------------
# 3. Correctness pin: channels 0-13 == world.get_window for ~100 random agents
# ---------------------------------------------------------------------------

def test_channels_0_13_match_world_get_window():
    """For 128x128 world, after reset(seed=0), spot-check up to 100 living agents."""
    world = World.generate(WorldConfig(width=128, height=128, seed=1))
    cfg   = SimConfig(max_agents=512, init_agents=256, window_radius=7, genome_dim=6)
    sim   = Simulation(world, cfg)
    sim.reset(seed=0)

    obs        = sim.observe()
    alive_mask = obs.alive_mask
    live_idx   = np.flatnonzero(alive_mask)

    rng      = np.random.default_rng(42)
    n_check  = min(100, live_idx.size)
    check_idx = rng.choice(live_idx, size=n_check, replace=False)

    r = cfg.window_radius
    mismatches = []
    for i in check_idx:
        y_i = int(sim.store.y[i])
        x_i = int(sim.store.x[i])
        ref_window = world.get_window(sim.t, y_i, x_i, r)  # (14, 15, 15)
        got_window = obs.spatial[i, 0:14]                   # (14, 15, 15)
        if not np.allclose(ref_window, got_window, atol=1e-5):
            max_diff = np.abs(ref_window - got_window).max()
            mismatches.append((i, y_i, x_i, max_diff))

    assert len(mismatches) == 0, (
        f"{len(mismatches)} agents failed channel 0-13 correctness check. "
        f"First failure: slot={mismatches[0][0]}, y={mismatches[0][1]}, "
        f"x={mismatches[0][2]}, max_diff={mismatches[0][3]:.2e}"
    )


# ---------------------------------------------------------------------------
# 4. Smoke run: reset() + 360 step() calls
# ---------------------------------------------------------------------------

def test_smoke_360_steps():
    """Should complete 360 steps without error at 512/256 on 128x128."""
    world = World.generate(WorldConfig(width=128, height=128, seed=7))
    cfg   = SimConfig(max_agents=512, init_agents=256, window_radius=7, genome_dim=6)
    sim   = Simulation(world, cfg)
    sim.reset(seed=99)

    from sim.simulation import Actions
    from sim.actions import N_PRIMARY, N_PARAM

    M = cfg.max_agents
    for step_i in range(360):
        actions = Actions(
            primary=np.zeros(M, dtype=np.int32),
            param=np.zeros(M, dtype=np.int32),
            emit=np.zeros(M, dtype=np.int32),
        )
        out = sim.step(actions)

    assert out.info["t"] == 360
    assert out.obs.spatial.shape == (M, 24, 15, 15)
    assert out.obs.vector.shape  == (M, 16)
    assert out.reward.shape == (M,)
    assert out.done.shape   == (M,)


# ---------------------------------------------------------------------------
# 5. Throughput microbench
# ---------------------------------------------------------------------------

def test_throughput_microbench():
    """Measure observe() throughput; print agent-windows/sec (no hard threshold)."""
    world = World.generate(WorldConfig(width=128, height=128, seed=3))
    cfg   = SimConfig(max_agents=512, init_agents=256, window_radius=7, genome_dim=6)
    sim   = Simulation(world, cfg)
    sim.reset(seed=5)

    # warm-up
    for _ in range(3):
        sim.observe()

    n_reps = 20
    t0 = time.perf_counter()
    for _ in range(n_reps):
        obs = sim.observe()
    elapsed = time.perf_counter() - t0

    n_alive  = int(obs.alive_mask.sum())
    windows_per_sec = (n_alive * n_reps) / elapsed
    calls_per_sec   = n_reps / elapsed

    print(
        f"\n[throughput] observe() over {n_reps} reps | "
        f"{calls_per_sec:.1f} calls/sec | "
        f"{windows_per_sec:.0f} agent-windows/sec | "
        f"{n_alive} living agents | "
        f"elapsed {elapsed*1000:.1f} ms total"
    )
    # Sanity: must complete without error
    assert obs.spatial is not None
