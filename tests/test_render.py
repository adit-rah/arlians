"""
Phase-0 renderer tests.

Covers:
  - biome_basemap returns (H, W, 3) uint8.
  - render_frame returns (H, W, 3) uint8 and overlays differ from basemap.
  - render_frame does not crash with zero agents / crops / structures.
  - save_mp4 writes a non-empty file (mp4 or gif fallback).
"""
import os
import tempfile

import numpy as np
import pytest

from world import World
from world.config import WorldConfig
from sim.config import SimConfig
from sim.simulation import Simulation
from sim.render import biome_basemap, render_frame, save_mp4


# ---------------------------------------------------------------------------
# Shared fixture — small world so tests are fast
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_world():
    return World.generate(WorldConfig(width=96, height=96, seed=2))


@pytest.fixture(scope="module")
def small_sim(small_world):
    cfg = SimConfig(max_agents=256, init_agents=64)
    sim = Simulation(small_world, cfg)
    sim.reset(seed=0)
    return sim


# ---------------------------------------------------------------------------
# biome_basemap
# ---------------------------------------------------------------------------

def test_biome_basemap_shape_dtype(small_world):
    img = biome_basemap(small_world)
    H, W = small_world.biome_map.shape
    assert img.shape == (H, W, 3), f"Expected ({H},{W},3), got {img.shape}"
    assert img.dtype == np.uint8


def test_biome_basemap_values_in_range(small_world):
    img = biome_basemap(small_world)
    assert img.min() >= 0 and img.max() <= 255


# ---------------------------------------------------------------------------
# render_frame — basic shape and dtype
# ---------------------------------------------------------------------------

def test_render_frame_shape_dtype(small_world, small_sim):
    frame = render_frame(small_world, small_sim.state, small_sim.store)
    H, W = small_world.biome_map.shape
    assert frame.shape == (H, W, 3), f"Expected ({H},{W},3), got {frame.shape}"
    assert frame.dtype == np.uint8


# ---------------------------------------------------------------------------
# render_frame — overlays change pixels vs basemap
# ---------------------------------------------------------------------------

def test_render_frame_agents_change_pixels(small_world, small_sim):
    """After reset() there are living agents; their pixels should differ from basemap."""
    store = small_sim.store
    state = small_sim.state

    base  = biome_basemap(small_world)
    frame = render_frame(small_world, state, store)

    alive_mask = store.alive & ~store.is_predator
    assert alive_mask.any(), "Need living agents for this test"

    ys = store.y[alive_mask]
    xs = store.x[alive_mask]

    changed = np.any(frame[ys, xs] != base[ys, xs], axis=-1)
    assert changed.any(), "Agent overlay pixels must differ from basemap"


def test_render_frame_crops_change_pixels(small_world, small_sim):
    """Manually set a crop tile and check the overlay changes that pixel."""
    import copy

    # Clone state to avoid polluting the shared fixture
    from sim.state import WorldState
    state = small_sim.state

    # Pick a tile that isn't already ocean/edge
    y, x = 48, 48
    old_stage = float(state.crop_stage[y, x])
    state.crop_stage[y, x] = 0.5  # set crop

    base  = biome_basemap(small_world)
    frame = render_frame(small_world, state, small_sim.store)

    # Restore
    state.crop_stage[y, x] = old_stage

    assert not np.array_equal(frame[y, x], base[y, x]), (
        "Crop tile pixel must differ from basemap"
    )


def test_render_frame_structures_change_pixels(small_world, small_sim):
    """Manually set a structure tile and check the overlay changes that pixel."""
    state = small_sim.state

    y, x = 50, 50
    old_type = int(state.structure_type[y, x])
    state.structure_type[y, x] = 1  # shelter

    base  = biome_basemap(small_world)
    frame = render_frame(small_world, state, small_sim.store)

    state.structure_type[y, x] = old_type  # restore

    assert not np.array_equal(frame[y, x], base[y, x]), (
        "Structure tile pixel must differ from basemap"
    )


# ---------------------------------------------------------------------------
# render_frame — does not crash with zero entities
# ---------------------------------------------------------------------------

def test_render_frame_zero_agents(small_world):
    """render_frame must not crash when the store has no living entities."""
    from sim.state import WorldState, EntityStore

    state = WorldState.create(96, 96)
    cfg   = SimConfig(max_agents=64, init_agents=8)
    store = EntityStore.create(cfg)  # all alive=False

    frame = render_frame(small_world, state, store)
    assert frame.shape == (96, 96, 3)
    assert frame.dtype == np.uint8


# ---------------------------------------------------------------------------
# save_mp4 — writes a non-empty file (mp4 or gif fallback)
# ---------------------------------------------------------------------------

def test_save_mp4_writes_nonempty_file(small_world, small_sim):
    """save_mp4 must write a non-empty file; mp4 or gif fallback both acceptable."""
    state = small_sim.state
    store = small_sim.store

    frames = [render_frame(small_world, state, store) for _ in range(5)]

    with tempfile.TemporaryDirectory() as tmpdir:
        mp4_path = os.path.join(tmpdir, "test_out.mp4")
        actual_path = save_mp4(frames, mp4_path, fps=15)

        assert os.path.exists(actual_path), f"Output file not found: {actual_path}"
        assert os.path.getsize(actual_path) > 0, "Output file is empty"
        # Must be either .mp4 or .gif
        assert actual_path.endswith(".mp4") or actual_path.endswith(".gif"), (
            f"Unexpected extension: {actual_path}"
        )
