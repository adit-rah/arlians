"""Tests for stream delta computation."""
from __future__ import annotations

import numpy as np
import pytest

from stream.server.delta import compute_step_delta
from stream.server.snapshot import EntityViz, VizSnapshot


def _empty_snapshot(t: int = 0) -> VizSnapshot:
    H, W = 8, 8
    zf = lambda: np.zeros((H, W), dtype=np.float32)
    zi = lambda dt: np.zeros((H, W), dtype=dt)
    return VizSnapshot(
        t=t,
        season_phase=0.0,
        n_living=0,
        wild_remaining=zf(),
        crop_stage=zf(),
        crop_health=zf(),
        structure_type=zi(np.int8),
        stored_food=zf(),
        event_mask=np.ones((H, W), dtype=np.float32),
        entities={},
    )


def test_first_delta_sparse_tiles():
    curr = _empty_snapshot(0)
    curr.crop_stage[2, 3] = 0.5
    curr.entities[0] = EntityViz(True, 2, 3, False)
    curr.n_living = 1
    d = compute_step_delta(0, None, curr)
    assert d["seq"] == 0
    assert any(t["y"] == 2 and t["x"] == 3 for t in d["tiles"])
    assert len(d["entities"]) == 1
    assert d["entities"][0]["op"] == "upsert"


def test_tile_change_detected():
    prev = _empty_snapshot(0)
    curr = prev.copy_arrays()
    curr.t = 1
    curr.wild_remaining[1, 1] = 0.5
    d = compute_step_delta(1, prev, curr)
    assert len(d["tiles"]) == 1
    assert d["tiles"][0]["layers"]["wild"] == pytest.approx(0.5)


def test_entity_move_and_remove():
    prev = _empty_snapshot(0)
    prev.entities[1] = EntityViz(True, 0, 0, False)
    prev.n_living = 1
    curr = prev.copy_arrays()
    curr.t = 1
    curr.entities = {1: EntityViz(True, 1, 1, False)}
    d = compute_step_delta(1, prev, curr)
    upserts = [e for e in d["entities"] if e["op"] == "upsert"]
    assert len(upserts) == 1
    assert upserts[0]["y"] == 1 and upserts[0]["x"] == 1

    curr2 = curr.copy_arrays()
    curr2.t = 2
    curr2.entities = {}
    curr2.n_living = 0
    d2 = compute_step_delta(2, curr, curr2)
    assert any(e["op"] == "remove" and e["slot"] == 1 for e in d2["entities"])


def test_structure_place():
    prev = _empty_snapshot(0)
    curr = prev.copy_arrays()
    curr.structure_type[4, 4] = 1
    d = compute_step_delta(1, prev, curr)
    assert d["tiles"][0]["layers"]["structureType"] == 1
