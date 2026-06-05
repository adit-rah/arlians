"""
Compute sparse tile/entity deltas between two VizSnapshots (protocol v1).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .snapshot import EntityViz, VizSnapshot

_PROTOCOL_VERSION = 1


def _tile_layers(
    prev: VizSnapshot | None,
    y: int,
    x: int,
    curr: VizSnapshot,
) -> dict[str, Any] | None:
    layers: dict[str, Any] = {}

    def changed(name: str, old_v: float, new_v: float, tol: float = 1e-5) -> bool:
        if prev is None:
            return True
        return abs(old_v - new_v) > tol

    wild = float(curr.wild_remaining[y, x])
    if prev is None or changed("wild", float(prev.wild_remaining[y, x]), wild):
        layers["wild"] = round(wild, 4)

    cs = float(curr.crop_stage[y, x])
    ch = float(curr.crop_health[y, x])
    if prev is None or changed("cropStage", float(prev.crop_stage[y, x]), cs):
        layers["cropStage"] = round(cs, 4)
    if prev is None or changed("cropHealth", float(prev.crop_health[y, x]), ch):
        layers["cropHealth"] = round(ch, 4)

    st = int(curr.structure_type[y, x])
    if prev is None or int(prev.structure_type[y, x]) != st:
        layers["structureType"] = st

    sf = float(curr.stored_food[y, x])
    if prev is None or changed("storedFood", float(prev.stored_food[y, x]), sf):
        layers["storedFood"] = round(sf, 4)

    em = float(curr.event_mask[y, x])
    if prev is None or changed("eventMask", float(prev.event_mask[y, x]), em, tol=1e-4):
        layers["eventMask"] = round(em, 4)

    # Drop keys that match "empty" defaults on first frame only — keep explicit zeros for clears
    if not layers:
        return None
    # If crop cleared, ensure client gets zeros
    if cs == 0.0 and "cropStage" not in layers and prev is not None:
        if float(prev.crop_stage[y, x]) != 0.0:
            layers["cropStage"] = 0.0
            layers["cropHealth"] = 0.0
    return layers


def _diff_tiles(prev: VizSnapshot | None, curr: VizSnapshot) -> list[dict[str, Any]]:
    H, W = curr.wild_remaining.shape
    changed_mask = np.zeros((H, W), dtype=bool)

    if prev is None:
        # First delta: only non-default tiles (avoid sending entire map)
        changed_mask |= curr.crop_stage > 0
        changed_mask |= curr.structure_type > 0
        changed_mask |= curr.stored_food > 0
        changed_mask |= curr.wild_remaining > 0.05
        changed_mask |= curr.event_mask < 0.999
    else:
        changed_mask |= np.abs(curr.wild_remaining - prev.wild_remaining) > 1e-5
        changed_mask |= curr.crop_stage != prev.crop_stage
        changed_mask |= curr.crop_health != prev.crop_health
        changed_mask |= curr.structure_type != prev.structure_type
        changed_mask |= np.abs(curr.stored_food - prev.stored_food) > 1e-5
        changed_mask |= np.abs(curr.event_mask - prev.event_mask) > 1e-4

    tiles: list[dict[str, Any]] = []
    ys, xs = np.where(changed_mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        layers = _tile_layers(prev, int(y), int(x), curr)
        if layers:
            tiles.append({"y": int(y), "x": int(x), "layers": layers})
    return tiles


def _diff_entities(prev: VizSnapshot | None, curr: VizSnapshot) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    prev_ent = prev.entities if prev is not None else {}
    curr_slots = set(curr.entities.keys())
    prev_slots = set(prev_ent.keys())

    for slot in sorted(prev_slots - curr_slots):
        out.append({"slot": slot, "op": "remove"})

    for slot in sorted(curr_slots):
        c = curr.entities[slot]
        p = prev_ent.get(slot)
        if p is None or p.y != c.y or p.x != c.x or p.is_predator != c.is_predator:
            out.append(
                {
                    "slot": slot,
                    "op": "upsert",
                    "y": c.y,
                    "x": c.x,
                    "isPredator": c.is_predator,
                }
            )
    return out


def compute_step_delta(
    seq: int,
    prev: VizSnapshot | None,
    curr: VizSnapshot,
) -> dict[str, Any]:
    """Build a JSON-serializable StepDelta dict."""
    return {
        "type": "StepDelta",
        "protocolVersion": _PROTOCOL_VERSION,
        "seq": seq,
        "t": curr.t,
        "seasonPhase": round(curr.season_phase, 4),
        "nLiving": curr.n_living,
        "tiles": _diff_tiles(prev, curr),
        "entities": _diff_entities(prev, curr),
    }
