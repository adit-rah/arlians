"""
SimEngine — persistent simulation + policy stepping for the live stream.
"""
from __future__ import annotations

import threading
from typing import Any

import numpy as np
import torch

from sim import Simulation, SimConfig, Actions
from sim.actions import build_mask
from train.policy import ArlianPolicy, DEVICE
from world.seasons import compute_season_state

from .config import StreamConfig, load_palette
from .delta import compute_step_delta
from .snapshot import VizSnapshot
from .world_loader import bootstrap_payload, load_world_from_dir


class SimEngine:
    def __init__(self, cfg: StreamConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._paused = False
        self._seq = 0
        self._prev_snapshot: VizSnapshot | None = None
        self._latest_delta: dict[str, Any] | None = None
        self._palette = load_palette()

        self.world = load_world_from_dir(cfg.world_dir)
        sim_cfg = SimConfig(max_agents=cfg.max_agents, init_agents=cfg.init_agents)
        self.sim = Simulation(self.world, sim_cfg)
        self.sim_cfg = sim_cfg

        self.policy = ArlianPolicy(n_symbols=sim_cfg.n_symbols).to(DEVICE)
        if cfg.checkpoint_path:
            ckpt = torch.load(cfg.checkpoint_path, map_location=DEVICE)
            self.policy.load_state_dict(ckpt["policy"] if "policy" in ckpt else ckpt)
        self.policy.eval()

        self.reset(cfg.reset_seed)

    def reset(self, seed: int | None = None) -> None:
        with self._lock:
            s = self.cfg.reset_seed if seed is None else seed
            self.sim.reset(seed=s)
            self._seq = 0
            self._prev_snapshot = None
            season = compute_season_state(self.sim.t, self.world.cfg)
            snap = VizSnapshot.from_sim(self.sim, season.season_phase)
            self._latest_delta = compute_step_delta(0, None, snap)
            self._prev_snapshot = snap.copy_arrays()

    @property
    def bootstrap(self) -> dict:
        return bootstrap_payload(self.world, self._palette)

    def health(self) -> dict:
        with self._lock:
            snap = self._prev_snapshot
            return {
                "running": True,
                "paused": self._paused,
                "t": int(self.sim.t),
                "nLiving": snap.n_living if snap else 0,
                "seq": self._seq,
                "device": str(DEVICE),
                "checkpoint": str(self.cfg.checkpoint_path) if self.cfg.checkpoint_path else None,
            }

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    def step_once(self) -> dict[str, Any]:
        """Advance one sim step; return StepDelta dict."""
        with self._lock:
            if self._paused:
                return self._latest_delta or {}

            sim = self.sim
            cfg = self.sim_cfg
            world = self.world

            mask = build_mask(world, sim.state, sim.store, cfg)
            with torch.inference_mode():
                obs = sim.observe()
                act = self.policy.act(obs.spatial, obs.vector, mask)

            actions = Actions(
                primary=act["primary"].cpu().numpy().astype(np.int32),
                param=act["param"].cpu().numpy().astype(np.int32),
                emit=act["emit"].cpu().numpy().astype(np.int32),
            )
            sim.step(actions)
            if not self.cfg.no_respawn:
                sim.respawn_dead(seed=int(sim.t))

            season = compute_season_state(sim.t, world.cfg)
            curr = VizSnapshot.from_sim(sim, season.season_phase)
            self._seq += 1
            delta = compute_step_delta(self._seq, self._prev_snapshot, curr)
            self._prev_snapshot = curr.copy_arrays()
            self._latest_delta = delta
            return delta

    @property
    def latest_delta(self) -> dict[str, Any] | None:
        with self._lock:
            return self._latest_delta

    def get_delta_after(self, seq: int) -> dict[str, Any] | None:
        with self._lock:
            if self._latest_delta is None:
                return None
            if int(self._latest_delta.get("seq", 0)) > seq:
                return self._latest_delta
            return None
