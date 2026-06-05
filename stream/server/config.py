"""
Stream server configuration (environment variables with defaults).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class StreamConfig:
    world_dir: Path
    checkpoint_path: Path | None
    init_agents: int
    max_agents: int
    reset_seed: int
    no_respawn: bool
    steps_per_sec: float
    host: str
    port: int

    @classmethod
    def from_env(cls) -> "StreamConfig":
        root = _repo_root()
        world_dir = Path(os.environ.get("WORLD_DIR", root / "data" / "world_42"))
        ckpt_raw = os.environ.get("CHECKPOINT_PATH", "")
        checkpoint = Path(ckpt_raw) if ckpt_raw else None
        if checkpoint and not checkpoint.is_absolute():
            checkpoint = root / checkpoint

        return cls(
            world_dir=Path(world_dir),
            checkpoint_path=checkpoint if checkpoint and checkpoint.exists() else None,
            init_agents=int(os.environ.get("INIT_AGENTS", "96")),
            max_agents=int(os.environ.get("MAX_AGENTS", "192")),
            reset_seed=int(os.environ.get("RESET_SEED", "0")),
            no_respawn=os.environ.get("NO_RESPAWN", "").lower() in ("1", "true", "yes"),
            steps_per_sec=float(os.environ.get("STEPS_PER_SEC", "8")),
            host=os.environ.get("STREAM_HOST", "0.0.0.0"),
            port=int(os.environ.get("STREAM_PORT", "8000")),
        )


def load_palette() -> dict:
    path = Path(__file__).resolve().parents[1] / "shared" / "palette.json"
    return json.loads(path.read_text())
