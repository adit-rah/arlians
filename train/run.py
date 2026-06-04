"""
Training entrypoint — runs PPO over the Arlians simulation.

Defaults are a small CPU/MPS smoke (verifies the stack composes + loss moves).
Scale it up with flags for a real GPU run (e.g. on Kaggle). Checkpoints every
--checkpoint-every updates and on exit, and --resume continues from the last one
(so a Kaggle/Colab disconnect doesn't lose progress).

Examples:
  # local smoke (CPU/MPS)
  python train/run.py

  # real run on a GPU, checkpointing to a working dir, resumable
  python train/run.py --width 256 --agents 4096 --init-agents 2000 \
      --updates 3000 --horizon 128 --checkpoint runs/ckpt.pt --resume

See docs/TRAINING_KAGGLE.md for the full Kaggle walkthrough.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from world.config import WorldConfig
from world.world import World
from sim.config import SimConfig
from sim.simulation import Simulation
from train.policy import ArlianPolicy, DEVICE
from train.ppo import train


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Arlians PPO.")
    # world / sim scale
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=None, help="defaults to --width")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--agents", type=int, default=512, help="max_agents (slot count)")
    ap.add_argument("--init-agents", type=int, default=256)
    # training
    ap.add_argument("--updates", type=int, default=50, help="number of PPO updates")
    ap.add_argument("--horizon", type=int, default=64, help="T: steps per rollout")
    ap.add_argument("--no-respawn", action="store_true",
                    help="disable respawn_dead (use at Phase 5+ so reproduction sustains pop)")
    # checkpointing / logging
    ap.add_argument("--checkpoint", type=str, default=None, help="checkpoint path (.pt)")
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--resume", action="store_true", help="resume from --checkpoint if present")
    ap.add_argument("--log", type=str, default=None, help="append per-update metrics as JSONL here")
    args = ap.parse_args()

    H = args.height or args.width
    print(f"[run] device={DEVICE}  world={args.width}x{H}  agents={args.agents}"
          f"  updates={args.updates}  T={args.horizon}")

    world = World.generate(WorldConfig(width=args.width, height=H, seed=args.seed))
    sim_cfg = SimConfig(max_agents=args.agents, init_agents=args.init_agents)
    sim = Simulation(world, sim_cfg)

    policy = ArlianPolicy(n_symbols=sim_cfg.n_symbols).to(DEVICE)
    print(f"[run] policy params: {sum(p.numel() for p in policy.parameters()):,}")

    log_file = open(args.log, "a") if args.log else None
    t0 = time.time()

    def log_fn(row):
        u = row["update"]
        el = time.time() - t0
        print(
            f"[train] {u + 1}/{args.updates} done  "
            f"reward={row['mean_reward']:.3f}  return={row['mean_return']:.2f}  "
            f"vloss={row['value_loss']:.3f}  ent={row['entropy']:.3f}  "
            f"living={row['n_living']:>5}  "
            f"({el:.0f}s, {(u + 1) / max(el, 1e-9):.2f} upd/s)",
            flush=True,
        )
        if log_file:
            log_file.write(json.dumps(row) + "\n"); log_file.flush()

    print(f"[run] training... (Ctrl-C safe if --checkpoint set; --resume to continue)")
    train(
        sim, policy,
        n_updates=args.updates, T=args.horizon,
        use_respawn=not args.no_respawn,
        checkpoint_path=args.checkpoint,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        log_fn=log_fn,
    )
    if log_file:
        log_file.close()
    print(f"[run] done in {time.time() - t0:.0f}s"
          + (f"  (checkpoint: {args.checkpoint})" if args.checkpoint else ""))


if __name__ == "__main__":
    main()
