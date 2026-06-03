"""
Entrypoint — small-scale PPO smoke run (build-spec §5 Phase 1 CODE-gate).

World: 128x128, seed=1
Sim:   max_agents=512, init_agents=256
Train: 50 updates, T=64

Full-scale behavioral gate is deferred to GPU. This run just verifies that
the entire stack (world gen -> sim -> policy -> PPO) composes without error
and that losses are finite and move.
"""
from __future__ import annotations

import sys
import os

# Make sure the repo root is on sys.path when run as a script
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
    print(f"[run] Device: {DEVICE}")

    # ---- world ----
    world_cfg = WorldConfig(width=128, height=128, seed=1)
    world = World.generate(world_cfg)

    # ---- sim ----
    sim_cfg = SimConfig(max_agents=512, init_agents=256)
    sim = Simulation(world, sim_cfg)

    # ---- policy ----
    policy = ArlianPolicy(n_symbols=sim_cfg.n_symbols)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[run] Policy parameters: {n_params:,}")

    # ---- train ----
    N_UPDATES = 50
    T = 64

    print(f"[run] Starting {N_UPDATES} PPO updates (T={T})...")
    print(f"{'Update':>6}  {'Reward':>8}  {'Return':>8}  {'PLoss':>8}  {'VLoss':>8}  {'Entropy':>8}  {'Living':>6}")
    print("-" * 70)

    history = train(sim, policy, n_updates=N_UPDATES, T=T)

    for row in history:
        u = row["update"]
        # Print every 5th update plus the first and last
        if u % 5 == 0 or u == N_UPDATES - 1:
            print(
                f"{u:>6}  "
                f"{row['mean_reward']:>8.4f}  "
                f"{row['mean_return']:>8.4f}  "
                f"{row['policy_loss']:>8.4f}  "
                f"{row['value_loss']:>8.4f}  "
                f"{row['entropy']:>8.4f}  "
                f"{row['n_living']:>6}"
            )

    print("\n[run] Done.")


if __name__ == "__main__":
    main()
