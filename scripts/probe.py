"""
Emergence probe — does the trained policy show survival skill and EMERGENT strategy?

The reward only encodes innate drives (survival + homeostasis + reproduction). Farming,
building, storing, and a self-sustaining population are NEVER rewarded — they can only
appear as learned, instrumentally-optimal behavior. This script runs the trained policy
against a random-weight baseline on the SAME world (no respawn — a real cohort that must
survive on its own) and checks three emergence gates:

  1. SURVIVAL   — trained cohort lives longer than random (max/mean lifespan).
  2. FARMING    — trained agents draw a meaningfully larger share of calories from crops
                  than random (agriculture emerged though it was never rewarded).
  3. PERSISTENCE— trained population survives a long horizon instead of collapsing.

Exit code is 0 only if all gates pass, so it doubles as a CI/notebook check.

Run:
  .venv/bin/python scripts/probe.py --checkpoint runs/ckpt.pt
  .venv/bin/python scripts/probe.py --checkpoint /kaggle/working/ckpt_512.pt --size 256 \
      --agents 256 --max-agents 1024 --steps 720
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from world import World, WorldConfig
from sim import Simulation, SimConfig, Actions
from sim.actions import build_mask
from train.policy import ArlianPolicy, DEVICE
from train.ppo import RolloutStats


def run_rollout(world, cfg, policy, steps: int, seed: int = 0) -> dict:
    """Roll a policy for `steps` with NO respawn (a real cohort) and return RolloutStats
    summary plus the final living population."""
    sim = Simulation(world, cfg)
    sim.reset(seed=seed)
    stats = RolloutStats(sim)
    for _ in range(steps):
        mask = build_mask(world, sim.state, sim.store, cfg)
        obs = sim.observe()
        with torch.no_grad():
            act = policy.act(obs.spatial, obs.vector, mask)
        actions = Actions(
            primary=act["primary"].cpu().numpy().astype(np.int32),
            param=act["param"].cpu().numpy().astype(np.int32),
            emit=act["emit"].cpu().numpy().astype(np.int32),
        )
        out = sim.step(actions)
        stats.record(sim, actions, out.info)
        if sim.store.n_living_agents() == 0:
            break
    s = stats.summary()
    s["final_pop"] = int(sim.store.n_living_agents())
    return s


def _fmt(s: dict) -> str:
    return (
        f"pop_mean={s['pop_mean']:.0f} final_pop={s['final_pop']} "
        f"max_age={s['max_age']} mean_age={s['mean_age']:.1f} "
        f"pct_farmed={s['pct_farmed']:.3f} built={s['structures_built']} "
        f"spec={s['specialization_index']:.3f} mi={s['signal_action_mi']:.3f} "
        f"births={s['births']} deaths={s['deaths_total']}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe a trained ArlianPolicy for emergence.")
    ap.add_argument("--checkpoint", type=str, required=True, help="trained policy .pt")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--agents", type=int, default=128, help="initial living agents")
    ap.add_argument("--max-agents", type=int, default=None, help="slot cap (default 4*agents)")
    ap.add_argument("--steps", type=int, default=720, help="rollout horizon (≈2 sim years)")
    ap.add_argument("--seed", type=int, default=42)
    # gate thresholds (tunable as the policy matures)
    ap.add_argument("--survival-margin", type=float, default=1.15,
                    help="trained max_age must exceed random max_age by this factor")
    ap.add_argument("--farm-margin", type=float, default=0.05,
                    help="trained pct_farmed must exceed random by at least this")
    ap.add_argument("--persist-frac", type=float, default=0.10,
                    help="trained final_pop must be >= this fraction of init agents")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    max_agents = args.max_agents if args.max_agents is not None else args.agents * 4
    print(f"[probe] world {args.size}x{args.size} seed={args.seed}  "
          f"agents={args.agents}/{max_agents}  steps={args.steps}  device={DEVICE}")
    world = World.generate(WorldConfig(width=args.size, height=args.size, seed=args.seed))
    cfg = SimConfig(max_agents=max_agents, init_agents=args.agents)

    trained = ArlianPolicy(n_symbols=cfg.n_symbols).to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    trained.load_state_dict(ckpt["policy"] if "policy" in ckpt else ckpt)
    trained.eval()

    random_policy = ArlianPolicy(n_symbols=cfg.n_symbols).to(DEVICE)
    random_policy.eval()

    print("[probe] rolling out trained policy...")
    t = run_rollout(world, cfg, trained, args.steps, seed=args.seed)
    print(f"   trained: {_fmt(t)}")
    print("[probe] rolling out random baseline...")
    r = run_rollout(world, cfg, random_policy, args.steps, seed=args.seed)
    print(f"   random : {_fmt(r)}")

    # ---- gates ----
    survival = t["max_age"] >= args.survival_margin * max(1, r["max_age"])
    farming = (t["pct_farmed"] - r["pct_farmed"]) >= args.farm_margin
    persistence = t["final_pop"] >= args.persist_frac * args.agents

    def line(name, ok, detail):
        print(f"   [{'PASS' if ok else 'FAIL'}] {name:12} {detail}")

    print("\n[probe] emergence gates:")
    line("survival", survival,
         f"trained max_age {t['max_age']} vs random {r['max_age']} "
         f"(need ≥{args.survival_margin}×)")
    line("farming", farming,
         f"pct_farmed trained {t['pct_farmed']:.3f} − random {r['pct_farmed']:.3f} "
         f"(need ≥{args.farm_margin})")
    line("persistence", persistence,
         f"final_pop {t['final_pop']} (need ≥{args.persist_frac:.0%} of {args.agents})")

    all_pass = survival and farming and persistence
    print(f"\n[probe] {'ALL GATES PASS — emergence confirmed' if all_pass else 'some gates failed (early in training, or thresholds too strict)'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
