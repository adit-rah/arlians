"""
Demo: watch an UNTRAINED policy (random weights) drive the Arlians.

No training is performed — this instantiates ArlianPolicy with its random
initialization and rolls it out, so you can see the baseline "before learning"
behavior for reference. Expect aimless/near-random behavior: agents wander, act
roughly uniformly over whatever the action mask allows, mostly fail to sustain
themselves under the default economy (respawn_dead keeps the world populated).

Outputs:
  data/demo_untrained.gif  — animated top-down rollout
  prints a one-year metrics summary

Run:
  .venv/bin/python scripts/demo_untrained.py
  .venv/bin/python scripts/demo_untrained.py --steps 360 --agents 800 --size 192
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v3 as iio

# allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from world import World, WorldConfig
from sim import Simulation, SimConfig, Actions
from sim.actions import build_mask
from sim.metrics import MetricsLogger
from sim.render import render_frame
from train.policy import ArlianPolicy, DEVICE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--agents", type=int, default=600)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/demo_untrained.gif")
    ap.add_argument("--render-every", type=int, default=3)
    ap.add_argument("--upscale", type=int, default=3)
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="load TRAINED policy weights from a checkpoint .pt; "
                         "omit to use random (untrained) weights")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    print(f"[demo] generating {args.size}x{args.size} world (seed={args.seed})...")
    world = World.generate(WorldConfig(width=args.size, height=args.size, seed=args.seed))
    cfg = SimConfig(max_agents=args.agents * 3, init_agents=args.agents)
    sim = Simulation(world, cfg)
    sim.reset(seed=0)

    policy = ArlianPolicy(n_symbols=cfg.n_symbols).to(DEVICE)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=DEVICE)
        policy.load_state_dict(ckpt["policy"] if "policy" in ckpt else ckpt)
        tag = f"TRAINED ({args.checkpoint})"
    else:
        tag = "untrained (random weights)"
    policy.eval()
    print(f"[demo] {tag} ArlianPolicy on {DEVICE} "
          f"({sum(p.numel() for p in policy.parameters()):,} params)")

    log = MetricsLogger()
    frames = []
    for t in range(args.steps):
        mask = build_mask(world, sim.state, sim.store, cfg)
        with torch.no_grad():
            act = policy.act(sim.observe().spatial, sim.observe().vector, mask)
        actions = Actions(
            primary=act["primary"].cpu().numpy().astype(np.int32),
            param=act["param"].cpu().numpy().astype(np.int32),
            emit=act["emit"].cpu().numpy().astype(np.int32),
        )
        out = sim.step(actions)
        sim.respawn_dead(seed=t)          # keep the world populated (Phase 1-4 crutch)
        log.record_step(sim, out.info, actions)
        if t % args.render_every == 0:
            fr = render_frame(world, sim.state, sim.store)
            if args.upscale > 1:
                fr = np.repeat(np.repeat(fr, args.upscale, 0), args.upscale, 1)
            frames.append(fr)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # duration (ms/frame) drives gif animation correctly (fps kwarg silently 1-frames it)
    iio.imwrite(args.out, np.stack(frames), duration=80, loop=0)

    s = log.year_summary()
    print(f"\n[demo] wrote {args.out}  ({len(frames)} frames)")
    print("[demo] untrained-policy rollout summary:")
    print(f"   population mean/min/max : {s['population_mean']:.0f}/{s['population_min']}/{s['population_max']}")
    print(f"   deaths_by_cause         : {s['deaths_by_cause']}")
    print(f"   pct_calories_farmed     : {s['pct_calories_farmed']:.3f}")
    print(f"   specialization_index    : {s['specialization_index']:.3f} nats")
    print(f"   signal_action_mi        : {s['signal_action_mi']:.3f} bits")
    print("   (all near baseline — this is what 'before training' looks like)")


if __name__ == "__main__":
    main()
