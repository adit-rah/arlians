"""
Generate a world and export it to disk.

Usage:
    python generate.py
    python generate.py --seed 7 --width 1024 --height 1024 --output ./data/world_7
    python generate.py --seed 42 --snapshot-steps 0 90 180 270
"""
import argparse
import time

from world.config import WorldConfig
from world.world import World
from world.export import export_world, export_season_snapshot


def main():
    parser = argparse.ArgumentParser(description="Generate an Arlians world.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--snapshot-steps", type=int, nargs="*", default=[])
    args = parser.parse_args()

    output = args.output or f"./data/world_{args.seed}"
    cfg = WorldConfig(seed=args.seed, width=args.width, height=args.height)

    t0 = time.time()
    world = World.generate(cfg)
    elapsed = time.time() - t0
    print(f"[world] Generation took {elapsed:.1f}s")

    export_world(world, output)
    for t in args.snapshot_steps:
        export_season_snapshot(world, t, output)

    import numpy as np
    land_count = int((world.elevation >= cfg.sea_level).sum())
    river_cells = int(world.river_mask.sum())
    wild_dense = float((world.base_resources["wild_food"] > 0.3).mean())
    biome_counts = {int(b): int((world.biome_map == b).sum()) for b in np.unique(world.biome_map)}
    print("\n--- World Summary ---")
    print(f"Land tiles:     {land_count:,} / {cfg.width * cfg.height:,}")
    print(f"River cells:    {river_cells:,}")
    print(f"Wild food >0.3: {wild_dense:.1%}  (target <15%)")
    print(f"Biomes:         {biome_counts}")


if __name__ == "__main__":
    main()
