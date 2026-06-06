# scripts/ — runnable tools

## `demo.py` — roll out a policy and render a top-down GIF

Loads `ArlianPolicy` (random weights by default, or `--checkpoint path.pt` for a trained
one), steps the sim, renders frames, writes a GIF, and prints a rollout metrics summary
(population, deaths-by-cause, pct_calories_farmed, specialization, signal MI).

```bash
.venv/bin/python scripts/demo.py                                  # untrained baseline -> data/demo.gif
.venv/bin/python scripts/demo.py --checkpoint runs/ckpt.pt --size 512 --agents 96 \
    --render-every 1 --no-respawn --out data/trained.gif
```

### Key flags / gotchas

- **`--no-respawn`** (added in the survival-sim reframe): without it, dead slots are
  re-seeded at *random* tiles every step and each slot is a fixed-color dot, so the GIF looks
  like agents **teleporting**. With it you watch a real cohort move and thin out — use it for
  any honest visualization.
- **`--render-every 1`** (now the default): agents move ≤1 tile/step, so a larger stride makes
  motion look jumpy. Keep it at 1 for smooth playback.
- GIF timing uses `duration=` (ms/frame) — **never** imageio's `fps` kwarg (it silently writes
  a 1-frame GIF via Pillow).
- This script uses the heavy `MetricsLogger` (full Section-H metrics) — fine for a one-off
  demo, too slow for the training loop (which uses `train/ppo.py:RolloutStats`).
- `--max-agents` defaults to `3 × --agents`; `--agents` must be ≤ `--max-agents`.

When you train a policy and want to see it, this is the tool. Compare a `--checkpoint` GIF
against the no-checkpoint baseline to judge whether anything was learned.
