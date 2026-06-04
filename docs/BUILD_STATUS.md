# Arlians Agent Layer — Build Status

Status as of the Phase 0–7 build (branch `build/sim-layer`). The simulation layer is
**code-complete**: all 8 mechanics, all 7 phases, **224 tests passing**. Emergence
itself awaits GPU training (see "Deferred" below).

> See [`FINDINGS.md`](FINDINGS.md) for insights, and `../planning-prompt-glowing-lake.md`
> for the full design + build spec (Parts I–III).

---

## What's built

| Layer | Modules | Status |
|---|---|---|
| Read-only substrate | `world/` (unchanged) | static world generator |
| Frozen contracts | `sim/config.py`, `sim/state.py`, `sim/actions.py`, `sim/observe.py`, `sim/simulation.py` | locked §1 |
| Dynamics | `sim/dynamics.py` (drives, forage, eat, drink, crops, thermal, build, storage, combat, craft) | Phases 1–6 |
| Threats | `sim/threats.py` (predators, catastrophes) | Phases 6–7 |
| Reproduction | `sim/reproduce.py` (births, genome mutation, death causes) | Phase 5 |
| Curriculum | `sim/curriculum.py` (difficulty D) | done |
| Metrics | `sim/metrics.py` (all Section-H signals) | Phases 0–7 |
| Render | `sim/render.py` (top-down RGB) | done |
| Learner | `train/` (policy, rollout, ppo, run) | Phase 1, reused all phases |

### The 8 mechanics (all live)
forage · farm (plant/grow/harvest) · drink · shelter · store · reproduce/evolve · fight/defend (predators, combat, weapons, walls) · signal — plus episodic catastrophes.

## Per-phase gate results (CODE gates — all green)

| Phase | Mechanic | Headline result |
|---|---|---|
| 0 | Vectorized env + observation | `observe()` == `world.get_window` (1e-5); 146k agent-windows/s |
| 1 | Energy + foraging + PPO | PPO loss 34.9→0.46, returns rise; population held |
| 2 | Hydration + DRINK | drinking → 3.1× fewer dehydration deaths |
| 3 | **Crops (keystone)** | **farmer 1.2 vs forager 0.0102 food (~120×)** |
| 4 | Thermal + shelter + storage | sheltered 0 exposure deaths vs unsheltered 8; storage bank-and-draw works |
| 5 | Reproduction + genome | births/mutation/soft-cap verified; genome drift tracked |
| 6 | Predators + combat + weapons + walls | walls block predators (0 vs 10 deaths); armed 2× dmg; predator-prey stable |
| 7 | Catastrophes + signaling | signal↔action MI 2.0 bits (structured) vs 0.009 (random); specialization 0.36 vs 0 |

## Test inventory

**224 tests**, one suite per component. Run all:
```bash
.venv/bin/python -m pytest tests/ -q
```
Key files: `test_contracts.py` (frozen schemas), `test_observe.py` (correctness pin),
`test_dynamics*.py` (Phases 1–4 mechanics), `test_combat_phase6.py`,
`test_predators_phase6.py`, `test_catastrophe_phase7.py`, `test_train.py` (PPO),
and per-phase `test_phaseN_gate.py` files.

## How to run

```bash
# 1. (one-time) the world must exist; regenerate if needed
.venv/bin/python generate.py --seed 42

# 2. tests
.venv/bin/python -m pytest tests/ -q

# 3. see an UNTRAINED policy in action (random weights) -> animated gif + metrics
.venv/bin/python scripts/demo.py

# 4. reduced-scale training smoke (CPU/MPS) -> PPO runs, loss moves
.venv/bin/python train/run.py
```

Env runs on numpy (CPU); the learner uses PyTorch (MPS on Apple Silicon, else CPU/CUDA).

## Deferred to GPU training (the behavioral gates)

The mechanisms are validated; **emergent behavior is not yet trained**. These all require
running the shared-weights PPO at scale:

- learned **farming > foraging** crossover (world already makes it optimal — §1 of FINDINGS)
- **winter survival** behavior (build shelter / store before winter)
- **multi-year self-sustaining population** (must beat Malthusian collapse — §2 of FINDINGS)
- **defensive building** under predation; weapon use
- **cooperation / specialization / emergent language** (the hardest target; metrics ready)

## Known follow-ups before/within training

1. **Toggle off `respawn_dead` at Phase 5+** so reproduction sustains the population
   (currently the training loop calls it every step as the Phase 1–4 crutch).
2. **Spatial metrics need trained policies** — `respawn_dead` churn dilutes clustering
   signals (§3 of FINDINGS).
3. **Throughput is CPU-bound** in the numpy env, not the GPU — the policy is small. For
   scale, vectorize the env hot paths (forage/build loops still iterate slots) or run
   many env workers in parallel.
4. Optional: average only *active* drives in `comfort` per phase (§5 of FINDINGS).

## Reference artifacts (in `data/`, git-ignored)

`phase0_frame.png` … `phase7_final.png`, `final_world.png` (4096-agent world),
`data/demo.gif` (default rollout; omit `--checkpoint` for untrained baseline). Regenerate via `scripts/demo.py`.
