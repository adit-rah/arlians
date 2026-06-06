# tests/ — pytest suite (deterministic, seeded)

One module per sim component, plus a per-phase **gate test** that asserts a numeric
threshold. Tests are the safety net for the frozen contracts and the per-mechanic behavior.

## Layout

- `test_contracts.py` — **the anti-divergence pin.** Asserts the frozen shapes/signatures
  (`SimConfig` fields, `WorldState`/`EntityStore` arrays, obs channel counts, action enum,
  `Simulation` method signatures). If you change a contract, this fails first — that's the
  point. Don't "fix" it by loosening the assertion; re-lock deliberately.
- `test_observe.py` — pins the vectorized `observe()` window against a reference
  `world.get_window` slice (atol ~1e-5). Correctness of the hot-path rewrite.
- `test_dynamics.py`, `test_dynamics_phase2/3/4.py` — drives decay, forage deplete/regrow,
  drink, crop grow→mature→rot, build/store, spoilage, thermal/exposure.
- `test_reproduce_phase5.py` — genome inherit+mutate, death resolution, soft cap.
- `test_combat_phase6.py`, `test_predators_phase6.py` — deterministic HP exchange, predator
  scent-gradient movement.
- `test_catastrophe_phase7.py` — event_mask effects per catastrophe type.
- `test_metrics.py`, `test_render.py` — logger aggregation; renderer output shape.
- `test_train.py` — PPO buffer masks dead slots, GAE resets at `done`, train loop composes,
  checkpoint/resume round-trips.
- `test_phase1_gate.py` … `test_phase7_gate.py` — the per-phase numeric **CODE gates**
  (mechanic fires, loss moves at reduced scale). Note: the **behavioral** gates (e.g.
  "≥50% calories farmed") are NOT met and are deferred to GPU-scale runs — don't expect them
  to pass at smoke scale.

## Run

```bash
.venv/bin/python -m pytest -q                       # everything
.venv/bin/python -m pytest tests/test_contracts.py -q
.venv/bin/python -m pytest tests/ -k "train or ppo or rollout" -q
```

## Conventions / gotchas

- Always seed RNGs; tests must be deterministic.
- Some helper fixtures set non-zero neutrals (e.g. `thermal=1.0`) — when a later phase makes a
  drive active, earlier single-agent test helpers may need that default or they break. (This
  bit us when thermal went live in Phase 4.)
- The random-action smoke helpers **exclude REPRODUCE** and set `pred_per_agents=0` +
  `catastrophe_prob=0`, otherwise Malthusian collapse / catastrophes make them flaky.
- Adding a mechanic? Add its component test AND extend (don't rewrite) the relevant gate test.
