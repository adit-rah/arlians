# Arlians Agent Layer — Findings & Insights

Non-obvious things learned while building the simulation layer (Phases 0–7). These are
the lessons that aren't recoverable from the code itself — read this before training.

> Companion docs: [`BUILD_STATUS.md`](BUILD_STATUS.md) (what's built / how to run),
> and the full design+build spec in `../planning-prompt-glowing-lake.md` (Parts I–III).

---

## 1. The world genuinely makes farming optimal — confirmed *before* any learning

The central thesis of the design (Principle 4: *the world must make civilization the
optimal survival strategy, not merely allow it*) is **empirically validated by scripted
agents**, no training required:

- On a sparse-wild world, a scripted **farmer produced 1.2 food vs a forager's 0.0102
  (~120×)** over a season (Phase 3 gate).
- Wild food is exponentially sparse (~0.1% of tiles), depletes on harvest, and collapses
  in winter; crops are a *separate* food source the agent creates on fertile/watered
  tiles. So where wild food is scarce, farming dominates by two orders of magnitude.

**Implication:** agents have a steep, real incentive to discover farming. If training
fails to produce farming, suspect the *learner* (exploration/credit assignment), not the
world.

## 2. Malthusian collapse is the dominant failure mode of reproduction

Under **naive play** (random actions, or "reproduce whenever able"), enabling
reproduction makes the population **breed itself to death** — reproduction's energy tax
(0.4/birth) outruns careless foraging. Observed repeatedly in Phase 5 scripted demos:
populations crash to 0 within ~300 steps.

**Implications:**
- Multi-year population stability is a genuinely *learned* behavior (balance breeding
  against survival + navigate to food), **not** a property of the mechanism.
- The soft carrying cap + difficulty curriculum exist precisely to manage this.
- The Phase-1 random smoke test had to *exclude* REPRODUCE (and later predators and
  catastrophes) to keep testing what it was designed to test.

## 3. `respawn_dead` churn dilutes every spatial/clustering metric

`Simulation.respawn_dead` (the Phase 1–4 training auto-reset) teleports fresh agents to
**uniformly random land tiles** each step. Under high mortality this means most
agent-steps belong to just-respawned, randomly-placed agents — which **washes out**
water-anchoring, settlement-clustering, and specialization signals (e.g. Phase 2
water-occupancy was 0.230 seeking vs 0.229 random — a real-but-tiny gap).

**Implications:**
- Spatial-clustering gates are only meaningful with **trained, low-mortality** policies.
- For training, consider respawning near water/parents rather than uniformly, or rely on
  reproduction (Phase 5+, children spawn near parents) instead of `respawn_dead`.

## 4. Shelter is finite insulation — the poles stay lethal (by design)

`shelter_temp_bonus` (0.3) cannot lift `eff_temp` above the warmth threshold (0.45) on
the coldest tiles (temp → ~0). So **shelter saves you in a temperate winter but not at
the frozen poles**. This is intended emergent pressure: it pushes settlement toward the
warmer fertile zones (where farming also works), rather than letting agents survive
anywhere. Tune `shelter_temp_bonus` / `thermal_target_temp` if you want habitable poles.

## 5. The reward "dilution" worry was mostly a non-issue

Early concern: in Phase 1, `comfort = (energy + hydration + thermal)/3` but hydration &
thermal were pinned at 1.0, so ⅔ of reward looked like a constant. **In practice this is
largely absorbed** by (a) the value-function baseline (a constant reward offset cancels
in the advantage) and (b) advantage normalization (the ×⅓ scale cancels too). So it's
cosmetic, not a real signal-dilution. Still, averaging only *active* drives per phase
would be marginally cleaner.

## 6. Two different wall-protection mechanisms

Walls protect against **predators by blocking their movement** (predators won't step onto
a wall tile), but `wall_defense_mult` (0.3×) only reduces damage in **agent-vs-agent**
combat. Two distinct mechanisms — don't expect `wall_defense_mult` to matter against
predators.

## 7. The signal↔action MI metric is a working "is language used?" detector

The Phase-7 mutual-information metric reads **2.0 bits** when an agent's signal
deterministically predicts its action, and **0.009 bits** when the signal is random. So
during training, `signal_action_mi` rising above ~0 is real evidence the emergent
communication channel is being *used*, not drifting. Likewise `specialization_index`
(pairwise JS divergence of action distributions): 0.0 when all agents behave identically,
0.36 nats when roles differ.

## 8. Combat correctness hinges on the pre-step HP snapshot

`resolve_combat` snapshots HP/positions **once** before applying any damage, accumulates
damage per target, then applies it atomically. This is what makes **mutual attacks both
land** in the same step (A and B attacking each other both take damage) — a subtle but
important fairness/determinism property.

## 9. Predator-prey stays stable; population tracks the target

With `pred_per_agents = 0.05`, predator count holds at exactly ~5% of living agents
(10 predators @ 200 agents; 78 @ ~1530) and neither population collapses over multi-year
runs. The scent-gradient AI (gaussian-blurred agent density, σ=3) gives predators a
navigable trail within `pred_sense_radius`.

## 10. Everything composes — the grand integration smoke

4096 agents on a 256² world with **every system live at once** (forage/farm/drink/
shelter/store/reproduce/evolve/fight + predators + catastrophes + signaling) runs a full
360-day year in **8.3s (43 steps/s)** with no errors and sensible cross-system metrics.
This is the real proof the layered phases didn't break each other.

---

## Process findings (for running fleets of build agents)

- **Freeze the contracts first.** All 14 Sonnet build agents worked against the frozen
  `§1` interfaces (config/state/actions/observe/simulation). **Zero contract violations**
  across the whole build — divergence never happened because there was nothing to diverge
  on. This is the single highest-leverage decision.
- **Agents drop. Plan for it.** 3 of 14 agents died mid-task (2 API socket errors, 1
  session limit). Because their work lived in isolated worktrees against frozen contracts,
  recovery was always: pull the (often unverified) code, review it, finish/run the tests
  centrally. No agent-drop cost more than a short orchestrator cleanup.
- **A dropped agent usually finished *implementation* but not *tests*.** Treat
  worktree code from a dropped agent as **unverified** — review it and write/run the tests
  before trusting it.
- **Per-phase two-tier gates kept us honest.** Every phase had a CODE gate (mechanism +
  scripted demo, validated now) and a BEHAVIORAL gate (emergence, deferred to GPU). Never
  conflate "the mechanism works" with "the behavior emerged."
- **Run a grand all-systems smoke at the end.** It caught a real plumbing gap
  (`deaths_by_cause` wasn't flowing from `step` to the logger) that no single-phase test
  would have surfaced.
