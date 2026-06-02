"""
World dynamics — pure functions advancing mutable state one step (build-spec §2.3-2.5).

Filled incrementally by the env-dynamics fleet role:
  Phase 1: forage depletion/regrowth (§2.4), drive decay + health update (§2.5)
  Phase 3: crop growth + winter-failure rot (§2.3)
  Phase 4: thermal update, spoilage, structure decay

All functions operate in-place / vectorized on WorldState + EntityStore, driven by
SimConfig + the world's seasonal scalars. No magic numbers — everything from SimConfig.
"""
