"""
Reproduction, death, and genome evolution (build-spec §2.8, §2.5 trait map).

Filled by the env-dynamics fleet role:
  Phase 1: death resolution (health <= 0 -> free the slot, mark done)
  Phase 5: REPRODUCE action (energy-gated, costed, cooldown), child placement in a
           nearby free slot, genome inheritance + gaussian mutation, lineage_id.
  traits_from_genome(): map genome[M,G] -> body traits (metabolism, water_need,
           cold_tol, capacity). Used by dynamics; selection is implicit via survival.
"""
