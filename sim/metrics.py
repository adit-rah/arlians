"""
MetricsLogger — per-year emergence metrics (build-spec §4.1).

Filled by the tooling fleet role (scaffold in Phase 0; metrics added as their
mechanics come online):
  population; births/deaths by cause; % calories foraged vs farmed; mean lifetime
  displacement; settlement clustering on fertile tiles; structures built/occupied;
  winter survival rate; storage by season; gene-frequency drift; specialization
  index (pairwise JS divergence of action distributions); signal<->action MI.

Outputs JSONL (+ optional TensorBoard). These distinguish *emerged* civilization
from accidental survival — they are how behavioral gates are judged.
"""
