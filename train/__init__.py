"""
Arlians learner — PyTorch shared-weights multi-agent PPO over the continuing stream.

Filled starting Phase 1 by the learner fleet role:
  policy.py  - ArlianPolicy (CNN over the window + MLP over interoception + multi-head)
  rollout.py - SlotRolloutBuffer (M slots x T horizon, death-masked)
  ppo.py     - continuing-stream PPO (death=done, birth=auto-reset, GAE resets at done)
  run.py     - entrypoint wiring SimConfig + TrainConfig

Reward is purely individual (homeostasis + alive); population/lineage persistence
emerges from genome selection + population-weighted gradients (decision 8).
"""
