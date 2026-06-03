"""
Tests for the Arlians PPO learner stack (build-spec §5 Phase 1).

Three test groups:
  1. Policy forward pass — shapes, ranges, masking correctness.
  2. Buffer GAE — advantage reset at done, dead-slot masking.
  3. Integration smoke — 3 PPO updates on tiny world; finite losses; population > 0.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

# ---- shared device
from train.policy import ArlianPolicy, DEVICE, N_PRIMARY, N_PARAM

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ===========================================================================
# 1. Policy forward pass
# ===========================================================================

class TestPolicyForward:
    """§3.1 policy architecture and masking (build-spec §5)."""

    @pytest.fixture
    def policy(self):
        torch.manual_seed(SEED)
        return ArlianPolicy(n_symbols=8)

    @pytest.fixture
    def batch(self):
        """Random obs for M=16 agents."""
        M = 16
        rng = np.random.default_rng(SEED)
        spatial = rng.random((M, 24, 15, 15), dtype=np.float32)
        vector  = rng.random((M, 16), dtype=np.float32)
        # All actions legal
        mask    = np.ones((M, N_PRIMARY), dtype=bool)
        return spatial, vector, mask, M

    def test_act_output_shapes(self, policy, batch):
        spatial, vector, mask, M = batch
        out = policy.act(spatial, vector, mask)
        assert out["primary"].shape == (M,), "primary shape"
        assert out["param"].shape   == (M,), "param shape"
        assert out["emit"].shape    == (M,), "emit shape"
        assert out["logp"].shape    == (M,), "logp shape"
        assert out["value"].shape   == (M,), "value shape"

    def test_act_action_ranges(self, policy, batch):
        spatial, vector, mask, M = batch
        out = policy.act(spatial, vector, mask)
        assert out["primary"].min() >= 0 and out["primary"].max() < N_PRIMARY, \
            "primary out of range"
        assert out["param"].min() >= 0 and out["param"].max() < N_PARAM, \
            "param out of range"
        assert out["emit"].min() >= 0 and out["emit"].max() < 8, \
            "emit out of range"

    def test_logp_finite(self, policy, batch):
        spatial, vector, mask, M = batch
        out = policy.act(spatial, vector, mask)
        assert torch.isfinite(out["logp"]).all(), "logp must be finite"
        assert torch.isfinite(out["value"]).all(), "value must be finite"

    def test_primary_mask_single_legal(self, policy):
        """
        Set mask to allow ONLY action index 2 (FORAGE) for a single agent.
        Sample 200 times; assert only FORAGE is sampled.
        """
        M = 1
        rng = np.random.default_rng(SEED)
        spatial = rng.random((M, 24, 15, 15), dtype=np.float32)
        vector  = rng.random((M, 16), dtype=np.float32)

        # Only action 2 is legal
        mask = np.zeros((M, N_PRIMARY), dtype=bool)
        mask[0, 2] = True

        torch.manual_seed(SEED)
        sampled = set()
        for _ in range(200):
            out = policy.act(spatial, vector, mask)
            sampled.add(int(out["primary"][0].item()))

        assert sampled == {2}, (
            f"Expected only action 2 to be sampled, got: {sampled}"
        )

    def test_evaluate_shapes(self, policy, batch):
        spatial, vector, mask, M = batch
        # Get actions
        out = policy.act(spatial, vector, mask)
        spatial_t   = torch.as_tensor(spatial, device=DEVICE)
        vector_t    = torch.as_tensor(vector, device=DEVICE)
        mask_t      = torch.as_tensor(mask, device=DEVICE)

        logp, entropy, value = policy.evaluate(
            spatial_t, vector_t, mask_t,
            out["primary"], out["param"], out["emit"],
        )
        assert logp.shape    == (M,), "evaluate logp shape"
        assert entropy.shape == (M,), "evaluate entropy shape"
        assert value.shape   == (M,), "evaluate value shape"
        assert torch.isfinite(logp).all(),    "evaluate logp must be finite"
        assert torch.isfinite(entropy).all(), "evaluate entropy must be finite"
        assert torch.isfinite(value).all(),   "evaluate value must be finite"
        assert (entropy >= 0).all(),          "entropy must be >= 0"


# ===========================================================================
# 2. Buffer GAE
# ===========================================================================

class TestBufferGAE:
    """
    Verify GAE correctness for the continuing-stream buffer (build-spec §3.2).

    Uses a hand-built 1-slot, T=6 trajectory:
      t=0..2 alive, r=1
      t=3    done (death boundary)
      t=4..5 alive (new life after respawn), r=1

    Expected: advantages at t<=3 must NOT leak into t>3.
    Dead timesteps (tracked via valid_mask=False) must have advantage=0 and return=0.
    """

    def _make_buffer(self):
        from train.rollout import SlotRolloutBuffer
        M, T, vlen, nsym = 1, 6, 16, 8
        buf = SlotRolloutBuffer(M, T, vlen, nsym)
        return buf

    def test_gae_no_leakage_across_done(self):
        """
        Advantages at t=0..2 (before death) must not be affected by transitions
        at t=4..5 (after respawn). We verify this by checking sign/magnitude changes
        when we flip rewards on the post-death side.

        We use normalize_adv=False to check the raw GAE values directly — normalization
        is a global rescale and is NOT the property under test here.
        """
        from train.rollout import SlotRolloutBuffer

        def _build_and_gae(post_reward):
            buf = SlotRolloutBuffer(1, 6, 16, 8)
            rng = np.random.default_rng(SEED)
            sp  = rng.random((1, 24, 15, 15), dtype=np.float32)
            vec = rng.random((1, 16), dtype=np.float32)
            pm  = np.ones((1, N_PRIMARY), dtype=bool)

            for t in range(6):
                done_t       = np.array([t == 3], dtype=bool)
                alive_t      = np.array([t != 3], dtype=bool)   # dead only at done step
                r_t = 1.0 if t <= 3 else float(post_reward)
                reward_t     = np.array([r_t], dtype=np.float32)
                value_t      = torch.zeros(1, device=DEVICE)
                buf.store(
                    spatial=sp, vector=vec, primary_mask=pm,
                    primary=torch.zeros(1, dtype=torch.int64),
                    param=torch.zeros(1, dtype=torch.int64),
                    emit=torch.zeros(1, dtype=torch.int64),
                    logp=torch.zeros(1),
                    value=value_t,
                    reward=reward_t,
                    done=done_t,
                    alive_mask=alive_t,
                )
            last_val = torch.zeros(1, device=DEVICE)
            # normalize_adv=False so we can compare raw GAE values
            buf.compute_gae(last_val, gamma=0.99, lam=0.95, normalize_adv=False)
            return buf.advantages[0].cpu().numpy().copy()

        adv_low  = _build_and_gae(post_reward=-100.0)
        adv_high = _build_and_gae(post_reward=+100.0)

        # Pre-death advantages (t=0,1,2): must be identical regardless of post-death reward
        # because done=True at t=3 zeroes the (1-done) factor that would otherwise let
        # post-death value estimates leak backward through the GAE recursion.
        np.testing.assert_array_almost_equal(
            adv_low[:3], adv_high[:3],
            decimal=5,
            err_msg="Pre-death advantages should not depend on post-death rewards (GAE leakage)",
        )

        # Post-death advantages (t=4,5) should differ based on post_reward
        assert not np.allclose(adv_low[4:], adv_high[4:]), \
            "Post-death advantages should differ when post-death rewards differ"

    def test_dead_timestep_masked_to_zero(self):
        """
        The done timestep (t=3) has alive_mask=False.
        Its advantage and return must be exactly 0.
        """
        from train.rollout import SlotRolloutBuffer
        buf = SlotRolloutBuffer(1, 6, 16, 8)
        rng = np.random.default_rng(SEED)
        sp  = rng.random((1, 24, 15, 15), dtype=np.float32)
        vec = rng.random((1, 16), dtype=np.float32)
        pm  = np.ones((1, N_PRIMARY), dtype=bool)

        for t in range(6):
            done_t  = np.array([t == 3], dtype=bool)
            alive_t = np.array([t != 3], dtype=bool)
            reward_t = np.array([1.0], dtype=np.float32)
            buf.store(
                spatial=sp, vector=vec, primary_mask=pm,
                primary=torch.zeros(1, dtype=torch.int64),
                param=torch.zeros(1, dtype=torch.int64),
                emit=torch.zeros(1, dtype=torch.int64),
                logp=torch.zeros(1),
                value=torch.zeros(1, device=DEVICE),
                reward=reward_t,
                done=done_t,
                alive_mask=alive_t,
            )

        buf.compute_gae(torch.zeros(1, device=DEVICE), gamma=0.99, lam=0.95)

        # t=3 is dead (alive_mask=False) -> advantage and return must be 0
        adv_dead  = buf.advantages[0, 3].item()
        ret_dead  = buf.returns[0, 3].item()
        assert adv_dead == pytest.approx(0.0), f"Dead slot advantage should be 0, got {adv_dead}"
        assert ret_dead == pytest.approx(0.0), f"Dead slot return should be 0, got {ret_dead}"


# ===========================================================================
# 3. Integration smoke
# ===========================================================================

class TestIntegrationSmoke:
    """
    3 PPO updates on a tiny world to verify the full stack composes correctly.
    Checks: no exceptions, finite losses, population stays > 0.
    """

    @pytest.fixture(scope="class")
    def trained(self):
        """Run 3 PPO updates and return (metrics_history, sim, policy)."""
        import sys, os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)

        torch.manual_seed(SEED)
        np.random.seed(SEED)

        from world.config import WorldConfig
        from world.world import World
        from sim.config import SimConfig
        from sim.simulation import Simulation
        from train.ppo import train

        world_cfg = WorldConfig(width=128, height=128, seed=SEED)
        world = World.generate(world_cfg)

        sim_cfg = SimConfig(max_agents=128, init_agents=64)
        sim = Simulation(world, sim_cfg)

        policy = ArlianPolicy(n_symbols=sim_cfg.n_symbols)

        # 3 updates, T=16 keeps the test fast
        history = train(sim, policy, n_updates=3, T=16)
        return history, sim, policy

    def test_no_exception(self, trained):
        """Training completes without raising."""
        history, sim, policy = trained
        assert len(history) == 3

    def test_losses_finite(self, trained):
        """All reported losses must be finite numbers."""
        history, sim, policy = trained
        for row in history:
            assert np.isfinite(row["policy_loss"]), f"policy_loss not finite: {row}"
            assert np.isfinite(row["value_loss"]),  f"value_loss not finite: {row}"
            assert np.isfinite(row["entropy"]),     f"entropy not finite: {row}"
            assert np.isfinite(row["total_loss"]),  f"total_loss not finite: {row}"

    def test_population_alive(self, trained):
        """Population must be > 0 at all update steps (respawn_dead keeps it alive)."""
        history, sim, policy = trained
        for row in history:
            assert row["n_living"] > 0, f"Population dropped to 0 at update {row['update']}"

    def test_mean_reward_finite(self, trained):
        """Mean reward must be finite (rewards are non-trivial when agents are alive)."""
        history, sim, policy = trained
        for row in history:
            assert np.isfinite(row["mean_reward"]), f"mean_reward not finite: {row}"
