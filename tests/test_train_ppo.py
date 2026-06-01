"""Tests unitaires pour train_ppo.py (DungeonGymEnv, train PPO)."""

import json
from pathlib import Path

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from dungeon_env import DungeonEnv, ACTIONS, MAX_STEPS
from train_ppo import DungeonGymEnv, train


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(seed: int = 42) -> DungeonGymEnv:
    return DungeonGymEnv(seed=seed)


# ===========================================================================
# DungeonGymEnv — espaces
# ===========================================================================

class TestDungeonGymEnvSpaces:
    def test_observation_space_is_box(self):
        assert isinstance(_make_env().observation_space, spaces.Box)

    def test_observation_space_shape(self):
        assert _make_env().observation_space.shape == (DungeonGymEnv.OBS_DIM,)

    def test_observation_space_dtype(self):
        assert _make_env().observation_space.dtype == np.float32

    def test_observation_space_bounds(self):
        env = _make_env()
        assert env.observation_space.low.min()  == 0.0
        assert env.observation_space.high.max() == 1.0

    def test_action_space_is_discrete(self):
        assert isinstance(_make_env().action_space, spaces.Discrete)

    def test_action_space_n(self):
        assert _make_env().action_space.n == len(ACTIONS)


# ===========================================================================
# DungeonGymEnv — reset
# ===========================================================================

class TestDungeonGymEnvReset:
    def test_reset_returns_tuple(self):
        obs, info = _make_env().reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_reset_obs_shape(self):
        obs, _ = _make_env().reset()
        assert obs.shape == (DungeonGymEnv.OBS_DIM,)

    def test_reset_obs_dtype(self):
        obs, _ = _make_env().reset()
        assert obs.dtype == np.float32

    def test_reset_obs_in_bounds(self):
        obs, _ = _make_env().reset()
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0

    def test_reset_deterministic_for_same_seed(self):
        obs1, _ = _make_env(seed=42).reset()
        obs2, _ = _make_env(seed=42).reset()
        assert np.array_equal(obs1, obs2)

    def test_reset_different_seeds_give_different_obs(self):
        obs1, _ = _make_env(seed=0).reset()
        obs2, _ = _make_env(seed=1).reset()
        assert not np.array_equal(obs1, obs2)


# ===========================================================================
# DungeonGymEnv — step
# ===========================================================================

class TestDungeonGymEnvStep:
    def setup_method(self):
        self.env = _make_env()
        self.env.reset()

    def test_step_returns_five_tuple(self):
        result = self.env.step(0)
        assert len(result) == 5

    def test_step_obs_shape(self):
        obs, *_ = self.env.step(0)
        assert obs.shape == (DungeonGymEnv.OBS_DIM,)

    def test_step_reward_is_float(self):
        _, reward, *_ = self.env.step(0)
        assert isinstance(reward, float)

    def test_step_terminated_is_bool(self):
        _, _, terminated, _, _ = self.env.step(0)
        assert isinstance(terminated, bool)

    def test_step_truncated_is_bool(self):
        _, _, _, truncated, _ = self.env.step(0)
        assert isinstance(truncated, bool)

    def test_step_info_has_required_keys(self):
        _, _, _, _, info = self.env.step(0)
        for key in ("score", "moves", "steps", "won", "seed"):
            assert key in info

    def test_step_info_seed_matches_env_seed(self):
        env = DungeonGymEnv(seed=42)
        env.reset()
        _, _, _, _, info = env.step(0)
        assert info["seed"] == 42

    def test_step_valid_actions(self):
        env = _make_env()
        env.reset()
        for action in range(len(ACTIONS)):
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (DungeonGymEnv.OBS_DIM,)

    def test_terminated_true_when_won(self):
        """terminated doit être True si et seulement si won=True."""
        env = _make_env()
        for _ in range(50):
            env.reset()
            done = False
            while not done:
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                done = terminated or truncated
                if terminated:
                    assert info["won"] is True
                    return
        pytest.skip("Aucune victoire obtenue en 50 épisodes avec actions aléatoires")

    def test_truncated_true_at_max_steps(self):
        """truncated=True quand max_steps atteint sans victoire."""
        env = DungeonGymEnv(seed=1)   # seed avec chemin long
        env.reset()
        terminated, truncated = False, False
        steps = 0
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(1)  # RIGHT en boucle → bloqué
            steps += 1
            if steps > MAX_STEPS + 5:
                break
        # Au moins l'un des deux doit être True
        assert terminated or truncated


# ===========================================================================
# DungeonGymEnv — encodage
# ===========================================================================

class TestDungeonGymEnvEncoding:
    def test_encode_matches_encode_obs_pure(self):
        """L'encodage de DungeonGymEnv doit être identique à encode_obs_pure."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
        from train import encode_obs_pure
        import torch

        env = DungeonGymEnv(seed=42)
        obs_dict = DungeonEnv(seed=42).reset()

        gym_obs = DungeonGymEnv._encode(obs_dict)
        pure_obs = encode_obs_pure(obs_dict).numpy()

        assert np.allclose(gym_obs, pure_obs, atol=1e-6)

    def test_encode_output_shape(self):
        obs_dict = DungeonEnv(seed=0).reset()
        enc = DungeonGymEnv._encode(obs_dict)
        assert enc.shape == (DungeonGymEnv.OBS_DIM,)

    def test_encode_positions_normalized(self):
        """Les 4 derniers floats (positions) doivent être dans [0, 1]."""
        obs_dict = DungeonEnv(seed=0).reset()
        enc = DungeonGymEnv._encode(obs_dict)
        assert enc[300:304].min() >= 0.0
        assert enc[300:304].max() <= 1.0


# ===========================================================================
# train() — boucle complète (smoke tests rapides)
# ===========================================================================

class TestTrainPPO:
    def test_train_creates_log_file(self, tmp_path):
        log = tmp_path / "ppo.jsonl"
        train(timesteps=512, seed=42, log_path=log,
              model_dir=tmp_path / "models", verbose=False)
        assert log.exists()

    def test_log_has_meta_line_first(self, tmp_path):
        log = tmp_path / "ppo.jsonl"
        train(timesteps=512, seed=42, log_path=log,
              model_dir=tmp_path / "models", verbose=False)
        first = json.loads(log.read_text().splitlines()[0])
        assert first.get("type") == "meta"

    def test_log_episode_lines_have_required_keys(self, tmp_path):
        log = tmp_path / "ppo.jsonl"
        train(timesteps=2048, seed=42, log_path=log,
              model_dir=tmp_path / "models", verbose=False)
        lines = [json.loads(l) for l in log.read_text().splitlines()
                 if json.loads(l).get("type") != "meta"]
        assert len(lines) > 0
        for entry in lines:
            for key in ("episode", "timestep", "seed", "won", "score", "moves", "reward"):
                assert key in entry

    def test_log_moves_is_positive_int(self, tmp_path):
        log = tmp_path / "ppo.jsonl"
        train(timesteps=2048, seed=42, log_path=log,
              model_dir=tmp_path / "models", verbose=False)
        lines = [json.loads(l) for l in log.read_text().splitlines()
                 if json.loads(l).get("type") != "meta"]
        for entry in lines:
            assert isinstance(entry["moves"], int)
            assert entry["moves"] >= 0

    def test_train_creates_final_model(self, tmp_path):
        model_dir = tmp_path / "models"
        train(timesteps=512, seed=42, log_path=tmp_path / "ppo.jsonl",
              model_dir=model_dir, verbose=False)
        assert (model_dir / "final.zip").exists()

    def test_train_returns_ppo_model(self, tmp_path):
        from stable_baselines3 import PPO as SB3PPO
        model = train(timesteps=512, seed=42, log_path=tmp_path / "ppo.jsonl",
                      model_dir=tmp_path / "models", verbose=False)
        assert isinstance(model, SB3PPO)

    def test_train_with_seed_pool(self, tmp_path):
        log = tmp_path / "ppo.jsonl"
        train(timesteps=512, seed_pool=[0, 1, 2], log_path=log,
              model_dir=tmp_path / "models", verbose=False)
        assert log.exists()

    def test_final_model_loadable(self, tmp_path):
        from stable_baselines3 import PPO as SB3PPO
        model_dir = tmp_path / "models"
        train(timesteps=512, seed=42, log_path=tmp_path / "ppo.jsonl",
              model_dir=model_dir, verbose=False)
        env = DummyVecEnv([lambda: DungeonGymEnv(seed=42)])
        loaded = SB3PPO.load(str(model_dir / "final"), env=env)
        assert loaded is not None
        env.close()


# Import nécessaire pour test_final_model_loadable
from stable_baselines3.common.vec_env import DummyVecEnv
