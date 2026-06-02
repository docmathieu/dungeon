"""Tests unitaires pour analyze/evaluate.py."""

import sys
import torch
import pytest
from pathlib import Path

# analyze/ n'est pas dans sys.path par défaut (conftest ajoute src/ seulement)
sys.path.insert(0, str(Path(__file__).parent.parent / "analyze"))

from evaluate import _parse_seeds, evaluate  # noqa: E402
from model import FiLMDQNetwork, ObsDQNetwork  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers — création de checkpoints temporaires
# ---------------------------------------------------------------------------

def _save_film(path: Path) -> None:
    net = FiLMDQNetwork()
    torch.save(net.state_dict(), path)


def _save_ppo(tmp_dir: Path) -> Path:
    from stable_baselines3 import PPO
    from train_ppo import DungeonGymEnv
    env = DungeonGymEnv(seed=0)
    model = PPO("MlpPolicy", env, verbose=0, n_steps=64, batch_size=32)
    out = tmp_dir / "final"
    model.save(str(out))
    env.close()
    return tmp_dir / "final.zip"


# ===========================================================================
# _parse_seeds — parsing de la liste de seeds
# ===========================================================================

class TestParseSeeds:
    def test_range(self):
        assert _parse_seeds("100-103") == [100, 101, 102, 103]

    def test_list(self):
        assert _parse_seeds("0,5,10") == [0, 5, 10]

    def test_single_via_range(self):
        assert _parse_seeds("42-42") == [42]

    def test_single_via_list(self):
        assert _parse_seeds("7") == [7]

    def test_range_length(self):
        assert len(_parse_seeds("0-99")) == 100

    def test_range_inclusive(self):
        result = _parse_seeds("5-8")
        assert result[0] == 5 and result[-1] == 8


# ===========================================================================
# evaluate — métriques de base (DQN)
# ===========================================================================

class TestEvaluateDQN:
    def test_returns_required_keys(self, tmp_path):
        _save_film(tmp_path / "film.pt")
        result = evaluate(tmp_path / "film.pt", seeds=[0, 1, 2])
        for key in ("n_seeds", "n_episodes", "total_episodes",
                    "wins", "win_rate", "score_mean_all", "score_mean_wins"):
            assert key in result

    def test_win_rate_in_range(self, tmp_path):
        _save_film(tmp_path / "film.pt")
        result = evaluate(tmp_path / "film.pt", seeds=[0, 1, 2])
        assert 0.0 <= result["win_rate"] <= 100.0

    def test_total_episodes_equals_n_seeds(self, tmp_path):
        _save_film(tmp_path / "film.pt")
        result = evaluate(tmp_path / "film.pt", seeds=[0, 1, 2, 3])
        assert result["total_episodes"] == 4
        assert result["n_seeds"] == 4
        assert result["n_episodes"] == 1

    def test_wins_le_total_episodes(self, tmp_path):
        _save_film(tmp_path / "film.pt")
        result = evaluate(tmp_path / "film.pt", seeds=list(range(5)))
        assert result["wins"] <= result["total_episodes"]

    def test_score_mean_all_in_range(self, tmp_path):
        _save_film(tmp_path / "film.pt")
        result = evaluate(tmp_path / "film.pt", seeds=[0, 1, 2])
        assert 0.0 <= result["score_mean_all"] <= 100.0

    def test_score_mean_wins_zero_when_no_wins(self, tmp_path):
        _save_film(tmp_path / "film.pt")
        result = evaluate(tmp_path / "film.pt", seeds=[0])
        if result["wins"] == 0:
            assert result["score_mean_wins"] == 0.0

    def test_obs_network_accepted(self, tmp_path):
        net = ObsDQNetwork()
        p = tmp_path / "obs.pt"
        torch.save(net.state_dict(), p)
        result = evaluate(p, seeds=[0, 1])
        assert result["total_episodes"] == 2


# ===========================================================================
# evaluate — n_episodes > 1
# ===========================================================================

class TestEvaluateNEpisodes:
    def test_total_episodes_is_n_seeds_times_n_episodes(self, tmp_path):
        _save_film(tmp_path / "film.pt")
        result = evaluate(tmp_path / "film.pt", seeds=[0, 1, 2], n_episodes=3)
        assert result["total_episodes"] == 9
        assert result["n_seeds"] == 3
        assert result["n_episodes"] == 3

    def test_dqn_deterministic_n_episodes_consistent(self, tmp_path):
        """DQN est déterministe : win_rate avec n=1 et n=3 doit être cohérent (0% ou 100%)."""
        _save_film(tmp_path / "film.pt")
        p = tmp_path / "film.pt"
        r1 = evaluate(p, seeds=[42], n_episodes=1)
        r3 = evaluate(p, seeds=[42], n_episodes=3)
        # même résultat × 3 : win_rate identique
        assert r1["win_rate"] == r3["win_rate"]

    def test_ppo_n_episodes(self, tmp_path):
        path = _save_ppo(tmp_path)
        result = evaluate(path, seeds=[0, 1], n_episodes=2)
        assert result["total_episodes"] == 4
        assert result["n_episodes"] == 2

    def test_wins_le_total_episodes_n_episodes(self, tmp_path):
        _save_film(tmp_path / "film.pt")
        result = evaluate(tmp_path / "film.pt", seeds=[0, 1], n_episodes=4)
        assert result["wins"] <= result["total_episodes"]


# ===========================================================================
# evaluate — PPO (déterministe et stochastique)
# ===========================================================================

class TestEvaluatePPO:
    def test_returns_valid_metrics(self, tmp_path):
        path = _save_ppo(tmp_path)
        result = evaluate(path, seeds=[0, 1, 2])
        assert 0.0 <= result["win_rate"] <= 100.0
        assert result["total_episodes"] == 3

    def test_deterministic_same_result_twice(self, tmp_path):
        """PPO déterministe : deux évaluations identiques sur le même seed."""
        path = _save_ppo(tmp_path)
        r1 = evaluate(path, seeds=[0], n_episodes=1)
        r2 = evaluate(path, seeds=[0], n_episodes=1)
        assert r1["wins"] == r2["wins"]
        assert r1["score_mean_all"] == r2["score_mean_all"]

    def test_stochastic_runs_without_error(self, tmp_path):
        path = _save_ppo(tmp_path)
        result = evaluate(path, seeds=[0], n_episodes=2, stochastic=True)
        assert isinstance(result["win_rate"], float)
        assert result["total_episodes"] == 2

    def test_stochastic_vs_deterministic_same_keys(self, tmp_path):
        path = _save_ppo(tmp_path)
        r_det  = evaluate(path, seeds=[0], stochastic=False)
        r_sto  = evaluate(path, seeds=[0], stochastic=True)
        assert set(r_det.keys()) == set(r_sto.keys())
