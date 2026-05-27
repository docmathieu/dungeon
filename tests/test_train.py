"""Tests unitaires pour train.py (encode_obs, ReplayBuffer, DQNetwork, DQNAgent, train)."""

import io
import json
from pathlib import Path
import pytest
import torch

from model import DQNetwork, INPUT_DIM, OUTPUT_DIM
from dungeon_env import DungeonEnv
from train import (
    encode_obs,
    ReplayBuffer,
    DQNAgent,
    train,
    _run_episode,
    _log_episode,
    _pretrained_label,
    _run_label,
    _run_name,
    _save_checkpoint,
    ACTIONS,
    EPSILON_START,
    EPSILON_END,
    BUFFER_SIZE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_obs(grid_val: int = 0, char_pos=(0, 0), exit_pos=(9, 9)) -> dict:
    """Observation minimale avec une grille uniforme."""
    return {
        "grid":     [grid_val] * 100,
        "char_pos": char_pos,
        "exit_pos": exit_pos,
    }


def _dummy_state() -> torch.Tensor:
    return torch.zeros(INPUT_DIM)


def _fill_buffer(buf: ReplayBuffer, n: int) -> None:
    """Remplit le buffer avec n transitions factices."""
    s = _dummy_state()
    for _ in range(n):
        buf.push(s, 0, 0.0, s, False)


# ===========================================================================
# encode_obs
# ===========================================================================

class TestEncodeObs:
    def test_output_is_tensor(self):
        t = encode_obs(_dummy_obs())
        assert isinstance(t, torch.Tensor)

    def test_output_shape(self):
        t = encode_obs(_dummy_obs())
        assert t.shape == (INPUT_DIM,)  # 304

    def test_dtype_is_float32(self):
        t = encode_obs(_dummy_obs())
        assert t.dtype == torch.float32

    def test_all_grass_grid_one_hot(self):
        """Grille tout-herbe : slots GRASS (index *3+0) = 1.0, autres = 0.0."""
        t = encode_obs(_dummy_obs(grid_val=0))
        for i in range(100):
            assert t[i * 3 + 0].item() == 1.0   # GRASS actif
            assert t[i * 3 + 1].item() == 0.0   # ROCK inactif
            assert t[i * 3 + 2].item() == 0.0   # WATER inactif

    def test_rock_tile_one_hot(self):
        """Première case ROCHE : one_hot[1] == 1.0, [0] et [2] == 0.0."""
        obs = _dummy_obs(grid_val=0)
        obs["grid"][0] = 1   # première case = ROCHE
        t = encode_obs(obs)
        assert t[0].item() == 0.0   # GRASS slot
        assert t[1].item() == 1.0   # ROCK slot
        assert t[2].item() == 0.0   # WATER slot

    def test_water_tile_one_hot(self):
        """Première case EAU : one_hot[2] == 1.0."""
        obs = _dummy_obs(grid_val=0)
        obs["grid"][0] = 2
        t = encode_obs(obs)
        assert t[2].item() == 1.0

    def test_positions_normalized(self):
        """char_pos et exit_pos sont normalisés dans [0, 1] aux indices 300-303."""
        obs = _dummy_obs(char_pos=(9, 3), exit_pos=(0, 6))
        t = encode_obs(obs)
        assert abs(t[300].item() - 9 / 9) < 1e-6   # cx
        assert abs(t[301].item() - 3 / 9) < 1e-6   # cy
        assert abs(t[302].item() - 0 / 9) < 1e-6   # ex
        assert abs(t[303].item() - 6 / 9) < 1e-6   # ey

    def test_origin_position_normalized_to_zero(self):
        obs = _dummy_obs(char_pos=(0, 0), exit_pos=(0, 0))
        t = encode_obs(obs)
        assert t[300].item() == 0.0
        assert t[301].item() == 0.0


# ===========================================================================
# ReplayBuffer
# ===========================================================================

class TestReplayBuffer:
    def test_initial_empty(self):
        assert len(ReplayBuffer()) == 0

    def test_push_increments_length(self):
        buf = ReplayBuffer()
        s = _dummy_state()
        buf.push(s, 0, 1.0, s, False)
        assert len(buf) == 1

    def test_capacity_attribute(self):
        buf = ReplayBuffer(capacity=500)
        assert buf.capacity == 500

    def test_old_entries_evicted_when_full(self):
        buf = ReplayBuffer(capacity=3)
        _fill_buffer(buf, 5)
        assert len(buf) == 3   # jamais plus que la capacité

    def test_sample_returns_correct_batch_size(self):
        buf = ReplayBuffer()
        _fill_buffer(buf, 100)
        batch = buf.sample(32)
        assert len(batch) == 32

    def test_sample_raises_when_buffer_too_small(self):
        buf = ReplayBuffer()
        _fill_buffer(buf, 2)
        with pytest.raises(ValueError):
            buf.sample(10)

    def test_transition_fields(self):
        buf = ReplayBuffer()
        s = _dummy_state()
        buf.push(s, 2, 0.5, s, True)
        t = buf.sample(1)[0]
        assert t.action == 2
        assert t.reward == 0.5
        assert t.done is True


# ===========================================================================
# DQNetwork
# ===========================================================================

class TestDQNetwork:
    def test_is_nn_module(self):
        assert isinstance(DQNetwork(), torch.nn.Module)

    def test_output_shape_single_input(self):
        net = DQNetwork()
        x   = torch.zeros(INPUT_DIM)
        out = net(x)
        assert out.shape == (OUTPUT_DIM,)

    def test_output_shape_batch(self):
        net   = DQNetwork()
        batch = torch.zeros(8, INPUT_DIM)
        out   = net(batch)
        assert out.shape == (8, OUTPUT_DIM)

    def test_custom_dimensions(self):
        net = DQNetwork(input_dim=10, hidden1=8, hidden2=4, output_dim=2)
        x   = torch.zeros(10)
        assert net(x).shape == (2,)

    def test_output_is_float32(self):
        net = DQNetwork()
        out = net(torch.zeros(INPUT_DIM))
        assert out.dtype == torch.float32


# ===========================================================================
# DQNAgent
# ===========================================================================

class TestDQNAgent:
    def test_initial_epsilon(self):
        assert DQNAgent().epsilon == EPSILON_START

    def test_decay_reduces_epsilon(self):
        agent = DQNAgent()
        before = agent.epsilon
        agent.decay_epsilon()
        assert agent.epsilon < before

    def test_decay_stops_at_minimum(self):
        agent = DQNAgent(eps_end=0.1, eps_decay=0.5)
        for _ in range(100):
            agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.1)

    def test_select_action_in_valid_range(self):
        agent = DQNAgent()
        s = _dummy_state()
        for _ in range(20):
            assert 0 <= agent.select_action(s) < len(ACTIONS)

    def test_learn_returns_none_when_buffer_too_small(self):
        agent = DQNAgent()
        buf   = ReplayBuffer()
        _fill_buffer(buf, 10)
        assert agent.learn(buf, batch_size=64) is None

    def test_learn_returns_float_when_buffer_sufficient(self):
        agent = DQNAgent()
        buf   = ReplayBuffer()
        _fill_buffer(buf, 100)
        loss = agent.learn(buf, batch_size=32)
        assert isinstance(loss, float)

    def test_sync_target_copies_weights(self):
        agent = DQNAgent()
        # Modifier les poids du réseau principal
        for p in agent.q_net.parameters():
            p.data.fill_(7.0)
        # Avant sync, les poids cibles sont différents
        agent.sync_target()
        for pq, pt in zip(agent.q_net.parameters(), agent.target_net.parameters()):
            assert torch.allclose(pq, pt)


# ===========================================================================
# _run_label / _run_name
# ===========================================================================

class TestPretrainedLabel:
    def test_none_returns_empty(self):
        assert _pretrained_label(None) == ""

    def test_extracts_timestamp_from_seed_path(self):
        p = Path("models/20260527_1222_seed42_ep2000/final.pt")
        assert _pretrained_label(p) == "20260527_1222"

    def test_extracts_timestamp_from_pool_path(self):
        p = Path("models/20260527_1430_pool10_ep3000/ep500.pt")
        assert _pretrained_label(p) == "20260527_1430"

    def test_length_is_13(self):
        p = Path("models/20260527_1222_seed42_ep2000/final.pt")
        assert len(_pretrained_label(p)) == 13


class TestRunLabel:
    def test_seed_label(self):
        assert _run_label(42, None) == "seed42"

    def test_pool_label_uses_length(self):
        assert _run_label(None, list(range(10))) == "pool10"

    def test_random_label(self):
        assert _run_label(None, None) == "random"

    def test_seed_takes_priority_over_none_pool(self):
        assert _run_label(7, None) == "seed7"


class TestRunName:
    def test_seed_format(self):
        assert _run_name("20260527_1430", 3000, 42, None) == "20260527_1430_seed42_ep3000"

    def test_pool_format(self):
        assert _run_name("20260527_1430", 10000, None, list(range(10))) == "20260527_1430_pool10_ep10000"

    def test_random_format(self):
        assert _run_name("20260527_1430", 5000, None, None) == "20260527_1430_random_ep5000"

    def test_contains_timestamp(self):
        name = _run_name("20260527_1430", 100, None, None)
        assert name.startswith("20260527_1430_")

    def test_contains_episodes(self):
        name = _run_name("20260527_1430", 1234, 0, None)
        assert name.endswith("_ep1234")

    def test_with_pretrained_label(self):
        name = _run_name("20260527_1300", 2000, None, list(range(10)), "20260527_1222")
        assert name == "20260527_1300_pool10_ep2000_from_20260527_1222"

    def test_without_pretrained_no_suffix(self):
        name = _run_name("20260527_1300", 2000, None, list(range(10)), "")
        assert name == "20260527_1300_pool10_ep2000"


# ===========================================================================
# _run_episode
# ===========================================================================

class TestRunEpisode:
    def setup_method(self):
        self.env   = DungeonEnv(seed=42)
        self.agent = DQNAgent()
        self.buf   = ReplayBuffer(BUFFER_SIZE)

    def test_returns_tuple_of_three(self):
        result = _run_episode(self.env, self.agent, self.buf)
        assert isinstance(result, tuple) and len(result) == 3

    def test_moves_is_list_of_strings(self):
        moves, _, _ = _run_episode(self.env, self.agent, self.buf)
        assert isinstance(moves, list)
        assert all(isinstance(m, str) for m in moves)

    def test_moves_are_valid_actions(self):
        moves, _, _ = _run_episode(self.env, self.agent, self.buf)
        assert all(m in ACTIONS for m in moves)

    def test_ep_reward_is_float(self):
        _, ep_reward, _ = _run_episode(self.env, self.agent, self.buf)
        assert isinstance(ep_reward, float)

    def test_info_has_required_keys(self):
        _, _, info = _run_episode(self.env, self.agent, self.buf)
        for key in ("score", "moves", "steps", "won"):
            assert key in info

    def test_buffer_grows_after_episode(self):
        _run_episode(self.env, self.agent, self.buf)
        assert len(self.buf) > 0

    def test_moves_not_empty(self):
        moves, _, _ = _run_episode(self.env, self.agent, self.buf)
        assert len(moves) > 0


# ===========================================================================
# _log_episode
# ===========================================================================

class TestLogEpisode:
    def _make_info(self, score=50, won=False):
        return {"score": score, "moves": 10, "steps": 10, "won": won}

    def test_writes_valid_json(self):
        f = io.StringIO()
        agent = DQNAgent()
        _log_episode(f, 1, self._make_info(), ["RIGHT", "DOWN"], agent, -0.05)
        entry = json.loads(f.getvalue().strip())
        assert isinstance(entry, dict)

    def test_all_fields_present(self):
        f = io.StringIO()
        agent = DQNAgent()
        _log_episode(f, 1, self._make_info(), ["RIGHT"], agent, -0.01)
        entry = json.loads(f.getvalue().strip())
        for key in ("episode", "score", "moves", "epsilon", "reward"):
            assert key in entry

    def test_episode_field_matches(self):
        f = io.StringIO()
        _log_episode(f, 7, self._make_info(), [], DQNAgent(), 0.0)
        assert json.loads(f.getvalue())["episode"] == 7

    def test_score_field_matches(self):
        f = io.StringIO()
        _log_episode(f, 1, self._make_info(score=83), [], DQNAgent(), 0.0)
        assert json.loads(f.getvalue())["score"] == 83

    def test_epsilon_is_rounded(self):
        f = io.StringIO()
        agent = DQNAgent()
        _log_episode(f, 1, self._make_info(), [], agent, 0.0)
        eps = json.loads(f.getvalue())["epsilon"]
        assert eps == round(agent.epsilon, 4)


# ===========================================================================
# _save_checkpoint
# ===========================================================================

class TestSaveCheckpoint:
    def test_creates_periodic_file(self, tmp_path):
        agent = DQNAgent()
        _save_checkpoint(agent, tmp_path, ep=500, episodes=3000,
                         score=80, t0=0.0, verbose=False)
        assert (tmp_path / "ep500.pt").exists()

    def test_creates_final_file(self, tmp_path):
        agent = DQNAgent()
        _save_checkpoint(agent, tmp_path, ep=3000, episodes=3000,
                         score=80, t0=0.0, verbose=False, final=True)
        assert (tmp_path / "final.pt").exists()

    def test_periodic_does_not_create_final(self, tmp_path):
        agent = DQNAgent()
        _save_checkpoint(agent, tmp_path, ep=500, episodes=3000,
                         score=80, t0=0.0, verbose=False)
        assert not (tmp_path / "final.pt").exists()

    def test_checkpoint_is_loadable(self, tmp_path):
        agent = DQNAgent()
        _save_checkpoint(agent, tmp_path, ep=500, episodes=3000,
                         score=80, t0=0.0, verbose=False)
        net = DQNetwork()
        net.load_state_dict(torch.load(tmp_path / "ep500.pt",
                                       weights_only=True))


# ===========================================================================
# train() — boucle complète (épisodes courts pour la rapidité)
# ===========================================================================

class TestTrainLoop:
    def test_creates_log_file(self, tmp_path):
        log = tmp_path / "test.jsonl"
        train(episodes=3, seed=42, log_path=log,
              model_dir=tmp_path / "models", verbose=False)
        assert log.exists()

    def test_log_line_count_matches_episodes(self, tmp_path):
        log = tmp_path / "test.jsonl"
        train(episodes=5, seed=42, log_path=log,
              model_dir=tmp_path / "models", verbose=False)
        lines = log.read_text().strip().splitlines()
        assert len(lines) == 5

    def test_log_entry_has_required_fields(self, tmp_path):
        log = tmp_path / "test.jsonl"
        train(episodes=2, seed=42, log_path=log,
              model_dir=tmp_path / "models", verbose=False)
        entry = json.loads(log.read_text().splitlines()[0])
        for key in ("episode", "score", "moves", "epsilon", "reward"):
            assert key in entry

    def test_log_episode_counter_increments(self, tmp_path):
        log = tmp_path / "test.jsonl"
        train(episodes=3, seed=42, log_path=log,
              model_dir=tmp_path / "models", verbose=False)
        episodes = [json.loads(l)["episode"] for l in log.read_text().splitlines()]
        assert episodes == [1, 2, 3]

    def test_creates_final_checkpoint(self, tmp_path):
        model_dir = tmp_path / "models"
        train(episodes=3, seed=42, log_path=tmp_path / "t.jsonl",
              model_dir=model_dir, verbose=False)
        assert (model_dir / "final.pt").exists()

    def test_checkpoint_loadable(self, tmp_path):
        """Le checkpoint final doit pouvoir être rechargé dans un DQNetwork."""
        model_dir = tmp_path / "models"
        train(episodes=3, seed=42, log_path=tmp_path / "t.jsonl",
              model_dir=model_dir, verbose=False)
        net = DQNetwork()
        net.load_state_dict(torch.load(model_dir / "final.pt",
                                       weights_only=True))

    def test_returns_dqn_agent(self, tmp_path):
        agent = train(episodes=2, seed=42, log_path=tmp_path / "t.jsonl",
                      model_dir=tmp_path / "models", verbose=False)
        assert isinstance(agent, DQNAgent)

    def test_score_is_int_in_log(self, tmp_path):
        log = tmp_path / "test.jsonl"
        train(episodes=3, seed=42, log_path=log,
              model_dir=tmp_path / "models", verbose=False)
        for line in log.read_text().splitlines():
            entry = json.loads(line)
            assert isinstance(entry["score"], int)
            assert 0 <= entry["score"] <= 100

    def test_adaptive_decay_reaches_epsilon_end_at_midpoint(self, tmp_path):
        """L'epsilon adaptatif doit atteindre EPSILON_END vers la moitié des épisodes."""
        episodes = 200
        agent = train(episodes=episodes, seed=42, log_path=tmp_path / "t.jsonl",
                      model_dir=tmp_path / "models", verbose=False)
        # Après 'episodes' décays, epsilon devrait être proche de EPSILON_END
        # (le decay est calculé pour atteindre EPSILON_END à episodes/2)
        assert agent.epsilon == pytest.approx(EPSILON_END, abs=0.01)

    def test_pretrained_weights_are_loaded(self, tmp_path):
        """Les poids du checkpoint pretrained doivent être chargés avant l'entraînement."""
        pre_dir = tmp_path / "20260527_1222_seed42_ep2000"
        pre_dir.mkdir()
        net = DQNetwork()
        for p in net.parameters():
            p.data.fill_(0.5)
        torch.save(net.state_dict(), pre_dir / "final.pt")

        agent = train(
            episodes   = 2,
            seed       = 42,
            pretrained = pre_dir / "final.pt",
            log_path   = tmp_path / "t.jsonl",
            model_dir  = tmp_path / "models",
            verbose    = False,
        )
        assert isinstance(agent, DQNAgent)

    def test_seed_pool_runs_without_error(self, tmp_path):
        """train() avec seed_pool doit compléter sans erreur."""
        log = tmp_path / "pool.jsonl"
        agent = train(episodes=10, seed_pool=[0, 1, 2], log_path=log,
                      model_dir=tmp_path / "models", verbose=False)
        assert isinstance(agent, DQNAgent)
        lines = log.read_text().strip().splitlines()
        assert len(lines) == 10
