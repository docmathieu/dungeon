"""Tests unitaires pour train.py (encode_obs, ReplayBuffer, DQNetwork, DQNAgent, train)."""

import json
import pytest
import torch

from model import DQNetwork, INPUT_DIM, OUTPUT_DIM
from train import (
    encode_obs,
    ReplayBuffer,
    DQNAgent,
    train,
    ACTIONS,
    EPSILON_START,
    EPSILON_END,
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
        assert (model_dir / "dqn_final.pt").exists()

    def test_checkpoint_loadable(self, tmp_path):
        """Le checkpoint final doit pouvoir être rechargé dans un DQNetwork."""
        model_dir = tmp_path / "models"
        train(episodes=3, seed=42, log_path=tmp_path / "t.jsonl",
              model_dir=model_dir, verbose=False)
        net = DQNetwork()
        net.load_state_dict(torch.load(model_dir / "dqn_final.pt",
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
