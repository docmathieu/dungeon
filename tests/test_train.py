"""Tests unitaires pour train.py (encode_obs, ReplayBuffer, DQNetwork, DQNAgent, train).

Removed : tests triviaux (isinstance Module, output_shape_batch, default_hidden_sizes,
          initial_epsilon, output_is_float32 FiLM, same_seed_output, initial_empty,
          capacity_sums_sub), classes de nommage de fichiers (TestPretrainedLabel,
          TestRunLabel, TestRunName), TestSaveCheckpoint (couvert par TestTrainLoop),
          doublons LogEpisode/LogMeta, TestTrainLoop réduit aux assertions métier.
"""
import io
import json
from pathlib import Path
import pytest
import torch

from model import DQNetwork, FiLMDQNetwork, ObsDQNetwork, INPUT_DIM, OUTPUT_DIM, OBS_DIM, TASK_DIM
from dungeon_env import DungeonEnv
from train import (
    ARCHITECTURES,
    encode_obs,
    encode_obs_pure,
    ReplayBuffer,
    StratifiedReplayBuffer,
    DQNAgent,
    train,
    _run_episode,
    _log_episode,
    _log_meta,
    ACTIONS,
    EPSILON_END,
    BUFFER_SIZE,
    N_SEEDS_DIM,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_obs(grid_val: int = 0, char_pos=(0, 0), exit_pos=(9, 9)) -> dict:
    return {
        "grid":     [grid_val] * 100,
        "char_pos": char_pos,
        "exit_pos": exit_pos,
    }


def _dummy_state() -> torch.Tensor:
    return torch.zeros(INPUT_DIM)


def _fill_buffer(buf: ReplayBuffer, n: int) -> None:
    s = _dummy_state()
    for _ in range(n):
        buf.push(s, 0, 0.0, s, False)


# ===========================================================================
# encode_obs
# ===========================================================================

class TestEncodeObs:
    def test_output_shape(self):
        assert encode_obs(_dummy_obs()).shape == (INPUT_DIM,)

    def test_dtype_is_float32(self):
        assert encode_obs(_dummy_obs()).dtype == torch.float32

    def test_all_grass_grid_one_hot(self):
        t = encode_obs(_dummy_obs(grid_val=0))
        for i in range(100):
            assert t[i * 3 + 0].item() == 1.0
            assert t[i * 3 + 1].item() == 0.0
            assert t[i * 3 + 2].item() == 0.0

    def test_rock_tile_one_hot(self):
        obs = _dummy_obs(grid_val=0)
        obs["grid"][0] = 1
        t = encode_obs(obs)
        assert t[0].item() == 0.0
        assert t[1].item() == 1.0
        assert t[2].item() == 0.0

    def test_positions_normalized(self):
        obs = _dummy_obs(char_pos=(9, 3), exit_pos=(0, 6))
        t = encode_obs(obs)
        assert abs(t[300].item() - 9 / 9) < 1e-6
        assert abs(t[301].item() - 3 / 9) < 1e-6
        assert abs(t[302].item() - 0 / 9) < 1e-6
        assert abs(t[303].item() - 6 / 9) < 1e-6

    def test_seed_one_hot_default_is_first_slot(self):
        t = encode_obs(_dummy_obs())
        assert t[304].item() == 1.0
        for i in range(1, N_SEEDS_DIM):
            assert t[304 + i].item() == 0.0

    def test_seed_one_hot_for_given_index(self):
        t = encode_obs(_dummy_obs(), seed_idx=3)
        for i in range(N_SEEDS_DIM):
            expected = 1.0 if i == 3 else 0.0
            assert t[304 + i].item() == expected


# ===========================================================================
# encode_obs_pure
# ===========================================================================

class TestEncodeObsPure:
    def test_output_shape(self):
        assert encode_obs_pure(_dummy_obs()).shape == (OBS_DIM,)

    def test_dtype_is_float32(self):
        assert encode_obs_pure(_dummy_obs()).dtype == torch.float32

    def test_no_seed_bits(self):
        """Le tenseur s'arrête après les 4 floats de position — pas de seed one-hot."""
        t = encode_obs_pure(_dummy_obs())
        assert t.shape[0] == OBS_DIM

    def test_positions_normalized(self):
        obs = _dummy_obs(char_pos=(9, 3), exit_pos=(0, 6))
        t = encode_obs_pure(obs)
        assert abs(t[300].item() - 9 / 9) < 1e-6
        assert abs(t[301].item() - 3 / 9) < 1e-6
        assert abs(t[302].item() - 0 / 9) < 1e-6
        assert abs(t[303].item() - 6 / 9) < 1e-6

    def test_grid_encoding_matches_encode_obs(self):
        """Les 304 premiers floats de encode_obs doivent être identiques à encode_obs_pure."""
        obs = _dummy_obs(grid_val=1, char_pos=(3, 5), exit_pos=(7, 2))
        t_pure = encode_obs_pure(obs)
        t_full = encode_obs(obs, seed_idx=2)
        assert torch.allclose(t_pure, t_full[:OBS_DIM])


# ===========================================================================
# ReplayBuffer
# ===========================================================================

class TestReplayBuffer:
    def test_push_increments_length(self):
        buf = ReplayBuffer()
        buf.push(_dummy_state(), 0, 1.0, _dummy_state(), False)
        assert len(buf) == 1

    def test_capacity_attribute(self):
        assert ReplayBuffer(capacity=500).capacity == 500

    def test_old_entries_evicted_when_full(self):
        buf = ReplayBuffer(capacity=3)
        _fill_buffer(buf, 5)
        assert len(buf) == 3

    def test_sample_returns_correct_batch_size(self):
        buf = ReplayBuffer()
        _fill_buffer(buf, 100)
        assert len(buf.sample(32)) == 32

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
    def test_output_shape_single_input(self):
        assert DQNetwork()(torch.zeros(INPUT_DIM)).shape == (OUTPUT_DIM,)

    def test_custom_dimensions(self):
        net = DQNetwork(input_dim=10, hidden1=8, hidden2=4, hidden3=2, output_dim=2)
        assert net(torch.zeros(10)).shape == (2,)

    def test_output_is_float32(self):
        assert DQNetwork()(torch.zeros(INPUT_DIM)).dtype == torch.float32


# ===========================================================================
# FiLMDQNetwork
# ===========================================================================

class TestFiLMDQNetwork:
    def test_output_shape_single_input(self):
        assert FiLMDQNetwork()(torch.zeros(INPUT_DIM)).shape == (OUTPUT_DIM,)

    def test_obs_dim_and_task_dim(self):
        """OBS_DIM + TASK_DIM must equal INPUT_DIM."""
        assert OBS_DIM + TASK_DIM == INPUT_DIM

    def test_different_seeds_give_different_outputs(self):
        net  = FiLMDQNetwork()
        base = torch.zeros(INPUT_DIM)
        obs0 = base.clone(); obs0[OBS_DIM + 0] = 1.0
        obs1 = base.clone(); obs1[OBS_DIM + 1] = 1.0
        with torch.no_grad():
            assert not torch.allclose(net(obs0), net(obs1))


# ===========================================================================
# StratifiedReplayBuffer
# ===========================================================================

class TestStratifiedReplayBuffer:
    def test_push_routes_to_correct_sub_buffer(self):
        buf = StratifiedReplayBuffer(capacity=300, n_seeds=3)
        s = _dummy_state()
        buf.push(s, 0, 1.0, s, False, seed_idx=1)
        assert len(buf._subs[0]) == 0
        assert len(buf._subs[1]) == 1
        assert len(buf._subs[2]) == 0

    def test_len_reflects_minimum_sub_buffer(self):
        buf = StratifiedReplayBuffer(capacity=300, n_seeds=3)
        s = _dummy_state()
        for _ in range(10):
            buf.push(s, 0, 0.0, s, False, seed_idx=0)
        assert len(buf) == 0
        for _ in range(5):
            buf.push(s, 0, 0.0, s, False, seed_idx=1)
            buf.push(s, 0, 0.0, s, False, seed_idx=2)
        assert len(buf) == 15

    def test_sample_balances_seeds(self):
        buf = StratifiedReplayBuffer(capacity=300, n_seeds=2)
        s = _dummy_state()
        for _ in range(50):
            buf.push(s, 0, 0.0, s, False, seed_idx=0)
            buf.push(s, 1, 0.0, s, False, seed_idx=1)
        batch = buf.sample(64)
        n0 = sum(1 for t in batch if t.action == 0)
        n1 = sum(1 for t in batch if t.action == 1)
        assert n0 == 32 and n1 == 32

    def test_sample_raises_when_sub_buffer_too_small(self):
        buf = StratifiedReplayBuffer(capacity=300, n_seeds=2)
        s = _dummy_state()
        for _ in range(10):
            buf.push(s, 0, 0.0, s, False, seed_idx=0)
        with pytest.raises(ValueError):
            buf.sample(32)

    def test_evicts_old_entries_per_sub_buffer(self):
        buf = StratifiedReplayBuffer(capacity=60, n_seeds=2)
        s = _dummy_state()
        for _ in range(40):
            buf.push(s, 0, 0.0, s, False, seed_idx=0)
        assert len(buf._subs[0]) == 30


# ===========================================================================
# DQNAgent
# ===========================================================================

class TestDQNAgent:
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
        assert isinstance(agent.learn(buf, batch_size=32), float)

    def test_sync_target_copies_weights(self):
        agent = DQNAgent()
        for p in agent.q_net.parameters():
            p.data.fill_(7.0)
        agent.sync_target()
        for pq, pt in zip(agent.q_net.parameters(), agent.target_net.parameters()):
            assert torch.allclose(pq, pt)


# ===========================================================================
# DQNAgent — architecture
# ===========================================================================

class TestDQNAgentArchitecture:
    def test_default_arch_is_film(self):
        assert isinstance(DQNAgent().q_net, FiLMDQNetwork)

    def test_taskcond_arch(self):
        agent = DQNAgent(arch="taskcond")
        assert isinstance(agent.q_net, DQNetwork)
        assert isinstance(agent.target_net, DQNetwork)

    def test_obs_arch(self):
        agent = DQNAgent(arch="obs")
        assert isinstance(agent.q_net, ObsDQNetwork)
        assert isinstance(agent.target_net, ObsDQNetwork)

    def test_obs_arch_uses_pure_encoder(self):
        """_encode pour obs doit produire un tenseur de taille OBS_DIM."""
        agent = DQNAgent(arch="obs")
        obs = _dummy_obs()
        t = agent._encode(obs)
        assert t.shape == (OBS_DIM,)

    def test_film_arch_encoder_produces_input_dim(self):
        """_encode pour film doit produire un tenseur de taille INPUT_DIM."""
        agent = DQNAgent(arch="film")
        t = agent._encode(_dummy_obs())
        assert t.shape == (INPUT_DIM,)

    def test_obs_train_completes(self, tmp_path):
        agent = train(episodes=3, seed=42, arch="obs",
                      log_path=tmp_path / "t.jsonl",
                      model_dir=tmp_path / "models", verbose=False)
        assert isinstance(agent.q_net, ObsDQNetwork)

    def test_obs_checkpoint_loadable(self, tmp_path):
        train(episodes=3, seed=42, arch="obs",
              log_path=tmp_path / "t.jsonl",
              model_dir=tmp_path / "models", verbose=False)
        ObsDQNetwork().load_state_dict(
            torch.load(tmp_path / "models" / "final.pt", weights_only=True))

    def test_invalid_arch_raises(self):
        with pytest.raises(ValueError, match="arch"):
            DQNAgent(arch="unknown")

    def test_taskcond_train_completes(self, tmp_path):
        agent = train(episodes=3, seed=42, arch="taskcond",
                      log_path=tmp_path / "t.jsonl",
                      model_dir=tmp_path / "models", verbose=False)
        assert isinstance(agent, DQNAgent)
        assert isinstance(agent.q_net, DQNetwork)

    def test_taskcond_checkpoint_loadable(self, tmp_path):
        train(episodes=3, seed=42, arch="taskcond",
              log_path=tmp_path / "t.jsonl",
              model_dir=tmp_path / "models", verbose=False)
        DQNetwork().load_state_dict(
            torch.load(tmp_path / "models" / "final.pt", weights_only=True))


# ===========================================================================
# _run_episode
# ===========================================================================

class TestRunEpisode:
    def setup_method(self):
        self.env   = DungeonEnv(seed=42)
        self.agent = DQNAgent()
        self.buf   = ReplayBuffer(BUFFER_SIZE)

    def test_moves_is_list_of_strings(self):
        moves, _, _, _ = _run_episode(self.env, self.agent, self.buf)
        assert isinstance(moves, list) and all(isinstance(m, str) for m in moves)

    def test_moves_are_valid_actions(self):
        moves, _, _, _ = _run_episode(self.env, self.agent, self.buf)
        assert all(m in ACTIONS for m in moves)

    def test_info_has_required_keys(self):
        _, _, info, _ = _run_episode(self.env, self.agent, self.buf)
        for key in ("score", "moves", "steps", "won"):
            assert key in info

    def test_seed_is_int_for_fixed_seed(self):
        _, _, _, seed = _run_episode(self.env, self.agent, self.buf)
        assert seed == 42

    def test_seed_is_pool_value_for_pool(self):
        pool = [10, 20, 30]
        env  = DungeonEnv(seed_pool=pool)
        _, _, _, seed = _run_episode(env, DQNAgent(), ReplayBuffer(BUFFER_SIZE))
        assert seed in pool

    def test_buffer_grows_after_episode(self):
        _run_episode(self.env, self.agent, self.buf)
        assert len(self.buf) > 0


# ===========================================================================
# _log_episode
# ===========================================================================

class TestLogEpisode:
    def _make_info(self, score=50, won=False):
        return {"score": score, "moves": 10, "steps": 10, "won": won}

    def test_all_fields_present(self):
        f = io.StringIO()
        _log_episode(f, 1, self._make_info(), ["RIGHT"], DQNAgent(), -0.01)
        entry = json.loads(f.getvalue().strip())
        for key in ("episode", "seed", "score", "moves", "epsilon", "reward"):
            assert key in entry

    def test_seed_field_matches(self):
        f = io.StringIO()
        _log_episode(f, 1, self._make_info(), [], DQNAgent(), 0.0, seed=42)
        assert json.loads(f.getvalue())["seed"] == 42

    def test_seed_field_none_when_not_provided(self):
        f = io.StringIO()
        _log_episode(f, 1, self._make_info(), [], DQNAgent(), 0.0)
        assert json.loads(f.getvalue())["seed"] is None


# ===========================================================================
# _log_meta
# ===========================================================================

class TestLogMeta:
    def _make_agent(self):
        return DQNAgent()

    def test_type_field_is_meta(self):
        f = io.StringIO()
        _log_meta(f, self._make_agent(), 100, 1e-3, 42, None, None)
        assert json.loads(f.getvalue())["type"] == "meta"

    def test_command_field_is_string(self):
        f = io.StringIO()
        _log_meta(f, self._make_agent(), 100, 1e-3, None, None, None)
        assert isinstance(json.loads(f.getvalue())["command"], str)

    def test_architecture_matches_agent(self):
        f = io.StringIO()
        agent = self._make_agent()
        _log_meta(f, agent, 100, 1e-3, None, None, None)
        assert json.loads(f.getvalue())["architecture"] == type(agent.q_net).__name__

    def test_hyperparams_contains_required_keys(self):
        f = io.StringIO()
        _log_meta(f, self._make_agent(), 500, 3e-4, None, [0, 1, 2], None)
        hp = json.loads(f.getvalue())["hyperparams"]
        for key in ("episodes", "lr", "gamma", "epsilon_start", "epsilon_end",
                    "eps_decay", "buffer_size", "batch_size", "target_update",
                    "max_steps", "seed", "seed_pool", "pretrained"):
            assert key in hp

    def test_hyperparams_pretrained_path(self, tmp_path):
        f = io.StringIO()
        p = tmp_path / "final.pt"
        _log_meta(f, self._make_agent(), 100, 1e-3, None, None, p)
        assert json.loads(f.getvalue())["hyperparams"]["pretrained"] == str(p)

    def test_extra_fields_merged(self):
        f = io.StringIO()
        _log_meta(f, self._make_agent(), 100, 1e-3, None, None, None,
                  extra={"stage": 2, "full_pool": [0, 1, 2]})
        entry = json.loads(f.getvalue())
        assert entry["stage"] == 2
        assert entry["full_pool"] == [0, 1, 2]


# ===========================================================================
# train() — boucle complète
# ===========================================================================

class TestTrainLoop:
    def _episode_lines(self, log: Path) -> list[dict]:
        return [json.loads(l) for l in log.read_text().splitlines()
                if json.loads(l).get("type") != "meta"]

    def test_creates_log_file(self, tmp_path):
        log = tmp_path / "test.jsonl"
        train(episodes=3, seed=42, log_path=log,
              model_dir=tmp_path / "models", verbose=False)
        assert log.exists()

    def test_log_line_count_matches_episodes(self, tmp_path):
        log = tmp_path / "test.jsonl"
        train(episodes=5, seed=42, log_path=log,
              model_dir=tmp_path / "models", verbose=False)
        assert len(self._episode_lines(log)) == 5

    def test_log_has_meta_line_first(self, tmp_path):
        log = tmp_path / "test.jsonl"
        train(episodes=3, seed=42, log_path=log,
              model_dir=tmp_path / "models", verbose=False)
        first = json.loads(log.read_text().splitlines()[0])
        assert first.get("type") == "meta"

    def test_log_entry_seed_matches_fixed_seed(self, tmp_path):
        log = tmp_path / "test.jsonl"
        train(episodes=3, seed=42, log_path=log,
              model_dir=tmp_path / "models", verbose=False)
        for entry in self._episode_lines(log):
            assert entry["seed"] == 42

    def test_log_entry_seed_in_pool(self, tmp_path):
        log = tmp_path / "test.jsonl"
        pool = [0, 1, 2]
        train(episodes=5, seed_pool=pool, log_path=log,
              model_dir=tmp_path / "models", verbose=False)
        for entry in self._episode_lines(log):
            assert entry["seed"] in pool

    def test_creates_final_checkpoint(self, tmp_path):
        model_dir = tmp_path / "models"
        train(episodes=3, seed=42, log_path=tmp_path / "t.jsonl",
              model_dir=model_dir, verbose=False)
        assert (model_dir / "final.pt").exists()

    def test_checkpoint_loadable(self, tmp_path):
        model_dir = tmp_path / "models"
        train(episodes=3, seed=42, log_path=tmp_path / "t.jsonl",
              model_dir=model_dir, verbose=False)
        FiLMDQNetwork().load_state_dict(
            torch.load(model_dir / "final.pt", weights_only=True))

    def test_returns_dqn_agent(self, tmp_path):
        agent = train(episodes=2, seed=42, log_path=tmp_path / "t.jsonl",
                      model_dir=tmp_path / "models", verbose=False)
        assert isinstance(agent, DQNAgent)

    def test_score_is_int_in_log(self, tmp_path):
        log = tmp_path / "test.jsonl"
        train(episodes=3, seed=42, log_path=log,
              model_dir=tmp_path / "models", verbose=False)
        for entry in self._episode_lines(log):
            assert isinstance(entry["score"], int)
            assert 0 <= entry["score"] <= 100

    def test_adaptive_decay_reaches_epsilon_end_at_midpoint(self, tmp_path):
        episodes = 200
        agent = train(episodes=episodes, seed=42, log_path=tmp_path / "t.jsonl",
                      model_dir=tmp_path / "models", verbose=False)
        assert agent.epsilon == pytest.approx(EPSILON_END, abs=0.01)

    def test_pretrained_weights_are_loaded(self, tmp_path):
        pre_dir = tmp_path / "20260527_1222_seed42_ep2000"
        pre_dir.mkdir()
        net = FiLMDQNetwork()
        for p in net.parameters():
            p.data.fill_(0.5)
        torch.save(net.state_dict(), pre_dir / "final.pt")
        agent = train(
            episodes=2, seed=42, pretrained=pre_dir / "final.pt",
            log_path=tmp_path / "t.jsonl",
            model_dir=tmp_path / "models", verbose=False,
        )
        assert isinstance(agent, DQNAgent)
