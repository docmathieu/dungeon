"""Tests unitaires pour DungeonEnv (interface Gym — Phase 1 RL).

Removed : tests sur états internes avant reset (state_none, steps_zero, default_max_steps),
          doublons format obs (grid_length_is_100, grid_values_are_0_1_or_2, reset_returns_dict),
          test_actions_constant (doublon de action_space), TestCurrentSeedIdx (couvert par
          TestCurrentSeed), tests step internes (returns_four_elements, done_false_before_max_steps,
          done_true_after_exactly, info_steps_matches, step_increments).
"""
import pytest

from dungeon_env import DungeonEnv, ACTIONS, REWARD_STEP, REWARD_BUMP
from game_state import GameState
from grid import Grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(seed: int = 0, **kwargs) -> DungeonEnv:
    env = DungeonEnv(seed=seed, **kwargs)
    env.reset()
    return env


def moves_from_path(path: list[tuple[int, int]]) -> list[str]:
    delta_to_dir = {
        (-1,  0): "LEFT",
        ( 1,  0): "RIGHT",
        ( 0, -1): "UP",
        ( 0,  1): "DOWN",
    }
    return [
        delta_to_dir[(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])]
        for i in range(len(path) - 1)
    ]


# ===========================================================================
# DungeonEnv — initialisation
# ===========================================================================

class TestDungeonEnvInit:
    def test_custom_max_steps(self):
        assert DungeonEnv(max_steps=50)._max_steps == 50

    def test_empty_seed_pool_raises(self):
        with pytest.raises(ValueError, match="vide"):
            DungeonEnv(seed_pool=[])

    def test_action_space_contains_four_actions(self):
        env = DungeonEnv()
        assert set(env.action_space) == {"LEFT", "RIGHT", "UP", "DOWN"}


# ===========================================================================
# DungeonEnv — reset()
# ===========================================================================

class TestDungeonEnvReset:
    def test_obs_has_required_keys(self):
        obs = DungeonEnv(seed=0).reset()
        assert set(obs.keys()) == {"grid", "char_pos", "exit_pos"}

    def test_char_pos_in_bounds(self):
        obs = DungeonEnv(seed=0).reset()
        x, y = obs["char_pos"]
        assert 0 <= x < Grid.WIDTH and 0 <= y < Grid.HEIGHT

    def test_exit_pos_in_bounds(self):
        obs = DungeonEnv(seed=0).reset()
        x, y = obs["exit_pos"]
        assert 0 <= x < Grid.WIDTH and 0 <= y < Grid.HEIGHT

    def test_char_and_exit_differ(self):
        obs = DungeonEnv(seed=0).reset()
        assert obs["char_pos"] != obs["exit_pos"]

    def test_seeded_reset_is_deterministic(self):
        obs1 = DungeonEnv(seed=42).reset()
        obs2 = DungeonEnv(seed=42).reset()
        assert obs1 == obs2

    def test_different_seeds_produce_different_obs(self):
        obs1 = DungeonEnv(seed=0).reset()
        obs2 = DungeonEnv(seed=99).reset()
        assert obs1 != obs2

    def test_second_reset_reinitialises(self):
        env = DungeonEnv(seed=5)
        env.reset()
        env.step("UP")
        obs1 = env.reset()
        obs2 = DungeonEnv(seed=5).reset()
        assert obs1 == obs2

    def test_terrain_is_solvable_after_reset(self):
        for seed in range(10):
            env = DungeonEnv(seed=seed)
            env.reset()
            assert env._state.is_solvable()


# ===========================================================================
# DungeonEnv — step()
# ===========================================================================

class TestDungeonEnvStep:
    def test_step_before_reset_raises(self):
        with pytest.raises(RuntimeError):
            DungeonEnv().step("RIGHT")

    def test_obs_format_in_step(self):
        obs, *_ = make_env(seed=0).step("RIGHT")
        assert set(obs.keys()) == {"grid", "char_pos", "exit_pos"}
        assert len(obs["grid"]) == 100

    def test_reward_negative_on_non_terminal_step(self):
        env = make_env(seed=0)
        _, reward, done, _ = env.step("UP")
        if not done:
            assert reward < 0.0

    def test_done_true_at_max_steps(self):
        env = DungeonEnv(seed=0, max_steps=1)
        env.reset()
        _, _, done, _ = env.step("UP")
        assert done is True

    def test_info_contains_required_keys(self):
        _, _, _, info = make_env(seed=0).step("RIGHT")
        for key in ("score", "moves", "steps", "won"):
            assert key in info

    def test_reward_normal_step_is_reward_step(self):
        env = make_env(seed=0)
        direction = moves_from_path(env._state.optimal_path[:2])[0]
        _, reward, done, _ = env.step(direction)
        if not done:
            assert reward == pytest.approx(REWARD_STEP)

    def test_reward_boundary_bump_is_reward_bump(self):
        env = make_env(seed=0)
        env._state.char_pos = (0, 0)
        env._state.exit_pos = (9, 9)
        _, reward, _, _ = env.step("UP")
        assert reward == pytest.approx(REWARD_BUMP)

    def test_reward_rock_bump_is_reward_bump(self):
        from grid import TileType
        from helpers import FakeGrid
        env = make_env(seed=0)
        overrides = {(4, 5): TileType.ROCK}
        fake = FakeGrid(overrides)
        env._state = GameState(fake, seed=0)
        env._state.char_pos = (3, 5)
        env._state.exit_pos = (9, 9)
        _, reward, _, _ = env.step("RIGHT")
        assert reward == pytest.approx(REWARD_BUMP)


# ===========================================================================
# DungeonEnv — victoire et reward
# ===========================================================================

class TestDungeonEnvWin:
    def test_reward_one_on_optimal_win(self):
        env = make_env(seed=0)
        path   = env._state.optimal_path
        moves  = moves_from_path(path)
        reward = 0.0
        done   = False
        for m in moves:
            obs, reward, done, info = env.step(m)
        assert done is True
        assert reward == 1.0

    def test_reward_less_than_one_on_suboptimal_win(self):
        env = make_env(seed=0)
        path  = env._state.optimal_path
        moves = moves_from_path(path)
        opposite = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP"}
        first = moves[0]
        env.step(first)
        env.step(opposite[first])
        reward = 0.0
        for m in moves:
            obs, reward, done, info = env.step(m)
        assert done is True
        assert reward < 1.0

    def test_done_true_on_win(self):
        env = make_env(seed=0)
        moves = moves_from_path(env._state.optimal_path)
        done  = False
        for m in moves:
            _, _, done, _ = env.step(m)
        assert done is True

    def test_reward_formula_exact(self):
        env = make_env(seed=0)
        path  = env._state.optimal_path
        moves = moves_from_path(path)
        opposite = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP"}
        first = moves[0]
        env.step(first)
        env.step(opposite[first])
        for m in moves:
            _, reward, done, info = env.step(m)
        if done:
            assert reward == info["score"] / 100.0

    def test_info_won_true_on_win(self):
        env = make_env(seed=0)
        moves = moves_from_path(env._state.optimal_path)
        for m in moves:
            _, _, _, info = env.step(m)
        assert info["won"] is True

    def test_reward_bump_when_max_steps_exceeded_on_boundary(self):
        env = DungeonEnv(seed=0, max_steps=1)
        env.reset()
        env._state.char_pos = (0, 0)
        env._state.exit_pos = (9, 9)
        _, reward, done, _ = env.step("UP")
        assert done is True
        assert reward == REWARD_BUMP


# ===========================================================================
# DungeonEnv — modes de seed
# ===========================================================================

class TestDungeonEnvSeedModes:
    def test_fixed_seed_reproducible_across_instances(self):
        obs1 = DungeonEnv(seed=7).reset()
        obs2 = DungeonEnv(seed=7).reset()
        assert obs1 == obs2

    def test_none_seed_varies_across_resets(self):
        env = DungeonEnv(seed=None)
        positions = {env.reset()["char_pos"] for _ in range(10)}
        assert len(positions) > 1

    def test_seed_pool_draws_from_pool_only(self):
        pool = [10, 20, 30]
        valid_obs = {DungeonEnv(seed=s).reset()["char_pos"] for s in pool}
        env = DungeonEnv(seed_pool=pool)
        for _ in range(20):
            obs = env.reset()
            assert obs["char_pos"] in valid_obs

    def test_seed_pool_varies_between_resets(self):
        pool = list(range(50))
        env = DungeonEnv(seed_pool=pool)
        positions = {env.reset()["char_pos"] for _ in range(20)}
        assert len(positions) > 1


# ===========================================================================
# DungeonEnv — observation
# ===========================================================================

class TestDungeonEnvObservation:
    def test_char_pos_updates_after_successful_move(self):
        env = make_env(seed=0)
        path      = env._state.optimal_path
        next_pos  = path[1]
        direction = moves_from_path(path[:2])[0]
        obs, _, _, _ = env.step(direction)
        assert obs["char_pos"] == next_pos

    def test_exit_pos_unchanged_after_move(self):
        env = make_env(seed=0)
        exit_pos = env._state.exit_pos
        obs, _, _, _ = env.step("UP")
        assert obs["exit_pos"] == exit_pos

    def test_grid_encoding_matches_tile_types(self):
        from grid import TileType
        env = make_env(seed=0)
        obs  = env._observe()
        grid = env._state.grid
        enc  = {TileType.GRASS: 0, TileType.ROCK: 1, TileType.WATER: 2}
        for idx, expected in enumerate(obs["grid"]):
            x = idx % Grid.WIDTH
            y = idx // Grid.WIDTH
            assert expected == enc[grid.get_tile(x, y)]


# ===========================================================================
# current_seed
# ===========================================================================

class TestCurrentSeed:
    def test_fixed_seed_returns_seed_value(self):
        env = DungeonEnv(seed=42)
        env.reset()
        assert env.current_seed == 42

    def test_pool_returns_value_not_index(self):
        pool = [10, 20, 30]
        env  = DungeonEnv(seed_pool=pool)
        for _ in range(20):
            env.reset()
            assert env.current_seed in pool

    def test_random_mode_returns_none(self):
        env = DungeonEnv(seed=None)
        env.reset()
        assert env.current_seed is None

    def test_consistent_with_seed_idx(self):
        pool = [5, 15, 25, 35]
        env  = DungeonEnv(seed_pool=pool)
        for _ in range(20):
            env.reset()
            assert env.current_seed == pool[env.current_seed_idx]
