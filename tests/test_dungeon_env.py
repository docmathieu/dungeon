"""Tests unitaires pour DungeonEnv (interface Gym — Phase 1 RL)."""

import pytest

from dungeon_env import DungeonEnv, ACTIONS, REWARD_STEP, REWARD_BUMP
from game_state import GameState
from grid import Grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(seed: int = 0, **kwargs) -> DungeonEnv:
    """Crée un DungeonEnv déjà réinitialisé (reset() appelé)."""
    env = DungeonEnv(seed=seed, **kwargs)
    env.reset()
    return env


def moves_from_path(path: list[tuple[int, int]]) -> list[str]:
    """Dérive la séquence de directions depuis une liste de positions."""
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
    def test_default_max_steps(self):
        assert DungeonEnv()._max_steps == DungeonEnv.MAX_STEPS

    def test_custom_max_steps(self):
        assert DungeonEnv(max_steps=50)._max_steps == 50

    def test_state_none_before_reset(self):
        assert DungeonEnv()._state is None

    def test_steps_zero_before_reset(self):
        assert DungeonEnv()._steps == 0

    def test_empty_seed_pool_raises(self):
        with pytest.raises(ValueError, match="vide"):
            DungeonEnv(seed_pool=[])

    def test_action_space_contains_four_actions(self):
        env = DungeonEnv()
        assert set(env.action_space) == {"LEFT", "RIGHT", "UP", "DOWN"}

    def test_actions_constant_contains_four(self):
        assert len(ACTIONS) == 4


# ===========================================================================
# DungeonEnv — reset()
# ===========================================================================

class TestDungeonEnvReset:
    def test_reset_returns_dict(self):
        obs = DungeonEnv(seed=0).reset()
        assert isinstance(obs, dict)

    def test_obs_has_required_keys(self):
        obs = DungeonEnv(seed=0).reset()
        assert set(obs.keys()) == {"grid", "char_pos", "exit_pos"}

    def test_grid_length_is_100(self):
        obs = DungeonEnv(seed=0).reset()
        assert len(obs["grid"]) == Grid.WIDTH * Grid.HEIGHT

    def test_grid_values_are_0_1_or_2(self):
        obs = DungeonEnv(seed=0).reset()
        assert all(v in (0, 1, 2) for v in obs["grid"])

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

    def test_steps_zero_after_reset(self):
        env = DungeonEnv(seed=0)
        env.reset()
        assert env._steps == 0

    def test_second_reset_reinitialises(self):
        """Deux resets successifs avec le même seed produisent la même obs."""
        env = DungeonEnv(seed=5)
        env.reset()
        env.step("UP")        # avance l'état
        obs1 = env.reset()    # doit remettre à zéro
        obs2 = DungeonEnv(seed=5).reset()
        assert obs1 == obs2

    def test_terrain_is_solvable_after_reset(self):
        """Le terrain retourné par reset() est toujours jouable."""
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

    def test_step_returns_four_elements(self):
        result = make_env(seed=0).step("RIGHT")
        assert len(result) == 4

    def test_obs_format_in_step(self):
        obs, *_ = make_env(seed=0).step("RIGHT")
        assert set(obs.keys()) == {"grid", "char_pos", "exit_pos"}
        assert len(obs["grid"]) == 100

    def test_reward_negative_on_non_terminal_step(self):
        """Tout pas non terminal donne une reward négative (REWARD_STEP ou REWARD_BUMP)."""
        env = make_env(seed=0)
        _, reward, done, _ = env.step("UP")
        if not done:
            assert reward < 0.0

    def test_done_false_before_max_steps(self):
        env = DungeonEnv(seed=0, max_steps=100)
        env.reset()
        _, _, done, _ = env.step("UP")   # un seul pas
        if not env._state.won:
            assert done is False

    def test_done_true_at_max_steps(self):
        """Après max_steps pas, done vaut True quelle que soit la situation."""
        env = DungeonEnv(seed=0, max_steps=1)
        env.reset()
        _, _, done, _ = env.step("UP")
        assert done is True

    def test_done_true_after_exactly_max_steps(self):
        env = DungeonEnv(seed=0, max_steps=3)
        env.reset()
        for i in range(2):
            _, _, done, _ = env.step("UP")
            if not env._state.won:
                assert done is False
        _, _, done, _ = env.step("UP")
        assert done is True

    def test_info_contains_required_keys(self):
        _, _, _, info = make_env(seed=0).step("RIGHT")
        for key in ("score", "moves", "steps", "won"):
            assert key in info

    def test_info_steps_matches_internal_counter(self):
        env = make_env(seed=0)
        env.step("RIGHT")
        _, _, _, info = env.step("LEFT")
        assert info["steps"] == env._steps == 2

    def test_step_increments_steps_counter(self):
        env = make_env(seed=0)
        assert env._steps == 0
        env.step("RIGHT")
        assert env._steps == 1
        env.step("LEFT")
        assert env._steps == 2

    def test_reward_normal_step_is_reward_step(self):
        """Déplacement réussi sans victoire → reward == REWARD_STEP."""
        env = make_env(seed=0)
        direction = moves_from_path(env._state.optimal_path[:2])[0]
        _, reward, done, _ = env.step(direction)
        if not done:
            assert reward == pytest.approx(REWARD_STEP)

    def test_reward_boundary_bump_is_reward_bump(self):
        """Choc contre le bord de grille → reward == REWARD_BUMP."""
        env = make_env(seed=0)
        env._state.char_pos = (0, 0)
        env._state.exit_pos = (9, 9)
        _, reward, _, _ = env.step("UP")   # bord supérieur
        assert reward == pytest.approx(REWARD_BUMP)

    def test_reward_rock_bump_is_reward_bump(self):
        """Choc contre un rocher → reward == REWARD_BUMP."""
        from grid import TileType
        from helpers import FakeGrid
        from game_state import GameState
        env = make_env(seed=0)
        # Forcer une roche à droite du personnage
        overrides = {(4, 5): TileType.ROCK}
        fake = FakeGrid(overrides)
        env._state = GameState(fake, seed=0)
        env._state.char_pos = (3, 5)
        env._state.exit_pos = (9, 9)
        _, reward, _, _ = env.step("RIGHT")   # frappe la roche en (4,5)
        assert reward == pytest.approx(REWARD_BUMP)


# ===========================================================================
# DungeonEnv — victoire et reward
# ===========================================================================

class TestDungeonEnvWin:
    def test_reward_one_on_optimal_win(self):
        """Suivre le chemin optimal donne reward == 1.0."""
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
        """Un détour avant la sortie réduit le reward en dessous de 1.0."""
        env = make_env(seed=0)
        path  = env._state.optimal_path
        moves = moves_from_path(path)
        # Ajouter un aller-retour au début (coût +2) avant le chemin optimal
        opposite = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP"}
        first = moves[0]
        env.step(first)
        env.step(opposite[first])   # revient en arrière
        reward = 0.0
        done   = False
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
        """reward = state.score / 100.0 exactement."""
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
        """Dernier pas = choc sur bord → reward == REWARD_BUMP même si done=True."""
        env = DungeonEnv(seed=0, max_steps=1)
        env.reset()
        env._state.char_pos = (0, 0)
        env._state.exit_pos = (9, 9)
        _, reward, done, _ = env.step("UP")  # boundary bump
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
        """seed=None → terrain différent à chaque reset() (statistiquement)."""
        env = DungeonEnv(seed=None)
        positions = {env.reset()["char_pos"] for _ in range(10)}
        assert len(positions) > 1   # au moins deux terrains distincts

    def test_seed_pool_draws_from_pool_only(self):
        """Chaque reset() retourne une obs correspondant à un seed du pool."""
        pool = [10, 20, 30]
        valid_obs = {DungeonEnv(seed=s).reset()["char_pos"] for s in pool}
        env = DungeonEnv(seed_pool=pool)
        for _ in range(20):
            obs = env.reset()
            assert obs["char_pos"] in valid_obs

    def test_seed_pool_varies_between_resets(self):
        """Un pool de plusieurs seeds produit des terrains variés."""
        pool = list(range(50))
        env = DungeonEnv(seed_pool=pool)
        positions = {env.reset()["char_pos"] for _ in range(20)}
        assert len(positions) > 1


# ===========================================================================
# DungeonEnv — observation
# ===========================================================================

class TestDungeonEnvObservation:
    def test_char_pos_in_obs_matches_state(self):
        env = make_env(seed=0)
        obs = env._observe()
        assert obs["char_pos"] == env._state.char_pos

    def test_char_pos_updates_after_successful_move(self):
        """Après un déplacement valide, char_pos dans l'obs change."""
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
        """La grille encodée reflète fidèlement les types de cases."""
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
# current_seed_idx
# ===========================================================================

class TestCurrentSeedIdx:
    def test_default_zero_before_reset(self):
        """Avant tout reset(), current_seed_idx vaut 0."""
        env = DungeonEnv(seed_pool=[10, 20, 30])
        assert env.current_seed_idx == 0

    def test_no_pool_always_zero(self):
        """Sans seed_pool, current_seed_idx reste 0 après reset()."""
        env = DungeonEnv(seed=42)
        for _ in range(5):
            env.reset()
            assert env.current_seed_idx == 0

    def test_pool_idx_in_valid_range(self):
        """Avec seed_pool de taille N, current_seed_idx est dans [0, N[."""
        pool = [10, 20, 30]
        env  = DungeonEnv(seed_pool=pool)
        for _ in range(20):
            env.reset()
            assert 0 <= env.current_seed_idx < len(pool)

    def test_pool_idx_changes_across_resets(self):
        """Sur suffisamment de resets, plusieurs index distincts sont observés."""
        env = DungeonEnv(seed_pool=list(range(10)))
        seen = {env.reset() and env.current_seed_idx for _ in range(50)}
        assert len(seen) > 1   # au moins 2 seeds différents tirés


# ===========================================================================
# current_seed
# ===========================================================================

class TestCurrentSeed:
    def test_fixed_seed_returns_seed_value(self):
        """Avec seed fixe, current_seed retourne la valeur du seed."""
        env = DungeonEnv(seed=42)
        env.reset()
        assert env.current_seed == 42

    def test_pool_returns_value_not_index(self):
        """Avec seed_pool, current_seed retourne la valeur du seed, pas son index."""
        pool = [10, 20, 30]
        env  = DungeonEnv(seed_pool=pool)
        for _ in range(20):
            env.reset()
            assert env.current_seed in pool

    def test_random_mode_returns_none(self):
        """Sans seed ni pool, current_seed vaut None."""
        env = DungeonEnv(seed=None)
        env.reset()
        assert env.current_seed is None

    def test_consistent_with_seed_idx(self):
        """current_seed == seed_pool[current_seed_idx]."""
        pool = [5, 15, 25, 35]
        env  = DungeonEnv(seed_pool=pool)
        for _ in range(20):
            env.reset()
            assert env.current_seed == pool[env.current_seed_idx]

