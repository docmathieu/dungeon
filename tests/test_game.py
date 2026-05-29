"""
Unit tests for Grid, GameState, scoring and trail.

Scope : business rules only — tile costs, move mechanics, win/score, trail, optimal path.
Removed : TileType enum constants, Grid layout stats, Grid._index internals,
          grass_cells helper, DIRECTIONS constant, direction case-insensitivity,
          GameState initial zero-values.
"""
import pytest
from unittest.mock import patch

from grid import Grid, TileType
from game_state import GameState
from pathfinder import PathFinder
from helpers import FakeGrid


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _first_cell_of(grid, tile: TileType) -> tuple[int, int]:
    for y in range(Grid.HEIGHT):
        for x in range(Grid.WIDTH):
            if grid.get_tile(x, y) == tile:
                return x, y
    pytest.skip(f"No {tile} cell found in test grid")


def make_state(
    char_pos: tuple = (5, 5),
    exit_pos: tuple = (8, 8),
    overrides: dict | None = None,
) -> GameState:
    """Build a GameState with fully-controlled positions and tile layout."""
    state = GameState(FakeGrid(overrides), seed=0)
    state.char_pos = char_pos
    state.exit_pos = exit_pos
    state._compute_optimal_path()
    state.trail = [char_pos]
    return state


# ===========================================================================
# Grid — passability and cost (business rules)
# ===========================================================================

class TestGridPassability:
    def setup_method(self):
        self.g = Grid(seed=0)

    def test_grass_is_passable(self):
        assert self.g.is_passable(*_first_cell_of(self.g, TileType.GRASS)) is True

    def test_water_is_passable(self):
        assert self.g.is_passable(*_first_cell_of(self.g, TileType.WATER)) is True

    def test_rock_is_not_passable(self):
        assert self.g.is_passable(*_first_cell_of(self.g, TileType.ROCK)) is False


class TestGridMoveCost:
    def setup_method(self):
        self.g = Grid(seed=0)

    def test_grass_costs_one(self):
        assert self.g.move_cost(*_first_cell_of(self.g, TileType.GRASS)) == 1

    def test_water_costs_two(self):
        assert self.g.move_cost(*_first_cell_of(self.g, TileType.WATER)) == 2

    def test_rock_costs_one(self):
        assert self.g.move_cost(*_first_cell_of(self.g, TileType.ROCK)) == 1


# ===========================================================================
# GameState — initialisation
# ===========================================================================

class TestGameStateInit:
    def test_char_placed_on_grass(self):
        grid = FakeGrid()
        state = GameState(grid, seed=0)
        assert grid.get_tile(*state.char_pos) == TileType.GRASS

    def test_exit_placed_on_grass(self):
        grid = FakeGrid()
        state = GameState(grid, seed=0)
        assert grid.get_tile(*state.exit_pos) == TileType.GRASS

    def test_char_and_exit_at_different_positions(self):
        state = GameState(FakeGrid(), seed=0)
        assert state.char_pos != state.exit_pos

    def test_positions_are_within_bounds(self):
        state = GameState(FakeGrid(), seed=0)
        for pos in (state.char_pos, state.exit_pos):
            x, y = pos
            assert 0 <= x < FakeGrid.WIDTH
            assert 0 <= y < FakeGrid.HEIGHT

    def test_seeded_positions_are_deterministic(self):
        s1 = GameState(FakeGrid(), seed=7)
        s2 = GameState(FakeGrid(), seed=7)
        assert s1.char_pos == s2.char_pos
        assert s1.exit_pos == s2.exit_pos

    def test_raises_when_only_one_grass_cell(self):
        overrides = {(x, y): TileType.ROCK for y in range(10) for x in range(10)}
        overrides[(0, 0)] = TileType.GRASS
        with pytest.raises(ValueError, match="Not enough grass"):
            GameState(FakeGrid(overrides), seed=0)

    def test_raises_when_no_grass_cells(self):
        overrides = {(x, y): TileType.ROCK for y in range(10) for x in range(10)}
        with pytest.raises(ValueError):
            GameState(FakeGrid(overrides), seed=0)


# ===========================================================================
# GameState — movement directions
# ===========================================================================

class TestApplyMoveDirections:
    def test_move_right(self):
        state = make_state(char_pos=(3, 5))
        state.apply_move("RIGHT")
        assert state.char_pos == (4, 5)

    def test_move_left(self):
        state = make_state(char_pos=(3, 5))
        state.apply_move("LEFT")
        assert state.char_pos == (2, 5)

    def test_move_up(self):
        state = make_state(char_pos=(3, 5))
        state.apply_move("UP")
        assert state.char_pos == (3, 4)

    def test_move_down(self):
        state = make_state(char_pos=(3, 5))
        state.apply_move("DOWN")
        assert state.char_pos == (3, 6)


# ===========================================================================
# GameState — move cost accumulation
# ===========================================================================

class TestApplyMoveCost:
    def test_grass_move_costs_one(self):
        state = make_state(char_pos=(3, 5))
        state.apply_move("RIGHT")
        assert state.move_count == 1

    def test_water_move_costs_two(self):
        state = make_state(char_pos=(3, 5), overrides={(4, 5): TileType.WATER})
        state.apply_move("RIGHT")
        assert state.move_count == 2

    def test_two_grass_moves_accumulate(self):
        state = make_state(char_pos=(1, 5))
        state.apply_move("RIGHT")
        state.apply_move("RIGHT")
        assert state.move_count == 2

    def test_mixed_terrain_accumulates_correctly(self):
        state = make_state(char_pos=(1, 5), overrides={(2, 5): TileType.WATER})
        state.apply_move("RIGHT")   # water +2
        state.apply_move("RIGHT")   # grass +1
        assert state.move_count == 3

    def test_rock_bump_costs_one(self):
        state = make_state(char_pos=(3, 5), overrides={(4, 5): TileType.ROCK})
        state.apply_move("RIGHT")
        assert state.move_count == 1


# ===========================================================================
# GameState — boundary blocking
# ===========================================================================

class TestApplyMoveBoundary:
    def test_left_boundary(self):
        state = make_state(char_pos=(0, 5))
        state.apply_move("LEFT")
        assert state.char_pos == (0, 5)
        assert state.move_count == 1

    def test_right_boundary(self):
        state = make_state(char_pos=(9, 5))
        state.apply_move("RIGHT")
        assert state.char_pos == (9, 5)
        assert state.move_count == 1

    def test_top_boundary(self):
        state = make_state(char_pos=(5, 0))
        state.apply_move("UP")
        assert state.char_pos == (5, 0)
        assert state.move_count == 1

    def test_bottom_boundary(self):
        state = make_state(char_pos=(5, 9))
        state.apply_move("DOWN")
        assert state.char_pos == (5, 9)
        assert state.move_count == 1

    def test_corner_top_left(self):
        state = make_state(char_pos=(0, 0))
        state.apply_move("UP")
        state.apply_move("LEFT")
        assert state.char_pos == (0, 0)
        assert state.move_count == 2

    def test_corner_bottom_right(self):
        state = make_state(char_pos=(9, 9))
        state.apply_move("DOWN")
        state.apply_move("RIGHT")
        assert state.char_pos == (9, 9)
        assert state.move_count == 2

    def test_multiple_boundary_bumps_accumulate(self):
        state = make_state(char_pos=(0, 5))
        state.apply_move("LEFT")
        state.apply_move("LEFT")
        state.apply_move("LEFT")
        assert state.move_count == 3

    def test_boundary_bump_does_not_change_trail(self):
        state = make_state(char_pos=(0, 5))
        state.apply_move("LEFT")
        assert state.trail == [(0, 5)]


# ===========================================================================
# GameState — rock blocking
# ===========================================================================

class TestApplyMoveRock:
    def test_blocked_by_adjacent_rock(self):
        state = make_state(char_pos=(3, 5), overrides={(4, 5): TileType.ROCK})
        state.apply_move("RIGHT")
        assert state.char_pos == (3, 5)
        assert state.move_count == 1

    def test_rock_above(self):
        state = make_state(char_pos=(3, 5), overrides={(3, 4): TileType.ROCK})
        state.apply_move("UP")
        assert state.char_pos == (3, 5)

    def test_surrounded_by_rocks_no_movement(self):
        overrides = {
            (4, 5): TileType.ROCK, (2, 5): TileType.ROCK,
            (3, 4): TileType.ROCK, (3, 6): TileType.ROCK,
        }
        state = make_state(char_pos=(3, 5), overrides=overrides)
        for direction in ("RIGHT", "LEFT", "UP", "DOWN"):
            state.apply_move(direction)
        assert state.char_pos == (3, 5)
        assert state.move_count == 4

    def test_multiple_bumps_same_rock_accumulate(self):
        state = make_state(char_pos=(3, 5), overrides={(4, 5): TileType.ROCK})
        state.apply_move("RIGHT")
        state.apply_move("RIGHT")
        state.apply_move("RIGHT")
        assert state.move_count == 3

    def test_rock_bump_does_not_change_trail(self):
        state = make_state(char_pos=(3, 5), overrides={(4, 5): TileType.ROCK})
        state.apply_move("RIGHT")
        assert state.trail == [(3, 5)]


# ===========================================================================
# GameState — win condition
# ===========================================================================

class TestWinCondition:
    def test_reaching_exit_sets_state(self):
        """Reaching exit sets won=True, score=100, info=GAGNE, char at exit."""
        state = make_state(char_pos=(4, 5), exit_pos=(5, 5))
        state.apply_move("RIGHT")
        assert state.won is True
        assert state.score == 100
        assert state.info == "GAGNE"
        assert state.char_pos == (5, 5)

    def test_not_won_before_reaching_exit(self):
        state = make_state(char_pos=(3, 5), exit_pos=(5, 5))
        state.apply_move("RIGHT")   # -> (4,5), not exit
        assert state.won is False
        assert state.score == 0
        assert state.info == ""

    def test_no_move_applied_after_win(self):
        state = make_state(char_pos=(4, 5), exit_pos=(5, 5))
        state.apply_move("RIGHT")   # wins
        pos, count, score = state.char_pos, state.move_count, state.score
        state.apply_move("RIGHT")   # ignored
        assert state.char_pos == pos
        assert state.move_count == count
        assert state.score == score

    def test_win_via_water_tile(self):
        state = make_state(
            char_pos=(4, 5), exit_pos=(5, 5),
            overrides={(5, 5): TileType.WATER},
        )
        state.apply_move("RIGHT")
        assert state.won is True
        assert state.move_count == 2   # water costs 2


# ===========================================================================
# GameState — win scoring (0–100)
# ===========================================================================

class TestWinScore:
    def test_optimal_path_scores_100(self):
        state = make_state(char_pos=(4, 5), exit_pos=(5, 5))
        state.apply_move("RIGHT")
        assert state.score == 100

    def test_suboptimal_path_scores_less_than_100(self):
        state = make_state(char_pos=(4, 5), exit_pos=(5, 5))
        state.apply_move("DOWN")
        state.apply_move("UP")
        state.apply_move("RIGHT")
        assert state.score < 100
        assert state.won is True

    def test_score_formula_exact(self):
        """round(100 × optimal / player): optimal=1, player=3 → score=33."""
        state = make_state(char_pos=(4, 5), exit_pos=(5, 5))
        state.apply_move("DOWN")
        state.apply_move("UP")
        state.apply_move("RIGHT")
        assert state.score == round(100 * 1 / 3)

    def test_score_uses_water_cost_in_optimal(self):
        state = make_state(
            char_pos=(4, 5), exit_pos=(5, 5),
            overrides={(5, 5): TileType.WATER},
        )
        state.apply_move("RIGHT")   # move_count=2, optimal=2
        assert state.score == 100

    def test_score_falls_back_to_100_when_optimal_cost_is_none(self):
        state = make_state(char_pos=(4, 5), exit_pos=(5, 5))
        state._optimal_cost = None
        state.apply_move("RIGHT")
        assert state.score == 100

    def test_boundary_bump_reduces_win_score(self):
        """char=(0,5), exit=(1,5). optimal=1, player: LEFT bump + RIGHT = 2 → score=50."""
        state = make_state(char_pos=(0, 5), exit_pos=(1, 5))
        state.apply_move("LEFT")
        state.apply_move("RIGHT")
        assert state.won is True
        assert state.score == round(100 * 1 / 2)

    def test_rock_bump_reduces_win_score(self):
        """char=(3,5), exit=(4,5), rock at (3,6). optimal=1, player: DOWN bump + RIGHT = 2 → score=50."""
        state = make_state(
            char_pos=(3, 5), exit_pos=(4, 5),
            overrides={(3, 6): TileType.ROCK},
        )
        state.apply_move("DOWN")
        state.apply_move("RIGHT")
        assert state.won is True
        assert state.score == round(100 * 1 / 2)


# ===========================================================================
# GameState — solvability & create_solvable
# ===========================================================================

class TestGameStateSolvable:
    def test_is_solvable_true_when_path_exists(self):
        state = make_state(char_pos=(0, 0), exit_pos=(9, 9))
        assert state.is_solvable() is True

    def test_is_solvable_false_when_enclosed_by_rocks(self):
        overrides = {(1, 0): TileType.ROCK, (0, 1): TileType.ROCK}
        state = make_state(char_pos=(0, 0), exit_pos=(5, 5), overrides=overrides)
        assert state.is_solvable() is False

    def test_is_solvable_false_when_exit_enclosed(self):
        overrides = {(8, 9): TileType.ROCK, (9, 8): TileType.ROCK}
        state = make_state(char_pos=(0, 0), exit_pos=(9, 9), overrides=overrides)
        assert state.is_solvable() is False

    def test_is_solvable_true_after_water_detour(self):
        overrides = {(1, 0): TileType.WATER, (0, 1): TileType.WATER}
        state = make_state(char_pos=(0, 0), exit_pos=(2, 2), overrides=overrides)
        assert state.is_solvable() is True

    def test_create_solvable_returns_solvable_state(self):
        assert GameState.create_solvable().is_solvable() is True

    def test_create_solvable_repeated_always_solvable(self):
        for _ in range(5):
            assert GameState.create_solvable().is_solvable() is True

    def test_create_solvable_retries_when_first_grid_unsolvable(self):
        with patch.object(PathFinder, 'find_shortest_path', side_effect=[None, ["RIGHT"]]):
            state = GameState.create_solvable(seed=0)
        assert state.is_solvable() is True

    def test_create_solvable_seed_incremented_on_retry(self):
        seen_seeds: list = []
        original_init = Grid.__init__

        def capturing_init(self_grid, seed=None):
            seen_seeds.append(seed)
            original_init(self_grid, seed=seed)

        with patch.object(Grid, '__init__', capturing_init):
            with patch.object(PathFinder, 'find_shortest_path', side_effect=[None, ["RIGHT"]]):
                GameState.create_solvable(seed=10)

        assert seen_seeds == [10, 11]

    def test_create_solvable_without_seed_retries_randomly(self):
        with patch.object(PathFinder, 'find_shortest_path', side_effect=[None, None, ["RIGHT"]]):
            state = GameState.create_solvable()
        assert state.is_solvable() is True


# ===========================================================================
# GameState — trail
# ===========================================================================

class TestTrail:
    def test_trail_initialized_with_char_pos(self):
        state = make_state(char_pos=(3, 5))
        assert state.trail == [(3, 5)]

    def test_trail_appended_on_successful_move(self):
        state = make_state(char_pos=(3, 5))
        state.apply_move("RIGHT")
        assert state.trail == [(3, 5), (4, 5)]

    def test_trail_not_appended_on_rock_block(self):
        state = make_state(char_pos=(3, 5), overrides={(4, 5): TileType.ROCK})
        state.apply_move("RIGHT")
        assert state.trail == [(3, 5)]

    def test_trail_not_appended_on_boundary(self):
        state = make_state(char_pos=(0, 5))
        state.apply_move("LEFT")
        assert state.trail == [(0, 5)]

    def test_trail_not_appended_after_win(self):
        state = make_state(char_pos=(4, 5), exit_pos=(5, 5))
        state.apply_move("RIGHT")   # wins at (5,5)
        state.apply_move("RIGHT")   # ignored
        assert state.trail == [(4, 5), (5, 5)]

    def test_trail_records_path_correctly(self):
        state = make_state(char_pos=(0, 0))
        state.apply_move("RIGHT")
        state.apply_move("DOWN")
        state.apply_move("LEFT")
        assert state.trail == [(0, 0), (1, 0), (1, 1), (0, 1)]

    def test_trail_can_revisit_positions(self):
        state = make_state(char_pos=(3, 5))
        state.apply_move("RIGHT")
        state.apply_move("LEFT")
        assert state.trail == [(3, 5), (4, 5), (3, 5)]


# ===========================================================================
# GameState — optimal_path
# ===========================================================================

class TestOptimalPath:
    def test_optimal_path_starts_at_char_pos(self):
        state = make_state(char_pos=(2, 3), exit_pos=(5, 6))
        assert state.optimal_path[0] == (2, 3)

    def test_optimal_path_ends_at_exit_pos(self):
        state = make_state(char_pos=(2, 3), exit_pos=(5, 6))
        assert state.optimal_path[-1] == (5, 6)

    def test_optimal_path_is_none_when_unreachable(self):
        overrides = {(1, 0): TileType.ROCK, (0, 1): TileType.ROCK}
        state = make_state(char_pos=(0, 0), exit_pos=(5, 5), overrides=overrides)
        assert state.optimal_path is None

    def test_optimal_path_each_position_adjacent_to_next(self):
        state = make_state(char_pos=(0, 0), exit_pos=(4, 3))
        path = state.optimal_path
        for i in range(len(path) - 1):
            ax, ay = path[i]
            bx, by = path[i + 1]
            assert abs(bx - ax) + abs(by - ay) == 1

    def test_optimal_path_cost_matches_optimal_cost(self):
        state = make_state(char_pos=(0, 0), exit_pos=(3, 3),
                           overrides={(1, 0): TileType.WATER})
        path = state.optimal_path
        grid = FakeGrid({(1, 0): TileType.WATER})
        walked_cost = sum(grid.move_cost(path[i+1][0], path[i+1][1])
                          for i in range(len(path) - 1))
        assert walked_cost == state._optimal_cost

    def test_optimal_path_avoids_rocks(self):
        overrides = {(1, 0): TileType.ROCK}
        state = make_state(char_pos=(0, 0), exit_pos=(2, 0), overrides=overrides)
        assert (1, 0) not in state.optimal_path
