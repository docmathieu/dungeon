"""
Comprehensive unit tests for the dungeon game.

Covers: Grid, TileType, GameState, DIRECTION_DELTA.
"""
import pytest
from unittest.mock import patch

from grid import Grid, TileType
from directions import DIRECTIONS
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
# TileType
# ===========================================================================

class TestTileType:
    def test_enum_string_values(self):
        assert TileType.GRASS.value == "grass"
        assert TileType.ROCK.value == "rock"
        assert TileType.WATER.value == "water"

    def test_three_members(self):
        assert len(list(TileType)) == 3

    def test_distinct_members(self):
        assert TileType.GRASS != TileType.ROCK
        assert TileType.GRASS != TileType.WATER
        assert TileType.ROCK != TileType.WATER


# ===========================================================================
# Grid — layout and distribution
# ===========================================================================

class TestGridLayout:
    def test_total_cell_count(self):
        assert len(Grid(seed=0)._cells) == 100

    def test_rock_count_is_thirty(self):
        assert Grid(seed=0)._cells.count(TileType.ROCK) == 30

    def test_water_count_is_twenty(self):
        assert Grid(seed=0)._cells.count(TileType.WATER) == 20

    def test_grass_count_is_fifty(self):
        assert Grid(seed=0)._cells.count(TileType.GRASS) == 50

    def test_seeded_generation_is_deterministic(self):
        assert Grid(seed=42)._cells == Grid(seed=42)._cells

    def test_different_seeds_produce_different_layouts(self):
        assert Grid(seed=1)._cells != Grid(seed=2)._cells

    def test_no_seed_is_random(self):
        # Without a fixed seed, repeated calls should usually differ.
        results = {tuple(Grid()._cells) for _ in range(5)}
        assert len(results) > 1

    def test_dimensions_constants(self):
        assert Grid.WIDTH == 10
        assert Grid.HEIGHT == 10


# ===========================================================================
# Grid — index helper
# ===========================================================================

class TestGridIndex:
    def setup_method(self):
        self.g = Grid(seed=0)

    def test_origin(self):
        assert self.g._index(0, 0) == 0

    def test_last_cell_in_first_row(self):
        assert self.g._index(9, 0) == 9

    def test_first_cell_in_second_row(self):
        assert self.g._index(0, 1) == 10

    def test_last_cell(self):
        assert self.g._index(9, 9) == 99

    def test_arbitrary_cell(self):
        assert self.g._index(3, 2) == 23


# ===========================================================================
# Grid — tile accessors
# ===========================================================================

class TestGridGetTile:
    def setup_method(self):
        self.g = Grid(seed=0)

    def test_returns_tile_type(self):
        assert isinstance(self.g.get_tile(0, 0), TileType)

    def test_all_cells_are_valid_tile_types(self):
        valid = set(TileType)
        for y in range(Grid.HEIGHT):
            for x in range(Grid.WIDTH):
                assert self.g.get_tile(x, y) in valid


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
        # move_cost is defined for any tile; rock costs 1 (but is impassable)
        assert self.g.move_cost(*_first_cell_of(self.g, TileType.ROCK)) == 1


class TestGridGrassCells:
    def setup_method(self):
        self.g = Grid(seed=0)

    def test_all_returned_cells_are_grass(self):
        for x, y in self.g.grass_cells():
            assert self.g.get_tile(x, y) == TileType.GRASS

    def test_count_matches_grass_in_cells(self):
        assert len(self.g.grass_cells()) == self.g._cells.count(TileType.GRASS)

    def test_coordinates_in_bounds(self):
        for x, y in self.g.grass_cells():
            assert 0 <= x < Grid.WIDTH
            assert 0 <= y < Grid.HEIGHT

    def test_no_rock_or_water_included(self):
        for x, y in self.g.grass_cells():
            assert self.g.get_tile(x, y) not in (TileType.ROCK, TileType.WATER)


# ===========================================================================
# GameState — initialisation
# ===========================================================================

class TestGameStateInit:
    def test_initial_move_count_is_zero(self):
        assert GameState(FakeGrid(), seed=0).move_count == 0

    def test_initial_score_is_zero(self):
        assert GameState(FakeGrid(), seed=0).score == 0

    def test_initial_info_is_empty(self):
        assert GameState(FakeGrid(), seed=0).info == ""

    def test_initial_won_is_false(self):
        assert GameState(FakeGrid(), seed=0).won is False

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
# DIRECTION_DELTA
# ===========================================================================

class TestDirections:
    def test_left(self):
        assert DIRECTIONS["LEFT"] == (-1, 0)

    def test_right(self):
        assert DIRECTIONS["RIGHT"] == (1, 0)

    def test_up(self):
        assert DIRECTIONS["UP"] == (0, -1)

    def test_down(self):
        assert DIRECTIONS["DOWN"] == (0, 1)

    def test_four_directions_defined(self):
        assert len(DIRECTIONS) == 4


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

    def test_lowercase_direction(self):
        state = make_state(char_pos=(3, 5))
        state.apply_move("right")
        assert state.char_pos == (4, 5)

    def test_mixed_case_direction(self):
        state = make_state(char_pos=(3, 5))
        state.apply_move("Right")
        assert state.char_pos == (4, 5)

    def test_invalid_direction_does_not_move(self):
        state = make_state(char_pos=(3, 5))
        state.apply_move("DIAGONAL")
        assert state.char_pos == (3, 5)

    def test_invalid_direction_does_not_count(self):
        state = make_state(char_pos=(3, 5))
        state.apply_move("INVALID")
        assert state.move_count == 0

    def test_empty_string_direction(self):
        state = make_state(char_pos=(3, 5))
        state.apply_move("")
        assert state.char_pos == (3, 5)
        assert state.move_count == 0


# ===========================================================================
# GameState — move cost accumulation
# ===========================================================================

class TestApplyMoveCost:
    def test_grass_move_costs_one(self):
        state = make_state(char_pos=(3, 5))
        state.apply_move("RIGHT")  # (4,5) defaults to GRASS
        assert state.move_count == 1

    def test_water_move_costs_two(self):
        state = make_state(char_pos=(3, 5), overrides={(4, 5): TileType.WATER})
        state.apply_move("RIGHT")
        assert state.move_count == 2

    def test_two_grass_moves_accumulate(self):
        state = make_state(char_pos=(1, 5))
        state.apply_move("RIGHT")  # -> (2,5) grass +1
        state.apply_move("RIGHT")  # -> (3,5) grass +1
        assert state.move_count == 2

    def test_mixed_terrain_accumulates_correctly(self):
        state = make_state(
            char_pos=(1, 5),
            overrides={(2, 5): TileType.WATER},
        )
        state.apply_move("RIGHT")  # -> (2,5) water +2
        state.apply_move("RIGHT")  # -> (3,5) grass +1
        assert state.move_count == 3

    def test_rock_bump_costs_one(self):
        # Moving into a rock counts as 1 move (no position change)
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
        assert state.move_count == 1   # boundary costs 1 like a rock

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
        assert state.move_count == 2   # two boundary bumps

    def test_corner_bottom_right(self):
        state = make_state(char_pos=(9, 9))
        state.apply_move("DOWN")
        state.apply_move("RIGHT")
        assert state.char_pos == (9, 9)
        assert state.move_count == 2

    def test_multiple_boundary_bumps_accumulate(self):
        """Hitting the same wall 3 times costs 3 moves total."""
        state = make_state(char_pos=(0, 5))
        state.apply_move("LEFT")
        state.apply_move("LEFT")
        state.apply_move("LEFT")
        assert state.move_count == 3

    def test_boundary_bump_does_not_change_trail(self):
        """Hitting a boundary must not append a new position to the trail."""
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
        assert state.char_pos == (3, 5)   # position unchanged
        assert state.move_count == 1       # rock bump costs 1

    def test_rock_above(self):
        state = make_state(char_pos=(3, 5), overrides={(3, 4): TileType.ROCK})
        state.apply_move("UP")
        assert state.char_pos == (3, 5)

    def test_surrounded_by_rocks_no_movement(self):
        overrides = {
            (4, 5): TileType.ROCK,
            (2, 5): TileType.ROCK,
            (3, 4): TileType.ROCK,
            (3, 6): TileType.ROCK,
        }
        state = make_state(char_pos=(3, 5), overrides=overrides)
        for direction in ("RIGHT", "LEFT", "UP", "DOWN"):
            state.apply_move(direction)
        assert state.char_pos == (3, 5)   # position unchanged
        assert state.move_count == 4       # 4 rock bumps × 1 each

    def test_multiple_bumps_same_rock_accumulate(self):
        """Hitting the same rock repeatedly accumulates the cost each time."""
        state = make_state(char_pos=(3, 5), overrides={(4, 5): TileType.ROCK})
        state.apply_move("RIGHT")
        state.apply_move("RIGHT")
        state.apply_move("RIGHT")
        assert state.move_count == 3

    def test_rock_bump_does_not_change_trail(self):
        """Bumping a rock must not append a new position to the trail."""
        state = make_state(char_pos=(3, 5), overrides={(4, 5): TileType.ROCK})
        state.apply_move("RIGHT")
        assert state.trail == [(3, 5)]

    def test_boundary_costs_one_like_rock(self):
        """Out-of-bounds moves cost 1, same penalty as hitting a rock."""
        state = make_state(char_pos=(0, 5))
        state.apply_move("LEFT")    # hits boundary
        assert state.move_count == 1


# ===========================================================================
# GameState — win condition
# ===========================================================================

class TestWinCondition:
    def test_reaching_exit_sets_won(self):
        state = make_state(char_pos=(4, 5), exit_pos=(5, 5))
        state.apply_move("RIGHT")
        assert state.won is True

    def test_reaching_exit_sets_score_to_100_when_optimal(self):
        state = make_state(char_pos=(4, 5), exit_pos=(5, 5))
        state.apply_move("RIGHT")
        assert state.score == 100

    def test_reaching_exit_sets_info_to_gagne(self):
        state = make_state(char_pos=(4, 5), exit_pos=(5, 5))
        state.apply_move("RIGHT")
        assert state.info == "GAGNE"

    def test_char_is_at_exit_after_win(self):
        state = make_state(char_pos=(4, 5), exit_pos=(5, 5))
        state.apply_move("RIGHT")
        assert state.char_pos == state.exit_pos

    def test_not_won_before_reaching_exit(self):
        state = make_state(char_pos=(3, 5), exit_pos=(5, 5))
        state.apply_move("RIGHT")  # -> (4,5), not exit
        assert state.won is False
        assert state.score == 0
        assert state.info == ""

    def test_no_move_applied_after_win(self):
        state = make_state(char_pos=(4, 5), exit_pos=(5, 5))
        state.apply_move("RIGHT")  # wins
        pos_after = state.char_pos
        count_after = state.move_count
        state.apply_move("RIGHT")  # should be ignored
        assert state.char_pos == pos_after
        assert state.move_count == count_after

    def test_no_score_change_after_win(self):
        state = make_state(char_pos=(4, 5), exit_pos=(5, 5))
        state.apply_move("RIGHT")
        state.apply_move("DOWN")
        assert state.score == 100

    def test_win_via_water_tile(self):
        state = make_state(
            char_pos=(4, 5),
            exit_pos=(5, 5),
            overrides={(5, 5): TileType.WATER},
        )
        state.apply_move("RIGHT")
        assert state.won is True
        assert state.move_count == 2  # water costs 2


# ===========================================================================
# GameState — win scoring (0–100)
# ===========================================================================

class TestWinScore:
    def test_optimal_path_scores_100(self):
        """One-step path: player cost == optimal cost → score 100."""
        state = make_state(char_pos=(4, 5), exit_pos=(5, 5))
        state.apply_move("RIGHT")
        assert state.score == 100

    def test_suboptimal_path_scores_less_than_100(self):
        """Player wastes two moves before reaching exit: score < 100."""
        # Optimal: RIGHT once, cost=1.  Player: DOWN, UP, RIGHT → cost=3.
        state = make_state(char_pos=(4, 5), exit_pos=(5, 5))
        state.apply_move("DOWN")   # (4,6)
        state.apply_move("UP")     # (4,5)
        state.apply_move("RIGHT")  # (5,5) = exit, move_count=3
        assert state.score < 100
        assert state.won is True

    def test_score_formula_exact(self):
        """round(100 × optimal / player): optimal=1, player=3 → score=33."""
        state = make_state(char_pos=(4, 5), exit_pos=(5, 5))
        state.apply_move("DOWN")
        state.apply_move("UP")
        state.apply_move("RIGHT")
        assert state.score == round(100 * 1 / 3)  # == 33

    def test_score_is_positive_when_winning(self):
        """Even a very suboptimal path yields score > 0 on win."""
        state = make_state(char_pos=(4, 5), exit_pos=(5, 5))
        for _ in range(4):
            state.apply_move("DOWN")
            state.apply_move("UP")
        state.apply_move("RIGHT")  # exit, move_count=9
        assert state.score > 0

    def test_score_never_exceeds_100(self):
        state = make_state(char_pos=(4, 5), exit_pos=(5, 5))
        state.apply_move("RIGHT")
        assert state.score <= 100

    def test_score_uses_water_cost_in_optimal(self):
        """Optimal through water costs 2; one-step player cost == optimal → 100."""
        state = make_state(
            char_pos=(4, 5),
            exit_pos=(5, 5),
            overrides={(5, 5): TileType.WATER},
        )
        state.apply_move("RIGHT")   # move_count=2, optimal=2
        assert state.score == 100

    def test_score_falls_back_to_100_when_optimal_cost_is_none(self):
        """If PathFinder finds no path (edge case), score defaults to 100 on win."""
        state = make_state(char_pos=(4, 5), exit_pos=(5, 5))
        state._optimal_cost = None
        state.apply_move("RIGHT")
        assert state.score == 100

    def test_score_zero_when_lost(self):
        state = make_state(char_pos=(3, 5), exit_pos=(8, 8))
        state.apply_move("RIGHT")
        assert state.score == 0

    def test_score_not_set_before_win(self):
        state = make_state(char_pos=(3, 5), exit_pos=(8, 8))
        state.apply_move("RIGHT")
        assert state.score == 0
        assert not state.won

    def test_boundary_bump_reduces_win_score(self):
        """Hitting the boundary before winning reduces the score below 100.

        Setup: char=(0,5), exit=(1,5).
        Optimal cost = 1 (one step RIGHT).
        Player: LEFT → hits boundary (+1), RIGHT → exit (+1), total = 2.
        Score = round(100 × 1 / 2) = 50.
        """
        state = make_state(char_pos=(0, 5), exit_pos=(1, 5))
        state.apply_move("LEFT")    # hits boundary → +1, pos unchanged
        state.apply_move("RIGHT")   # moves to exit (1,5) → +1
        assert state.won is True
        assert state.score == round(100 * 1 / 2)  # == 50

    def test_rock_bump_reduces_win_score(self):
        """Bumping a rock before winning reduces the score below 100.

        Setup: char=(3,5), exit=(4,5), rock at (3,6).
        Optimal cost = 1 (one step RIGHT).
        Player: DOWN → bumps rock (+1), RIGHT → exit (+1), total = 2.
        Score = round(100 × 1 / 2) = 50.
        """
        state = make_state(
            char_pos=(3, 5),
            exit_pos=(4, 5),
            overrides={(3, 6): TileType.ROCK},
        )
        state.apply_move("DOWN")    # bumps (3,6) rock → +1, pos unchanged
        state.apply_move("RIGHT")   # moves to exit (4,5) → +1
        assert state.won is True
        assert state.score == round(100 * 1 / 2)  # == 50


# ===========================================================================
# GameState — solvability & create_solvable
# ===========================================================================

class TestGameStateSolvable:
    def test_is_solvable_true_when_path_exists(self):
        state = make_state(char_pos=(0, 0), exit_pos=(9, 9))
        assert state.is_solvable() is True

    def test_is_solvable_false_when_enclosed_by_rocks(self):
        # (0,0) at top-left: UP/LEFT hit boundary, RIGHT/DOWN are rocks → no path
        overrides = {(1, 0): TileType.ROCK, (0, 1): TileType.ROCK}
        state = make_state(char_pos=(0, 0), exit_pos=(5, 5), overrides=overrides)
        assert state.is_solvable() is False

    def test_is_solvable_false_when_exit_enclosed(self):
        # Exit at (9,9) surrounded by rocks/boundary → no path from anywhere
        overrides = {(8, 9): TileType.ROCK, (9, 8): TileType.ROCK}
        state = make_state(char_pos=(0, 0), exit_pos=(9, 9), overrides=overrides)
        assert state.is_solvable() is False

    def test_is_solvable_true_after_water_detour(self):
        # All direct cells are water but passable; path still exists
        overrides = {(1, 0): TileType.WATER, (0, 1): TileType.WATER}
        state = make_state(char_pos=(0, 0), exit_pos=(2, 2), overrides=overrides)
        assert state.is_solvable() is True

    def test_create_solvable_returns_solvable_state(self):
        state = GameState.create_solvable()
        assert state.is_solvable() is True

    def test_create_solvable_repeated_always_solvable(self):
        for _ in range(5):
            assert GameState.create_solvable().is_solvable() is True

    def test_create_solvable_retries_when_first_grid_unsolvable(self):
        """find_shortest_path returns None on first call → retries → returns solvable state."""
        with patch.object(PathFinder, 'find_shortest_path', side_effect=[None, ["RIGHT"]]):
            state = GameState.create_solvable(seed=0)
        assert state.is_solvable() is True

    def test_create_solvable_seed_incremented_on_retry(self):
        """With a fixed seed, each retry increments the seed so grids differ."""
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
        """seed=None: retries use new random grids (seed stays None)."""
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

    def test_trail_length_equals_successful_moves_plus_one(self):
        state = make_state(char_pos=(0, 5))
        for _ in range(5):
            state.apply_move("RIGHT")
        assert len(state.trail) == 6

    def test_trail_records_path_correctly(self):
        state = make_state(char_pos=(0, 0))
        state.apply_move("RIGHT")  # (1, 0)
        state.apply_move("DOWN")   # (1, 1)
        state.apply_move("LEFT")   # (0, 1)
        assert state.trail == [(0, 0), (1, 0), (1, 1), (0, 1)]

    def test_trail_can_revisit_positions(self):
        state = make_state(char_pos=(3, 5))
        state.apply_move("RIGHT")  # (4, 5)
        state.apply_move("LEFT")   # (3, 5) back to start
        assert state.trail == [(3, 5), (4, 5), (3, 5)]

    def test_trail_includes_exit_as_last_entry_on_win(self):
        state = make_state(char_pos=(4, 5), exit_pos=(5, 5))
        state.apply_move("RIGHT")
        assert state.trail[-1] == (5, 5)

    def test_trail_blocked_move_not_counted(self):
        overrides = {
            (4, 5): TileType.ROCK,
            (2, 5): TileType.ROCK,
            (3, 4): TileType.ROCK,
            (3, 6): TileType.ROCK,
        }
        state = make_state(char_pos=(3, 5), overrides=overrides)
        for d in ("RIGHT", "LEFT", "UP", "DOWN"):
            state.apply_move(d)
        assert state.trail == [(3, 5)]

    def test_trail_water_tile_appended_once(self):
        state = make_state(char_pos=(3, 5), overrides={(4, 5): TileType.WATER})
        state.apply_move("RIGHT")
        assert state.trail == [(3, 5), (4, 5)]


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
        # (0,0) enclosed: boundary blocks UP/LEFT, rocks block RIGHT/DOWN
        overrides = {(1, 0): TileType.ROCK, (0, 1): TileType.ROCK}
        state = make_state(char_pos=(0, 0), exit_pos=(5, 5), overrides=overrides)
        assert state.optimal_path is None

    def test_optimal_path_length_on_all_grass(self):
        # Manhattan distance = 3 → 4 positions (start + 3 steps)
        state = make_state(char_pos=(0, 0), exit_pos=(2, 1))
        assert len(state.optimal_path) == 4

    def test_optimal_path_direct_one_step(self):
        state = make_state(char_pos=(4, 5), exit_pos=(5, 5))
        assert state.optimal_path == [(4, 5), (5, 5)]

    def test_optimal_path_each_position_adjacent_to_next(self):
        state = make_state(char_pos=(0, 0), exit_pos=(4, 3))
        path = state.optimal_path
        for i in range(len(path) - 1):
            ax, ay = path[i]
            bx, by = path[i + 1]
            assert abs(bx - ax) + abs(by - ay) == 1  # Manhattan step of 1

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
        path = state.optimal_path
        assert (1, 0) not in path

    def test_optimal_path_consistent_with_is_solvable(self):
        state = make_state(char_pos=(0, 0), exit_pos=(9, 9))
        assert state.is_solvable() == (state.optimal_path is not None)

    def test_optimal_path_init_uses_single_pathfinder_call(self):
        """find_shortest_path should be called once at init, not twice."""
        call_count = 0
        original = PathFinder.find_shortest_path

        def counting_find(self_pf, grid, start, end):
            nonlocal call_count
            call_count += 1
            return original(self_pf, grid, start, end)

        with patch.object(PathFinder, 'find_shortest_path', counting_find):
            GameState(FakeGrid(), seed=0)
        assert call_count == 1
