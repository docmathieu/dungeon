"""Unit tests for PathFinder (Dijkstra shortest-path)."""
import pytest

from grid import TileType
from pathfinder import PathFinder


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class FakeGrid:
    """All-grass 10×10 grid; individual cells can be overridden."""
    WIDTH = 10
    HEIGHT = 10

    def __init__(self, overrides: dict | None = None):
        self._overrides: dict[tuple[int, int], TileType] = overrides or {}

    def get_tile(self, x: int, y: int) -> TileType:
        return self._overrides.get((x, y), TileType.GRASS)

    def is_passable(self, x: int, y: int) -> bool:
        return self.get_tile(x, y) != TileType.ROCK

    def move_cost(self, x: int, y: int) -> int:
        return 2 if self.get_tile(x, y) == TileType.WATER else 1


DELTA = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}


def walk(grid, start: tuple[int, int], moves: list[str]) -> tuple[tuple[int, int], int]:
    """Simulate moves on a grid; return (final_pos, total_cost)."""
    x, y = start
    cost = 0
    for m in moves:
        dx, dy = DELTA[m]
        nx, ny = x + dx, y + dy
        cost += grid.move_cost(nx, ny)
        x, y = nx, ny
    return (x, y), cost


# ===========================================================================
# Start == end
# ===========================================================================

class TestPathFinderStartEqualsEnd:
    def test_find_returns_empty_list(self):
        assert PathFinder().find_shortest_path(FakeGrid(), (3, 3), (3, 3)) == []

    def test_cost_is_zero(self):
        assert PathFinder().shortest_cost(FakeGrid(), (3, 3), (3, 3)) == 0


# ===========================================================================
# Basic paths on all-grass grid
# ===========================================================================

class TestPathFinderBasic:
    def setup_method(self):
        self.pf = PathFinder()
        self.grid = FakeGrid()

    def test_one_step_right(self):
        assert self.pf.find_shortest_path(self.grid, (0, 0), (1, 0)) == ["RIGHT"]

    def test_one_step_down(self):
        assert self.pf.find_shortest_path(self.grid, (0, 0), (0, 1)) == ["DOWN"]

    def test_one_step_left(self):
        assert self.pf.find_shortest_path(self.grid, (1, 0), (0, 0)) == ["LEFT"]

    def test_one_step_up(self):
        assert self.pf.find_shortest_path(self.grid, (0, 1), (0, 0)) == ["UP"]

    def test_path_leads_to_destination(self):
        start, end = (1, 1), (4, 6)
        path = self.pf.find_shortest_path(self.grid, start, end)
        final, _ = walk(self.grid, start, path)
        assert final == end

    def test_cost_equals_manhattan_distance_on_all_grass(self):
        for start, end in [((0, 0), (3, 2)), ((1, 2), (8, 3)), ((0, 0), (9, 9))]:
            cost = self.pf.shortest_cost(self.grid, start, end)
            assert cost == abs(end[0] - start[0]) + abs(end[1] - start[1])

    def test_path_length_equals_manhattan_distance_on_all_grass(self):
        start, end = (2, 1), (5, 4)
        path = self.pf.find_shortest_path(self.grid, start, end)
        assert len(path) == abs(end[0] - start[0]) + abs(end[1] - start[1])

    def test_far_corner_to_far_corner(self):
        start, end = (0, 0), (9, 9)
        path = self.pf.find_shortest_path(self.grid, start, end)
        final, cost = walk(self.grid, start, path)
        assert final == end
        assert cost == 18  # Manhattan distance


# ===========================================================================
# Water — weighted cost
# ===========================================================================

class TestPathFinderWater:
    def setup_method(self):
        self.pf = PathFinder()

    def test_single_water_cell_costs_two(self):
        grid = FakeGrid(overrides={(1, 0): TileType.WATER})
        assert self.pf.shortest_cost(grid, (0, 0), (1, 0)) == 2

    def test_water_then_grass_cost_is_three(self):
        grid = FakeGrid(overrides={(1, 0): TileType.WATER})
        assert self.pf.shortest_cost(grid, (0, 0), (2, 0)) == 3  # water(2) + grass(1)

    def test_path_through_water_is_correct(self):
        grid = FakeGrid(overrides={(1, 0): TileType.WATER})
        path = self.pf.find_shortest_path(grid, (0, 0), (2, 0))
        final, cost = walk(grid, (0, 0), path)
        assert final == (2, 0)
        assert cost == 3

    def test_prefers_grass_detour_over_water_corridor(self):
        """3 water cells then grass: direct costs 7, 6-step grass detour costs 6."""
        overrides = {
            (1, 0): TileType.WATER,
            (2, 0): TileType.WATER,
            (3, 0): TileType.WATER,
        }
        grid = FakeGrid(overrides)
        # Direct (0,0)→(4,0): water+water+water+grass = 2+2+2+1 = 7
        # Detour via row 1:   down+right×4+up = 1+1+1+1+1+1 = 6
        assert self.pf.shortest_cost(grid, (0, 0), (4, 0)) == 6

    def test_detour_path_arrives_at_destination(self):
        overrides = {
            (1, 0): TileType.WATER,
            (2, 0): TileType.WATER,
            (3, 0): TileType.WATER,
        }
        grid = FakeGrid(overrides)
        start, end = (0, 0), (4, 0)
        path = self.pf.find_shortest_path(grid, start, end)
        final, cost = walk(grid, start, path)
        assert final == end
        assert cost == 6

    def test_direct_water_cheaper_than_long_detour(self):
        """One water cell (cost 2) is cheaper than a 3-step all-grass detour (cost 3)."""
        grid = FakeGrid(overrides={(1, 0): TileType.WATER})
        assert self.pf.shortest_cost(grid, (0, 0), (1, 0)) == 2


# ===========================================================================
# Obstacles — rocks blocking paths
# ===========================================================================

class TestPathFinderObstacles:
    def setup_method(self):
        self.pf = PathFinder()

    def test_no_path_from_enclosed_corner(self):
        """(0,0) enclosed: boundary blocks UP/LEFT, rocks block RIGHT/DOWN."""
        overrides = {(1, 0): TileType.ROCK, (0, 1): TileType.ROCK}
        grid = FakeGrid(overrides)
        assert self.pf.find_shortest_path(grid, (0, 0), (5, 5)) is None

    def test_shortest_cost_returns_none_when_no_path(self):
        overrides = {(1, 0): TileType.ROCK, (0, 1): TileType.ROCK}
        grid = FakeGrid(overrides)
        assert self.pf.shortest_cost(grid, (0, 0), (5, 5)) is None

    def test_path_goes_around_single_rock(self):
        overrides = {(1, 0): TileType.ROCK}
        grid = FakeGrid(overrides)
        start, end = (0, 0), (2, 0)
        path = self.pf.find_shortest_path(grid, start, end)
        assert path is not None
        final, _ = walk(grid, start, path)
        assert final == end

    def test_detour_around_rock_costs_more_than_direct(self):
        """Direct (0,0)→(2,0) costs 2 on grass; rock forces 4-step detour costing 4."""
        overrides = {(1, 0): TileType.ROCK}
        grid = FakeGrid(overrides)
        assert self.pf.shortest_cost(grid, (0, 0), (2, 0)) == 4

    def test_corridor_only_valid_route(self):
        """Only column x=5 is passable; path must travel that column."""
        overrides = {
            (x, y): TileType.ROCK
            for y in range(10)
            for x in range(10)
            if x != 5
        }
        grid = FakeGrid(overrides)
        path = self.pf.find_shortest_path(grid, (5, 0), (5, 9))
        assert path is not None
        assert all(m == "DOWN" for m in path)
        assert self.pf.shortest_cost(grid, (5, 0), (5, 9)) == 9

    def test_path_around_rock_wall_arrives_correctly(self):
        """Vertical rock wall from (5,0) to (5,8) forces a detour around (5,9)."""
        overrides = {(5, y): TileType.ROCK for y in range(9)}
        grid = FakeGrid(overrides)
        start, end = (0, 0), (9, 0)
        path = self.pf.find_shortest_path(grid, start, end)
        final, _ = walk(grid, start, path)
        assert final == end


# ===========================================================================
# Cost consistency — walked cost matches reported cost
# ===========================================================================

class TestPathFinderCostConsistency:
    def setup_method(self):
        self.pf = PathFinder()

    def test_walked_cost_matches_reported_cost(self):
        overrides = {(2, 0): TileType.WATER, (3, 1): TileType.WATER}
        grid = FakeGrid(overrides)
        start, end = (0, 0), (4, 3)
        path = self.pf.find_shortest_path(grid, start, end)
        reported = self.pf.shortest_cost(grid, start, end)
        _, walked = walk(grid, start, path)
        assert walked == reported

    def test_find_and_cost_agree_on_mixed_terrain(self):
        overrides = {
            (1, 0): TileType.WATER,
            (2, 1): TileType.ROCK,
            (3, 2): TileType.WATER,
        }
        grid = FakeGrid(overrides)
        start, end = (0, 0), (5, 4)
        path = self.pf.find_shortest_path(grid, start, end)
        assert path is not None
        final, walked = walk(grid, start, path)
        reported = self.pf.shortest_cost(grid, start, end)
        assert final == end
        assert walked == reported

    def test_optimal_path_not_suboptimal(self):
        """Any other valid path to the same destination costs >= shortest_cost."""
        grid = FakeGrid(overrides={(1, 0): TileType.WATER})
        start, end = (0, 0), (2, 0)
        best = self.pf.shortest_cost(grid, start, end)
        # Manually compute the two candidate paths
        direct_cost = grid.move_cost(1, 0) + grid.move_cost(2, 0)   # via water: 3
        detour_cost = grid.move_cost(0, 1) + grid.move_cost(1, 1) + grid.move_cost(2, 1) + grid.move_cost(2, 0)  # 4
        assert best <= direct_cost
        assert best <= detour_cost
