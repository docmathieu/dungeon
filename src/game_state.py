import random
from directions import DIRECTIONS
from grid import Grid
from pathfinder import PathFinder


class GameState:
    def __init__(self, grid: Grid, seed: int | None = None):
        self.grid = grid
        rng = random.Random(seed)

        grass = grid.grass_cells()
        if len(grass) < 2:
            raise ValueError("Not enough grass cells to place character and exit")

        positions = rng.sample(grass, 2)
        self.char_pos: tuple[int, int] = positions[0]
        self.exit_pos: tuple[int, int] = positions[1]

        self.move_count: int = 0
        self.score: int = 0
        self.info: str = ""
        self.won: bool = False

        self._optimal_cost: int | None = PathFinder().shortest_cost(
            grid, self.char_pos, self.exit_pos
        )
        self.trail: list[tuple[int, int]] = [self.char_pos]

    def is_solvable(self) -> bool:
        """Return True if a path exists from character position to exit."""
        return self._optimal_cost is not None

    @classmethod
    def create_solvable(cls, seed: int | None = None) -> "GameState":
        """Return a GameState guaranteed to have a path from character to exit.

        Retries with new grids until one is solvable. When seed is provided,
        it is incremented on each retry for deterministic behaviour.
        """
        while True:
            grid = Grid(seed=seed)
            state = cls(grid, seed=seed)
            if state.is_solvable():
                return state
            if seed is not None:
                seed += 1

    def apply_move(self, direction: str) -> None:
        if self.won:
            return

        delta = DIRECTIONS.get(direction.upper())
        if delta is None:
            return

        nx = self.char_pos[0] + delta[0]
        ny = self.char_pos[1] + delta[1]

        if not (0 <= nx < self.grid.WIDTH and 0 <= ny < self.grid.HEIGHT):
            return
        if not self.grid.is_passable(nx, ny):
            return

        self.move_count += self.grid.move_cost(nx, ny)
        self.char_pos = (nx, ny)
        self.trail.append(self.char_pos)

        if self.char_pos == self.exit_pos:
            if self._optimal_cost is not None:
                self.score = round(100 * self._optimal_cost / self.move_count)
            else:
                self.score = 100
            self.info = "GAGNE"
            self.won = True
