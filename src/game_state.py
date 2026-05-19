import random
from grid import Grid


DIRECTION_DELTA: dict[str, tuple[int, int]] = {
    "LEFT":  (-1,  0),
    "RIGHT": ( 1,  0),
    "UP":    ( 0, -1),
    "DOWN":  ( 0,  1),
}


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

    def apply_move(self, direction: str) -> None:
        if self.won:
            return

        delta = DIRECTION_DELTA.get(direction.upper())
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

        if self.char_pos == self.exit_pos:
            self.score = 1
            self.info = "GAGNE"
            self.won = True
