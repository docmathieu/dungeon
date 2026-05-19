import enum
import random


class TileType(enum.Enum):
    GRASS = "grass"
    ROCK = "rock"
    WATER = "water"


class Grid:
    WIDTH = 10
    HEIGHT = 10

    def __init__(self, seed: int | None = None):
        rng = random.Random(seed)
        total = self.WIDTH * self.HEIGHT  # 100
        cells = [TileType.GRASS] * total

        rock_count = int(total * 0.30)
        water_count = int(total * 0.20)

        indices = list(range(total))
        rng.shuffle(indices)
        for i in indices[:rock_count]:
            cells[i] = TileType.ROCK
        for i in indices[rock_count: rock_count + water_count]:
            cells[i] = TileType.WATER

        self._cells = cells

    def _index(self, x: int, y: int) -> int:
        return y * self.WIDTH + x

    def get_tile(self, x: int, y: int) -> TileType:
        return self._cells[self._index(x, y)]

    def is_passable(self, x: int, y: int) -> bool:
        return self.get_tile(x, y) != TileType.ROCK

    def move_cost(self, x: int, y: int) -> int:
        return 2 if self.get_tile(x, y) == TileType.WATER else 1

    def grass_cells(self) -> list[tuple[int, int]]:
        return [
            (x, y)
            for y in range(self.HEIGHT)
            for x in range(self.WIDTH)
            if self.get_tile(x, y) == TileType.GRASS
        ]
