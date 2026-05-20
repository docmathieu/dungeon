from grid import TileType


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

    def grass_cells(self) -> list:
        return [
            (x, y)
            for y in range(self.HEIGHT)
            for x in range(self.WIDTH)
            if self.get_tile(x, y) == TileType.GRASS
        ]
