import heapq

DIRECTIONS: dict[str, tuple[int, int]] = {
    "UP":    ( 0, -1),
    "DOWN":  ( 0,  1),
    "LEFT":  (-1,  0),
    "RIGHT": ( 1,  0),
}


class PathFinder:
    """Dijkstra shortest-path finder for the dungeon grid.

    Works with any object that exposes WIDTH, HEIGHT, is_passable(x,y),
    and move_cost(x,y) — including FakeGrid in tests.
    """

    def find_shortest_path(
        self,
        grid,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> list[str] | None:
        """Return the cheapest move sequence from start to end, or None if unreachable."""
        result = self._dijkstra(grid, start, end)
        return result[1] if result is not None else None

    def shortest_cost(
        self,
        grid,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> int | None:
        """Return the minimum move cost from start to end, or None if unreachable."""
        result = self._dijkstra(grid, start, end)
        return result[0] if result is not None else None

    def _dijkstra(
        self,
        grid,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> tuple[int, list[str]] | None:
        if start == end:
            return (0, [])

        # heap entries: (cumulative_cost, x, y, path_so_far)
        heap: list = [(0, start[0], start[1], [])]
        visited: set[tuple[int, int]] = set()

        while heap:
            cost, x, y, path = heapq.heappop(heap)

            if (x, y) in visited:
                continue
            visited.add((x, y))

            if (x, y) == end:
                return (cost, path)

            for direction, (dx, dy) in DIRECTIONS.items():
                nx, ny = x + dx, y + dy
                if not (0 <= nx < grid.WIDTH and 0 <= ny < grid.HEIGHT):
                    continue
                if not grid.is_passable(nx, ny):
                    continue
                if (nx, ny) in visited:
                    continue
                new_cost = cost + grid.move_cost(nx, ny)
                heapq.heappush(heap, (new_cost, nx, ny, path + [direction]))

        return None
