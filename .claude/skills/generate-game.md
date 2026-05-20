Generate the complete dungeon game source code.

Read CLAUDE.md for the full specification, then create `src/game.py` with the following requirements:

## Architecture
Produce the following files under `src/`:

**`src/directions.py`** — no pygame import
- Module-level constant `DIRECTIONS: dict[str, tuple[int,int]]` mapping `"LEFT"/"RIGHT"/"UP"/"DOWN"` to `(dx, dy)`.
- Imported by `game_state.py` and `pathfinder.py`; do NOT redefine locally in either.

**`src/grid.py`** — no pygame import
- `TileType` (enum): GRASS, ROCK, WATER
- `Grid`: generates the 10×10 tile map (30% ROCK, 20% WATER, rest GRASS). Methods: `get_tile(x,y)`, `is_passable(x,y)`, `move_cost(x,y)`.

**`src/pathfinder.py`** — no pygame import
- `PathFinder`: Dijkstra weighted shortest-path finder. Methods:
  - `find_shortest_path(grid, start, end) -> list[str] | None` — cheapest move sequence or None if unreachable.
  - `shortest_cost(grid, start, end) -> int | None` — minimum cost (grass=1, water=2) or None.
- Works with any grid exposing `WIDTH`, `HEIGHT`, `is_passable(x,y)`, `move_cost(x,y)`.

**`src/game_state.py`** — no pygame import
- `GameState`: holds character position, exit position, move count, score (0–100), info message. Method: `apply_move(direction: str)`. Pure logic, no side effects on display.
- At init, precomputes `_optimal_cost` via `PathFinder` for the current char/exit positions.
- At init, initialises `trail: list[tuple[int,int]] = [char_pos]`. Each successful `apply_move` appends the new position; blocked and out-of-bounds moves do not append.
- On win: `score = round(100 × optimal_cost / move_count)` (100 = optimal, <100 = suboptimal, 0 = lost). Falls back to 100 if no optimal path is found.
- `is_solvable() -> bool`: returns True if `_optimal_cost is not None` (path exists from char to exit).
- `create_solvable(seed=None) -> GameState` (classmethod): generates Grid+GameState pairs until `is_solvable()` is True. With a fixed seed, increments it on each retry for determinism.

**`src/simulation.py`** — no pygame import
- `Simulation`: takes a `GameState`, an instruction string, and an optional `queue.Queue` (default `None`).
- Runs as a `threading.Thread`.
- If `queue` is provided (UI mode): pauses 0.3s between moves, puts a repaint signal in the queue after each move.
- If `queue` is `None` (headless mode): no pause, no queue — loops at maximum speed. This mode is intended for future reinforcement-learning workloads where hundreds of `Simulation` instances run via `multiprocessing.Pool` without any display.
- Stops on win or end of sequence in both modes.

**`src/ui.py`** — pygame
- `GameUI`: creates the pygame window, renders grid and HUD, runs the event loop. Reads signals from the queue on each frame tick to update display without blocking.

**`src/main.py`** — entry point
- Instantiates Grid, GameState, GameUI and starts the event loop.

## Visual spec
- Window background: black (0,0,0)
- Grid: 10 columns × 10 rows, each cell 10×10 px, 1px separator lines between cells
- GRASS: green (34,139,34)
- ROCK: grey (128,128,128)
- WATER: blue (30,144,255)
- Trail: yellow line segments (2 px) connecting the centers of consecutive positions in `GameState.trail`; drawn after the grid and before the character so the character renders on top
- Character: small yellow circle or stick figure drawn with pygame.draw primitives, centered in its cell
- Exit: small yellow rectangle/door drawn with pygame.draw primitives, centered in its cell

## HUD layout (above grid)
- Label + value "Déplacements : 0"
- Label + value "Note : 0"
- Label + value "Information : "
All in white text, pygame.font

## Controls layout (below grid)
- Button "Restart"
- Text input field "instruct" (captures arrow key presses → appends ←↑→↓ characters, unicode ←↑→↓)
- Button "Start"

## Simulation rules
- Arrow keys fill the instruct field (left=←, up=↑, right=→, down=↓)
- Start button or Enter triggers simulation
- Per move: check target cell; ROCK → skip (no move, no cost); WATER → passable, costs 2 moves; boundary → skip
- Increment "Déplacements" by the cost on each attempted move that succeeds
- Win condition: character lands on exit cell → Note=round(100×optimal/player_cost), Information="GAGNE", stop simulation (Note=100 means optimal path)
- Lose condition: sequence ends without reaching exit → Note=0, Information="PERDU" (only if simulation was not manually stopped)
- Restart button: call `GameState.create_solvable()` (retries until a path exists), reset all fields

## Constraints
- Python 3.12, pygame only (no tkinter, no other GUI lib)
- All pygame calls must be on the main thread; in UI mode the Simulation thread communicates via a thread-safe queue
- In headless mode (queue=None), Simulation has zero pygame dependency — safe to instantiate from multiprocessing workers
- Entry point: `python src/main.py`
- Create the `src/` directory if it does not exist

After writing the file, verify it parses with `python -m py_compile src/game.py` and report any errors.
