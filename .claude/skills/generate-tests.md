Generate comprehensive pytest unit tests for the dungeon game.

Read `src/game.py` to understand the exact class/method signatures, then create `tests/test_game.py`.

## Requirements

Test only pure-logic classes (Grid, GameState). Do NOT import pygame in tests — if needed, mock `pygame` at module level with `unittest.mock`.

### Grid tests
- `test_grid_tile_counts`: verify ~30% ROCK, ~20% WATER (±5% tolerance for randomness, run 10 grid instances)
- `test_grid_rock_impassable`: ROCK tiles must return False for passability
- `test_grid_water_passable`: WATER tiles must return True for passability and cost 2
- `test_grid_grass_passable`: GRASS tiles must return True and cost 1
- `test_character_on_grass`: initial character position is always GRASS
- `test_exit_on_grass`: exit position is always GRASS
- `test_character_exit_different`: character and exit start on different cells

### GameState / movement tests
- `test_move_into_rock_blocked`: moving into a ROCK cell leaves position unchanged, déplacements unchanged
- `test_move_into_grass`: position updates, déplacements +1
- `test_move_into_water`: position updates, déplacements +2
- `test_move_out_of_bounds`: moving off grid edges leaves position unchanged
- `test_win_condition`: when character moves onto exit cell → score=1, info="GAGNE"
- `test_no_move_after_win`: further moves after win have no effect
- `test_lose_condition`: sequence ends without reaching exit → score=0, info="PERDU"
- `test_no_perdu_when_stopped`: manually stopped simulation does NOT set "PERDU"
- `test_no_perdu_when_won`: won simulation does NOT set "PERDU"
- `test_deplacement_counter`: counter accumulates correctly over a sequence of moves

### Restart test
- `test_restart_resets_state`: after win, restart → score=0, info="", déplacements=0, new grid generated

## Constraints
- Use pytest fixtures for grid setup
- Each test function is independent (no shared mutable state)
- Seed random where needed to make tests deterministic
- Create `tests/__init__.py` (empty) so pytest discovers the package
- Create the `tests/` directory if it does not exist

After writing the files, verify with `python -m pytest tests/ --collect-only` and report collected test count.
