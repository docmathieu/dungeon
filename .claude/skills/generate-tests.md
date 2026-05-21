Generate comprehensive pytest unit tests for the dungeon game.

Read `src/game.py` to understand the exact class/method signatures, then create `tests/test_game.py`.

## Requirements

Test only pure-logic classes (Grid, GameState). Do NOT import pygame in tests — if needed, mock `pygame` at module level with `unittest.mock`.

## Shared test utilities (`tests/helpers.py`)
- `FakeGrid`: all-grass 10×10 grid with per-cell overrides. Implements `get_tile`, `is_passable`, `move_cost`, `grass_cells`. Import it with `from helpers import FakeGrid`; do NOT redefine it in individual test files.
- `DIRECTIONS` is available via `from directions import DIRECTIONS`; do NOT define a local `DELTA` or similar constant.

### Grid tests
- `test_grid_tile_counts`: verify ~30% ROCK, ~20% WATER (±5% tolerance for randomness, run 10 grid instances)
- `test_grid_rock_impassable`: ROCK tiles must return False for passability
- `test_grid_water_passable`: WATER tiles must return True for passability and cost 2
- `test_grid_grass_passable`: GRASS tiles must return True and cost 1
- `test_character_on_grass`: initial character position is always GRASS
- `test_exit_on_grass`: exit position is always GRASS
- `test_character_exit_different`: character and exit start on different cells

### GameState / movement tests
- `test_move_into_rock_blocked`: moving into a ROCK cell leaves position unchanged, déplacements +1 (penalty)
- `test_multiple_bumps_same_rock_accumulate`: hitting the same rock 3 times → move_count == 3
- `test_rock_bump_does_not_change_trail`: bumping a rock must not append to `trail`
- `test_boundary_costs_one_like_rock`: out-of-bounds move costs 1, same penalty as hitting a rock
- `test_multiple_boundary_bumps_accumulate`: hitting the same wall 3 times → move_count == 3
- `test_boundary_bump_does_not_change_trail`: hitting boundary must not append to `trail`
- `test_rock_bump_costs_one_in_simulation`: Simulation applies rock bump cost (move_count +1, pos unchanged)
- `test_boundary_bump_costs_one_in_simulation`: Simulation applies boundary cost (move_count +1, pos unchanged)
- `test_rock_bump_reduces_win_score`: bumping a rock before reaching exit lowers the score (known formula)
- `test_boundary_bump_reduces_win_score`: hitting boundary before reaching exit lowers the score (known formula)
- `test_move_into_grass`: position updates, déplacements +1
- `test_move_into_water`: position updates, déplacements +2
- `test_move_out_of_bounds`: moving off grid edges leaves position unchanged, déplacements +1 (same penalty as rock)
- `test_win_condition`: when character moves onto exit cell → score=100 (if optimal), info="GAGNE"
- `test_no_move_after_win`: further moves after win have no effect
- `test_lose_condition`: sequence ends without reaching exit → score=0, info="PERDU"
- `test_no_perdu_when_stopped`: manually stopped simulation does NOT set "PERDU"
- `test_no_perdu_when_won`: won simulation does NOT set "PERDU"
- `test_deplacement_counter`: counter accumulates correctly over a sequence of moves

### Win scoring tests (0–100)
- `test_optimal_path_scores_100`: player cost == optimal cost → score 100
- `test_suboptimal_path_scores_less_than_100`: extra moves before exit → score < 100
- `test_score_formula_exact`: verify `round(100 × optimal / player)` with known values
- `test_score_is_positive_when_winning`: any winning path yields score > 0
- `test_score_never_exceeds_100`: score is always ≤ 100
- `test_score_uses_water_cost_in_optimal`: water cost (2) included correctly in formula
- `test_score_falls_back_to_100_when_optimal_cost_is_none`: edge case where PathFinder returns None
- `test_score_zero_when_lost`: losing keeps score at 0
- `test_score_not_set_before_win`: score stays 0 until exit is reached

### Optimal path tests
- `test_optimal_path_starts_at_char_pos`: `optimal_path[0] == char_pos`
- `test_optimal_path_ends_at_exit_pos`: `optimal_path[-1] == exit_pos`
- `test_optimal_path_is_none_when_unreachable`: enclosed start → `None`
- `test_optimal_path_length_on_all_grass`: length == Manhattan distance + 1
- `test_optimal_path_direct_one_step`: single-step path verified exactly
- `test_optimal_path_each_position_adjacent_to_next`: consecutive positions are 1 Manhattan step apart
- `test_optimal_path_cost_matches_optimal_cost`: walked cost == `_optimal_cost`
- `test_optimal_path_avoids_rocks`: no rock cell appears in the path
- `test_optimal_path_init_uses_single_pathfinder_call`: `find_shortest_path` called exactly once at init

### Trail tests
- `test_trail_initialized_with_char_pos`: trail starts as `[char_pos]`
- `test_trail_appended_on_successful_move`: each passable move adds the new position
- `test_trail_not_appended_on_rock_block`: blocked move leaves trail unchanged
- `test_trail_not_appended_on_boundary`: out-of-bounds move leaves trail unchanged
- `test_trail_not_appended_after_win`: moves after win don't modify trail
- `test_trail_length_equals_successful_moves_plus_one`: `len(trail) == moves + 1`
- `test_trail_records_path_correctly`: verifies exact sequence of positions
- `test_trail_can_revisit_positions`: going back and forth appends duplicate positions
- `test_trail_includes_exit_as_last_entry_on_win`: exit is `trail[-1]` after winning
- `test_trail_water_tile_appended_once`: water tile appended as single position (cost is separate)

### Solvability tests
- `test_is_solvable_true_when_path_exists`: open grid → True
- `test_is_solvable_false_when_enclosed_by_rocks`: start surrounded by rocks/boundary → False
- `test_is_solvable_false_when_exit_enclosed`: exit surrounded → False
- `test_create_solvable_returns_solvable_state`: result always has `is_solvable() == True`
- `test_create_solvable_retries_when_first_grid_unsolvable`: mock PathFinder to return None then a value; verify retry occurred
- `test_create_solvable_seed_incremented_on_retry`: with fixed seed, verify Grid is called with seed, then seed+1
- `test_create_solvable_without_seed_retries_randomly`: seed=None path retries correctly

### PathFinder (Dijkstra) tests
- `test_start_equals_end`: returns empty list and cost 0
- `test_one_step_*`: one-step paths in all 4 directions
- `test_path_leads_to_destination`: walked path lands on expected cell
- `test_cost_equals_manhattan_distance_on_all_grass`: on uniform grid, cost = |dx|+|dy|
- `test_single_water_cell_costs_two`: water cell adds cost 2
- `test_prefers_grass_detour_over_water_corridor`: Dijkstra picks cheaper multi-step grass route over expensive water shortcut
- `test_no_path_from_enclosed_corner`: returns None when start is surrounded by rocks/boundary
- `test_corridor_only_valid_route`: finds only path through a 1-cell-wide corridor
- `test_walked_cost_matches_reported_cost`: `find_shortest_path` and `shortest_cost` are consistent

### Restart test
- `test_restart_resets_state`: after win, restart → score=0, info="", déplacements=0, new grid generated

## Constraints
- Use pytest fixtures for grid setup
- Each test function is independent (no shared mutable state)
- Seed random where needed to make tests deterministic
- Create `tests/__init__.py` (empty) so pytest discovers the package
- Create the `tests/` directory if it does not exist

After writing the files, verify with `python -m pytest tests/ --collect-only` and report collected test count.
