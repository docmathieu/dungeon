Generate `src/dungeon_env.py` — the Gym-compatible interface wrapping GameState for RL training.

## Purpose

`DungeonEnv` is a translation layer between the game (`GameState`, `Grid`) and any RL algorithm.
It exposes only two methods (`reset`, `step`) so the training code never touches game internals.

## Three seed modes

```python
DungeonEnv(seed=42)           # same grid every reset()
DungeonEnv(seed=None)         # new random grid every reset()
DungeonEnv(seed_pool=[...])   # random draw from a fixed pool each reset()
```

`seed_pool=[]` must raise `ValueError`.

## reset() → dict

Calls `GameState.create_solvable(seed)`, resets `_steps` to 0, returns `_observe()`.

## step(action) → (dict, float, bool, dict)

1. Calls `self._state.apply_move(action)` (case-insensitive, accepts "LEFT"/"RIGHT"/"UP"/"DOWN")
2. Increments `self._steps`
3. Computes:
   - `won  = self._state.won`
   - `done = won or self._steps >= self._max_steps`
   - `reward = self._state.score / 100.0 if won else 0.0`
4. Returns `(_observe(), reward, done, info)`

`info` dict keys: `score`, `moves`, `steps`, `won`.

Raises `RuntimeError` if called before `reset()`.

## _observe() → dict

```python
{
    "grid":     list[int],            # 100 values, row-major (y outer, x inner)
                                      # 0=GRASS, 1=ROCK, 2=WATER
    "char_pos": tuple[int, int],      # (x, y)
    "exit_pos": tuple[int, int],      # (x, y)
}
```

## action_space property

Returns the constant `ACTIONS = ("LEFT", "RIGHT", "UP", "DOWN")`.

## Constraints

- No pygame import
- No numpy dependency (plain Python int list — training layer converts to tensor)
- `max_steps` defaults to `DungeonEnv.MAX_STEPS = 100`
- `_pick_seed()` private method handles the three seed modes

## Tests

After writing the file, verify with:
```
python -m pytest tests/test_dungeon_env.py -v
```
Expected: 43 tests passing.
