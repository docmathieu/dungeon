Generate `src/model.py` and `src/train.py` — DQN training loop for DungeonEnv.

## Files to produce

### `src/model.py`
- `INPUT_DIM = 304`, `HIDDEN1 = 128`, `HIDDEN2 = 64`, `OUTPUT_DIM = 4`
- `DQNetwork(nn.Module)`: `nn.Sequential(Linear→ReLU→Linear→ReLU→Linear)`
- `forward(x)` accepts shape `(input_dim,)` or `(batch, input_dim)`

### `src/train.py`

**`encode_obs(obs) → torch.Tensor`** — shape (304,), float32
- one-hot grid: for each of 100 tiles, set `one_hot[i*3 + tile_value] = 1.0` → 300 floats
- char_pos: `(x/9, y/9)` → 2 floats
- exit_pos: `(x/9, y/9)` → 2 floats

**`ReplayBuffer(capacity)`**
- Internal `collections.deque(maxlen=capacity)`
- `push(state, action, reward, next_state, done)`
- `sample(batch_size) → list[Transition]` — raises `ValueError` if buffer too small
- `__len__`, `capacity` property

**`DQNAgent`**
- Two `DQNetwork` instances: `q_net` (trained) and `target_net` (frozen copy)
- `select_action(state) → int` — epsilon-greedy
- `learn(buffer, batch_size) → float | None` — Bellman MSE loss, returns None if buffer too small
- `decay_epsilon()` — `epsilon = max(eps_end, epsilon * eps_decay)`
- `sync_target()` — copies q_net weights to target_net

**`_now() → str`** — timestamp `yyyymmdd_hhmm`
**`_run_label(seed, seed_pool) → str`** — `seed42` / `pool10` / `random`
**`_run_name(timestamp, episodes, seed, seed_pool) → str`** — identifiant unique du run

**`train(episodes, seed, seed_pool, log_path, model_dir, verbose) → DQNAgent`**
- Creates `DungeonEnv`, `DQNAgent`, `ReplayBuffer`
- Per episode: `_run_episode` → `_log_episode` → `_save_checkpoint` (every 500 ep)
- Every `TARGET_UPDATE_FREQ` episodes: sync_target
- Final: `_save_checkpoint(..., final=True)` → `final.pt`

## Hyperparameters (module-level constants)
```
EPISODES=5000, BATCH_SIZE=64, LEARNING_RATE=1e-3, GAMMA=0.99
EPSILON_START=1.0, EPSILON_END=0.05, EPSILON_DECAY=0.995
BUFFER_SIZE=10_000, TARGET_UPDATE_FREQ=100, CHECKPOINT_FREQ=500
```

## Log format (one JSON line per episode)
```json
{"episode": 1, "score": 0, "moves": ["LEFT", "UP", ...], "epsilon": 0.995, "reward": 0.0}
```

## Output naming convention
- Log  : `logs/yyyymmdd_hhmm_{label}_ep{N}[_from_{timestamp}].jsonl`
- Models: `models/yyyymmdd_hhmm_{label}_ep{N}[_from_{timestamp}]/ep500.pt … final.pt`
- `{label}` = `seed42` | `pool10` | `random`
- `_from_{timestamp}` présent uniquement si `--pretrained` est fourni

## CLI entry point
```bash
python src/train.py --episodes 2000 --seed 42
python src/train.py --episodes 2000 --seed-pool 0,1,2,3 --lr 3e-4
python src/train.py --episodes 2000 --seed-pool 0,1,2,3 --lr 3e-4 \
                    --pretrained models/20260527_1222_seed42_ep2000/final.pt
```

## Constraints
- No pygame import in model.py or train.py
- `sys.path.insert(0, os.path.dirname(__file__))` at top of train.py for script execution
- `if __name__ == "__main__"` guard for CLI

## Tests
```
python -m pytest tests/test_train.py -v
```
Expected: 35 tests passing.

---

# Curriculum learning — `src/curriculum.py`

Progressive pool expansion: train on 1 seed until mastered, then widen the pool.

## CLI entry point
```bash
python src/curriculum.py --pool 0,1,2,3,4,5,6,7,8,9 \
                         --stages 1,3,6,10 \
                         --max-episodes-per-stage 2000 \
                         --win-rate-threshold 0.8 \
                         --lr 3e-4
```

## Key functions
- `_win_rate(scores, window=100) → float` — fraction of wins in the last `window` episodes (score > 0 = win)
- `_train_stage(...) → Path` — trains until win rate ≥ threshold or max_episodes; returns `model_dir/final.pt`
- `run_curriculum(...) → Path` — iterates stages, passes `final.pt` as pretrained for the next stage

## Stopping criterion per stage
- **Early stop**: if `ep >= WIN_RATE_WINDOW (100)` and `_win_rate(scores) >= win_rate_threshold` → move to next stage
- **Timeout**: if `max_episodes_per_stage` exhausted without reaching threshold → still proceed to next stage

## Output naming
Each stage produces:
- `logs/yyyymmdd_hhmm_{label}_ep{N}[_from_{timestamp}].jsonl`
- `models/yyyymmdd_hhmm_{label}_ep{N}[_from_{timestamp}]/final.pt`

`{label}` = `seed42` (1 seed) or `pool3` (multiple seeds). `_from_{timestamp}` present from stage 2 onward.

## Tests
```
python -m pytest tests/test_curriculum.py -v
```
Expected: 22 tests passing.
