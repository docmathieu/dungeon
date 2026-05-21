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

**`train(episodes, seed, seed_pool, log_path, model_dir, verbose) → DQNAgent`**
- Creates `DungeonEnv`, `DQNAgent`, `ReplayBuffer`
- Per episode: reset → loop(select_action → step → push → learn) → decay_epsilon
- Every `TARGET_UPDATE_FREQ` episodes: sync_target
- Every `CHECKPOINT_FREQ` episodes: save `models/dqn_ep{N}.pt`
- Per episode: write JSON line to log_path
- Final: save `models/dqn_final.pt`

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

## CLI entry point
```bash
python src/train.py --episodes 5000 --seed 42
python src/train.py --seed-pool 0,1,2,3
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
