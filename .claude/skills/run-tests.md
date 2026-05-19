Run the dungeon game unit tests and report results.

## Steps

1. Check that `.venv` exists. If not, run:
   ```
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```

2. Run tests:
   ```
   .venv\Scripts\pytest tests/ -v
   ```

3. If `pytest-cov` is in requirements.txt, also run:
   ```
   .venv\Scripts\pytest tests/ -v --cov=src --cov-report=term-missing
   ```

4. Report:
   - Total tests collected
   - Passed / Failed / Errors
   - Coverage percentage (if available)
   - For each failure: test name, assertion that failed, line number

5. If any test fails, suggest the minimal fix needed in `src/game.py` or `tests/test_game.py`.
