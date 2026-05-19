Start the dungeon game.

## Steps

1. Check that `.venv` exists. If not, run:
   ```
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```

2. Check that `src/main.py` exists. If not, invoke the `/generate-game` skill first.

3. Launch the game:
   ```
   .venv\Scripts\python src/main.py
   ```

4. The pygame window should open. Inform the user:
   - Use arrow keys in the "instruct" field to enter a movement sequence
   - Click "Start" or press Enter to run the simulation
   - Click "Restart" to reset the game
