# AGENTS — Dungeon POC

Ce fichier décrit les agents Claude Code disponibles dans ce projet.
Chaque agent est déclenché via le skill correspondant dans Claude Code (VSCode ou CLI).

---

## Agent : generate-game
**Skill** : `/generate-game`
**Fichiers produits** : `src/` (7 fichiers)

Génère le code source complet du jeu selon la spécification dans CLAUDE.md.

Structure de fichiers produite :
```
src/
├── directions.py     ← constante DIRECTIONS partagée (LEFT/RIGHT/UP/DOWN → dx,dy)
├── grid.py           ← TileType (GRASS/ROCK/WATER), Grid (30% ROCK, 20% WATER)
├── pathfinder.py     ← PathFinder : Dijkstra pondéré (herbe=1, eau=2)
├── game_state.py     ← GameState : déplacements, score 0-100, trail, optimal_path
├── simulation.py     ← Simulation (thread, modes UI et headless)
├── ui.py             ← GameUI : rendu pygame, trails jaune/rouge, HUD
└── main.py           ← point d'entrée : python src/main.py
```

Fonctionnalités couvertes :
- Grille pygame 10×10, cases 40px, séparateurs 1px, fond noir
- Herbe/Pierre/Eau avec répartition aléatoire et seed reproductible
- Personnage et sortie placés aléatoirement sur cases herbe
- Restart automatique si Dijkstra ne trouve aucun chemin (`create_solvable`)
- Trail jaune = chemin du joueur ; trail rouge (+2px) = chemin optimal, visible en fin de partie
- Score 0-100 : `round(100 × optimal_cost / player_cost)`, 100 = chemin optimal
- Interface : labels déplacements/note/information, input "instruct" (flèches), boutons start/restart
- Simulation : pause 0.5s par pas, détection victoire/défaite

Contraintes imposées à l'agent :
- `directions.py`, `grid.py`, `game_state.py`, `pathfinder.py`, `simulation.py` : aucun import pygame
- `directions.DIRECTIONS` est la seule définition du mapping direction→delta ; ne pas redéfinir localement
- `GameState` appelle `PathFinder().find_shortest_path()` **une seule fois** à l'init pour dériver `_optimal_cost` ET `optimal_path`
- Point d'entrée unique : `python src/main.py`

---

## Agent : generate-tests
**Skill** : `/generate-tests`
**Fichiers produits** : `tests/helpers.py`, `tests/test_game.py`, `tests/test_pathfinder.py`
**Prérequis** : `src/` doit exister

Génère les tests unitaires pytest couvrant :

**`tests/helpers.py`** — utilitaires partagés
- `FakeGrid` : grille tout-herbe 10×10 avec overrides par cellule (ne pas redéfinir dans les fichiers de test)

**`tests/test_game.py`** — logique de jeu
- `TileType` : valeurs enum, membres distincts
- `Grid` : proportions tuiles, seed déterministe, index, accesseurs, passabilité, coût
- `GameState` : init, déplacements (4 directions, bordures, roches, eau), accumulation coût
- Condition de victoire : score 0-100, formule exacte, eau prise en compte
- `trail` : initialisé avec char_pos, alimenté à chaque déplacement réussi, inchangé si bloqué
- `optimal_path` : positions du chemin optimal, None si inaccessible, cohérent avec `_optimal_cost`
- `is_solvable()` et `create_solvable()` : retry jusqu'à obtenir un terrain jouable
- `Simulation` : parse flèches, headless, queue repaint, stop event, condition PERDU

**`tests/test_pathfinder.py`** — PathFinder Dijkstra
- start == end, chemins de base (4 directions, coin à coin)
- Coût eau (2) vs herbe (1), préférence du détour herbe sur corridor eau
- Rochers bloquants, aucun chemin (None), couloir unique
- Cohérence coût rapporté vs chemin simulé

Contraintes :
- `pytest` uniquement, aucun import pygame
- `from helpers import FakeGrid` — ne jamais redéfinir FakeGrid dans un fichier de test
- `from directions import DIRECTIONS` — ne jamais redéfinir un mapping direction→delta local
- `make_state()` dans test_game.py doit recalculer `_optimal_cost`, `optimal_path` ET `trail` après avoir forcé `char_pos`/`exit_pos`

---

## Agent : run-tests
**Skill** : `/run-tests`
**Prérequis** : tests générés, dépendances installées (`.venv`)

Lance `.venv\Scripts\python.exe -m pytest tests/ -v` et rapporte :
- Nombre de tests passés / échoués (cible : 170+)
- En cas d'échec : résumé des assertions qui ont échoué et fichier concerné

---

## Agent : run-game
**Skill** : `/run-game`
**Prérequis** : `src/main.py` doit exister, `pygame` installé dans `.venv`

Démarre le jeu via `.venv\Scripts\python.exe src/main.py`.

---

## Agent : build-exe
**Skill** : `/build-exe`
**Produit** : `dist/dungeon.exe`
**Prérequis** : `src/main.py` doit exister, `pyinstaller` installé dans `.venv`

Génère un exécutable Windows autonome via PyInstaller :
- `--onefile` : fichier unique
- `--noconsole` : pas de fenêtre console
- `--name dungeon`
- Point d'entrée : `src/main.py`
- Inclut tous les modules `src/` dans le bundle

---

## Agents à venir — Roadmap RL

### Agent : dungeon-env *(Phase 1)*
**Fichier produit** : `src/dungeon_env.py`

Implémentera l'interface Gym (`reset` / `step`) pour l'entraînement RL :
```python
env = DungeonEnv(seed=42)           # terrain reproductible
env = DungeonEnv(seed=None)         # terrain aléatoire à chaque reset()
env = DungeonEnv(seed_pool=[...])   # tirage dans un ensemble fixe
obs              = env.reset()      # → dict {grid, char_pos, exit_pos}
obs, reward, done, info = env.step("RIGHT")
# reward = score / 100  (1.0 = chemin optimal)
```

### Agent : train-rl *(Phase 2)*
**Fichiers produits** : `src/train.py`, `models/`, `logs/`

Boucle d'entraînement headless avec `multiprocessing.Pool`.
Modèle : MLP PyTorch (304 → 128 → 64 → 4), algorithme DQN puis PPO.
Logs JSON : `{"grid_seed": X, "state_seed": X, "moves": [...], "score": 87}`.

### Agent : replay-model *(Phase 3)*
Charge un checkpoint `.pt` et rejoue la partie dans pygame via seed + séquence générée par le modèle.
Lancement prévu : `python src/main.py --seed 42 --replay logs/episode_xxx.jsonl`
