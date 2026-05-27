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
- Choc contre un rocher ou bord de grille : `move_count += 1`, position inchangée, trail inchangé
- Interface : labels déplacements/note/information, légende touches, bouton restart uniquement
- Contrôle : touches fléchées → `apply_move()` direct (pas de thread, pas de queue dans l'UI)

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

## Agent : dungeon-env ✅ *(Phase 1 RL)*
**Skill** : `/dungeon-env`
**Fichier produit** : `src/dungeon_env.py`
**Tests** : `tests/test_dungeon_env.py` (43 tests)

Interface Gym (`reset` / `step`) wrappant `GameState` pour l'entraînement RL :

```python
env = DungeonEnv(seed=42)           # terrain reproductible
env = DungeonEnv(seed=None)         # terrain aléatoire à chaque reset()
env = DungeonEnv(seed_pool=[...])   # tirage dans un ensemble fixe

obs              = env.reset()      # → dict {grid, char_pos, exit_pos}
obs, reward, done, info = env.step("RIGHT")
# reward : victoire → score/100.0 (1.0=optimal)
#          déplacement normal → -0.01
#          choc mur/bord      → -0.05
# done   = True si victoire ou steps >= max_steps (défaut 100)
```

Observation :
- `grid` : list[int] de 100 valeurs (0=HERBE, 1=ROCHE, 2=EAU), ordre ligne-major
- `char_pos` / `exit_pos` : tuple (x, y)

Contraintes :
- Aucun import pygame
- Pas de dépendance numpy (encodage en int Python natif — la couche d'entraînement convertit en tenseur)

---

## Agents à venir — Roadmap RL

### Agent : train-rl ✅ *(Phase 2)*
**Skill** : `/train-rl`
**Fichiers produits** : `src/model.py`, `src/train.py`, `models/`, `logs/`
**Tests** : `tests/test_train.py` (35 tests)

Boucle d'entraînement DQN headless.

```bash
python src/train.py --episodes 5000 --seed-pool 0,1,2,...
```

Composants :
- `DQNetwork` — MLP PyTorch : 304 → Dense(256,ReLU) → Dense(128,ReLU) → Dense(64,ReLU) → 4 sorties
- `ReplayBuffer` — buffer circulaire FIFO, capacité 10 000
- `StratifiedReplayBuffer(capacity, n_seeds)` — un sous-buffer par seed, garantit `batch_size // n_seeds` transitions par seed dans chaque batch
- `DQNAgent` — epsilon-greedy (1.0→0.05), réseau cible synchronisé tous les 100 épisodes
- `encode_obs()` — one-hot grille (300) + positions normalisées (4) = 304 floats
- Logs JSON : `{"episode", "score", "moves", "epsilon", "reward"}` (un par épisode)
- Nommage automatique : `logs/yyyymmdd_hhmm_{label}_ep{N}[_from_{timestamp}].jsonl` et `models/yyyymmdd_hhmm_{label}_ep{N}[_from_{timestamp}]/`
- `{label}` = `seed42` / `pool10` / `random` — checkpoints `ep<N>.pt` tous les 500 épisodes + `final.pt`
- `--pretrained path/final.pt` : repart des poids d'un run précédent (transfer learning)

### Agent : curriculum ✅ *(Phase 2 — curriculum)*
**Skill** : `/curriculum`
**Fichiers produits** : `src/curriculum.py`, `models/`, `logs/`
**Tests** : `tests/test_curriculum.py` (22 tests)

Entraînement progressif par curriculum : élargit le pool de seeds par étapes, en passant à l'étape
suivante quand l'agent atteint le win rate cible ou quand `--max-episodes-per-stage` est épuisé.

```bash
python src/curriculum.py --pool 0,1,2,3,4,5,6,7,8,9 \
                         --stages 1,3,6,10 \
                         --max-episodes-per-stage 2000 \
                         --win-rate-threshold 0.8 \
                         --lr 3e-4,1e-4
```

Composants :
- `_pad_lr(lrs, n)` — normalise la liste de lr à n éléments (répète la dernière valeur ou tronque)
- `_win_rate(scores, window=100)` — taux de victoire sur les N derniers épisodes (score > 0 = victoire)
- `_train_stage(...)` — entraîne jusqu'à maîtrise (win rate ≥ seuil sur 100 ep) ou max_episodes
- `run_curriculum(pool, stages, lr, ...)` — orchestre les étapes, route `lr[stage_idx]` à chaque stage
- Chaque étape produit : `logs/yyyymmdd_hhmm_{label}_ep{N}[_from_{timestamp}].jsonl` + `models/yyyymmdd_hhmm_{label}_ep{N}[_from_{timestamp}]/final.pt`
- Transfer learning automatique : chaque étape repart des poids de l'étape précédente (`_from_` dans le nom)

Paramètres CLI :
- `--pool` (obligatoire) : seeds disponibles, ex. `0,1,2,3,4,5,6,7,8,9`
- `--stages` : seeds par étape (défaut `1,3,6,10`)
- `--max-episodes-per-stage` : plafond épisodes par étape / timeout (défaut `2000`)
- `--win-rate-threshold` : taux de victoire cible pour progresser (défaut `0.8`)
- `--lr` : learning rate(s) par étape, ex. `3e-4,1e-4` — liste de mêmes longueur que `stages` ou plus courte (dernière valeur répétée) (défaut `1e-3`)

### Agent : replay-model *(Phase 3)*
Charge un checkpoint `.pt` et rejoue la partie dans pygame via seed + séquence générée par le modèle.
Lancement prévu : `python src/main.py --seed 42 --replay logs/episode_xxx.jsonl`
